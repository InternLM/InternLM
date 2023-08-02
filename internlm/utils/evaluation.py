from contextlib import contextmanager

import torch
import torch.distributed as dist
from tqdm import tqdm

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.metrics import AccPerplex


@contextmanager
def switch_evaluation_no_pipeline_scheduler(trainer, grad_accum_size, grad_accum_batch_size):
    if not gpc.is_using_pp():
        trainer.schedule.data_process_func = None
        prev_grad_accum_size = trainer.schedule._grad_accum_size
        prev_grad_accum_batch_size = trainer.schedule._grad_accum_batch_size
        try:
            trainer.schedule._grad_accum_size = grad_accum_size
            trainer.schedule._grad_accum_batch_size = grad_accum_batch_size
            yield
        finally:
            trainer.schedule._grad_accum_size = prev_grad_accum_size
            trainer.schedule._grad_accum_batch_size = prev_grad_accum_batch_size


@contextmanager
def switch_evaluation_pipeline_scheduler(trainer, num_microbatches, tensor_shape):
    if gpc.is_using_pp():
        trainer.schedule.data_process_func = None
        prev_num_microbatches = trainer.schedule.num_microbatches
        prev_tensor_shape = trainer.schedule.tensor_shape
        try:
            trainer.schedule.num_microbatches = num_microbatches
            trainer.schedule.tensor_shape = tensor_shape
            yield
        finally:
            trainer.schedule.num_microbatches = prev_num_microbatches
            trainer.schedule.tensor_shape = prev_tensor_shape


def evaluate_on_val_dls(
    trainer,
    val_dls,
    writer,
    logger,
    step_count,
    tokenizer=None,
    update_panel: bool = False,
):
    torch.cuda.empty_cache()
    trainer.eval()
    verbose = gpc.is_rank_for_log()
    data_cfg = gpc.config.data

    for val_name, val_dl in val_dls.items():
        if len(val_dl) == 0 and verbose:
            logger.info(f"Validation dataset: {val_name} is empty")
            continue

        val_metric = AccPerplex(
            device=torch.cuda.current_device(),
            tp_pg=gpc.get_group(ParallelMode.TENSOR),
            dp_pg=gpc.get_group(ParallelMode.DATA),
            tokenizer=tokenizer,
        )
        val_loss = 0
        val_idx = -1
        for val_idx, batch in tqdm(
            enumerate(val_dl),
            desc="Val.",
            total=len(val_dl),
            position=1,
            disable=not verbose,
            leave=False,
        ):
            with torch.inference_mode():
                if gpc.is_using_pp():
                    total_val_bsz = len(batch[1])
                    assert total_val_bsz % data_cfg.micro_bsz == 0
                    num_microbatches = total_val_bsz // data_cfg.micro_bsz
                    tensor_shape = torch.Size(
                        [data_cfg.micro_bsz, batch[0]["input_ids"].shape[1], gpc.config.HIDDEN_SIZE]
                    )

                    with switch_evaluation_pipeline_scheduler(
                        trainer=trainer, num_microbatches=num_microbatches, tensor_shape=tensor_shape
                    ):
                        _, _, loss = trainer.execute_schedule(
                            batch, forward_only=True, return_loss=True, return_output_label=False, post_fn=val_metric
                        )
                else:
                    total_val_bsz = len(batch[1])
                    assert total_val_bsz % data_cfg.micro_bsz == 0
                    grad_accum_size = total_val_bsz // data_cfg.micro_bsz
                    grad_accum_batch_size = data_cfg.micro_bsz

                    with switch_evaluation_no_pipeline_scheduler(
                        trainer=trainer, grad_accum_size=grad_accum_size, grad_accum_batch_size=grad_accum_batch_size
                    ):
                        _, _, loss = trainer.execute_schedule(
                            batch, forward_only=True, return_loss=True, return_output_label=False, post_fn=val_metric
                        )
            if verbose:
                val_loss += loss.item()

        assert val_idx != -1
        dist.barrier()
        val_res = val_metric.get_metric()

        if verbose and len(val_dl) != 0:
            val_loss = val_loss / (val_idx + 1 + 1e-6)
            infos = {
                f"val/{val_name}_loss": val_loss,
                f"val/{val_name}_acc": val_res["acc"],
                f"val/{val_name}_plex": val_res["perplexity"],
            }
            val_metric = {
                "step": step_count,
                "val_loss": val_loss,
                "val_acc": val_res["acc"],
                "val_perplexity": val_res["perplexity"],
            }
            for key, value in infos.items():
                writer.add_scalar(key=key, value=value, step=step_count)
            infos["step"] = step_count
            if update_panel:
                logger.info(
                    f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()]),
                    extra=val_metric,
                )
            else:
                logger.info(
                    f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()])
                )

    trainer.train()
    torch.cuda.empty_cache()
    dist.barrier()

from contextlib import contextmanager

import torch
import torch.distributed as dist
from tqdm import tqdm

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.scheduler import SchedulerMetricHook
from internlm.model.metrics import AccPerplex


@contextmanager
def switch_evaluation_no_pipeline_scheduler(trainer, grad_accum_size, grad_accum_batch_size, metric_hook_list):
    if not gpc.is_using_pp():
        prev_data_process_func = trainer.schedule.data_process_func
        prev_grad_accum_size = trainer.schedule._grad_accum_size
        prev_grad_accum_batch_size = trainer.schedule._grad_accum_batch_size
        prev_metric_hooks = trainer.schedule._hooks
        try:
            trainer.schedule.data_process_func = None
            trainer.schedule._grad_accum_size = grad_accum_size
            trainer.schedule._grad_accum_batch_size = grad_accum_batch_size
            trainer.schedule._hooks = metric_hook_list
            yield
        finally:
            trainer.schedule.data_process_func = prev_data_process_func
            trainer.schedule._grad_accum_size = prev_grad_accum_size
            trainer.schedule._grad_accum_batch_size = prev_grad_accum_batch_size
            trainer.schedule._hooks = prev_metric_hooks


@contextmanager
def switch_evaluation_pipeline_scheduler(trainer, num_microbatches, tensor_shape, metric_hook_list):
    if gpc.is_using_pp():
        pre_data_process_func = trainer.schedule.data_process_func
        prev_num_microbatches = trainer.schedule.num_microbatches
        prev_tensor_shape = trainer.schedule.tensor_shape
        prev_metric_hooks = trainer.schedule._hooks
        try:
            trainer.schedule.data_process_func = None
            trainer.schedule.num_microbatches = num_microbatches
            trainer.schedule.tensor_shape = tensor_shape
            trainer.schedule._hooks = metric_hook_list
            yield
        finally:
            trainer.schedule.data_process_func = pre_data_process_func
            trainer.schedule.num_microbatches = prev_num_microbatches
            trainer.schedule.tensor_shape = prev_tensor_shape
            trainer.schedule._hooks = prev_metric_hooks


@contextmanager
def switch_sequence_parallel_mode():
    prev_mode = gpc.config.parallel.sequence_parallel
    try:
        gpc.config.parallel.sequence_parallel = False
        yield
    finally:
        gpc.config.parallel.sequence_parallel = prev_mode


def evaluate_on_val_dls(
    trainer,
    val_dls,
    writer,
    logger,
    step_count,
    update_panel: bool = False,
    streaming: bool = False,
):
    with switch_sequence_parallel_mode():
        torch.cuda.empty_cache()
        trainer.eval()
        verbose = gpc.is_rank_for_log()
        data_cfg = gpc.config.data

        for val_name, val_dl in val_dls.items():
            if not streaming and len(val_dl) == 0 and verbose:
                logger.info(f"Validation dataset: {val_name} is empty")
                continue

            val_metric = AccPerplex(
                device=torch.cuda.current_device(),
                tp_pg=gpc.get_group(ParallelMode.TENSOR),
                dp_pg=gpc.get_group(ParallelMode.DATA),
            )
            val_sche_metric_hook = SchedulerMetricHook(metric=val_metric)

            val_loss = 0
            val_idx = -1
            for val_idx, batch in tqdm(
                enumerate(val_dl),
                desc="Val.",
                total=len(val_dl) if not streaming else None,
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
                            trainer=trainer,
                            num_microbatches=num_microbatches,
                            tensor_shape=tensor_shape,
                            metric_hook_list=[val_sche_metric_hook],
                        ):
                            _, _, loss = trainer.execute_schedule(
                                batch, forward_only=True, return_loss=True, return_output_label=False
                            )
                    else:
                        total_val_bsz = len(batch[1])
                        assert total_val_bsz % data_cfg.micro_bsz == 0
                        grad_accum_size = total_val_bsz // data_cfg.micro_bsz
                        grad_accum_batch_size = data_cfg.micro_bsz
                        with switch_evaluation_no_pipeline_scheduler(
                            trainer=trainer,
                            grad_accum_size=grad_accum_size,
                            grad_accum_batch_size=grad_accum_batch_size,
                            metric_hook_list=[val_sche_metric_hook],
                        ):
                            _, _, loss = trainer.execute_schedule(
                                batch, forward_only=True, return_loss=True, return_output_label=False
                            )
                if verbose:
                    val_loss += loss.item()

            assert val_idx != -1
            dist.barrier()

            val_res = val_metric.get_metric()
            if verbose and (streaming or len(val_dl) != 0):
                val_loss = val_loss / (val_idx + 1 + 1e-6)
                infos = {
                    "step": step_count,
                    f"val/{val_name}_loss": val_loss,
                    f"val/{val_name}_acc": val_res["acc"],
                    f"val/{val_name}_plex": val_res["perplexity"],
                }

                for key, value in infos.items():
                    writer.add_scalar(key=key, value=value, step=step_count)

                if update_panel:
                    logger.info(
                        f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()]),
                        extra={
                            "step": step_count,
                            "val_loss": val_loss,
                            "val_acc": val_res["acc"],
                            "val_perplexity": val_res["perplexity"],
                        },
                    )
                else:
                    logger.info(
                        f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()])
                    )

        trainer.train()
        torch.cuda.empty_cache()
        dist.barrier()

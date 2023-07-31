import torch
import torch.distributed as dist
from tqdm import tqdm

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.metrics import AccPerplex


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
            val_bsz = len(batch[1])
            assert val_bsz % gpc.config.micro_bsz == 0
            trainer.schedule.num_microbatches = val_bsz // data_cfg.micro_bsz
            trainer.schedule.tensor_shape = torch.Size(
                [data_cfg.micro_bsz, batch[0]["input_ids"].shape[1], gpc.config.HIDDEN_SIZE]
            )

            with torch.inference_mode():
                _, _, loss = trainer.execute_schedule(
                    batch, forward_only=True, return_loss=True, return_output_label=False, post_fn=val_metric
                )
                if verbose:
                    val_loss += loss.item()
            if val_idx > 20:
                break

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
                    msg=(f"Validation on {val_name}:" + ",".join([f"{key}={value}" for key, value in infos.items()])),
                    extra=val_metric,
                )
            else:
                logger.info(f"Validation on {val_name}:" + ",".join([f"{key}={value}" for key, value in infos.items()]))

    trainer.train()
    torch.cuda.empty_cache()
    dist.barrier()


def evaluate_on_val_dls_wo_pp(
    engine,
    val_dls,
    writer,
    logger,
    enable,
    step_count,
    feat_mask=None,
    ffn_mask=None,
    layer_mask=None,
    tokenizer=None,
):
    torch.cuda.empty_cache()
    engine.eval()
    for val_name, val_dl in val_dls.items():
        val_metric = AccPerplex(
            device=torch.cuda.current_device(),
            tp_pg=gpc.get_group(ParallelMode.PARALLEL_1D),
            dp_pg=gpc.get_group(ParallelMode.DATA),
            tokenizer=tokenizer,
        )
        val_loss, val_idx = 0, -1
        for val_idx, batch in tqdm(
            enumerate(val_dl), desc="Val.", total=len(val_dl), position=1, disable=not enable, leave=False
        ):
            val_bsz = len(batch[1])

            if feat_mask is not None:
                batch[0]["feat_mask"] = feat_mask[:1].repeat(val_bsz, 1, 1)
                batch[0]["ffn_mask"] = ffn_mask[:1].repeat(val_bsz, 1, 1)
            if layer_mask is not None:
                batch[0]["layer_mask"] = layer_mask[:1].repeat(val_bsz, 1)

            with torch.inference_mode():
                _, _, loss = engine.execute_schedule(
                    batch, forward_only=True, return_loss=True, return_output_label=False, post_fn=val_metric
                )
                if enable:
                    val_loss += loss.item()
            if val_idx > 40:
                break
        assert val_idx != -1
        dist.barrier()
        val_res = val_metric.get_metric()
        if len(val_dl) == 0:
            try:
                if gpc.get_global_rank() == 0:
                    logger.info(f"Validation dataset:{val_name} is empty")
            except KeyError:
                pass

        if enable and len(val_dl) != 0:
            val_loss = val_loss / (val_idx + 1 + 1e-6)
            infos = {
                f"val/{val_name}_loss": val_loss,
                f"val/{val_name}_acc": val_res["acc"],
                f"val/{val_name}_plex": val_res["perplexity"],
                # f"val/{val_name}_loss": val_loss,
            }
            val_metric = {
                "step": step_count,
                "val_loss": val_loss,
                "val_acc": val_res["acc"],
                "val_perplexity": val_res["perplexity"],
            }
            for key, value in infos.items():
                writer.add_scalar(key, value, step_count)
            infos["step"] = step_count
            logger.info(
                msg=(f"Validation on {val_name}:" + ",".join([f"{key}={value}" for key, value in infos.items()])),
                extra=val_metric,
            )

    engine.train()
    torch.cuda.empty_cache()
    dist.barrier()

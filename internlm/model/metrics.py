from typing import List

import torch
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
from torch_scatter import scatter

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.parallel import is_no_pp_or_last_stage


class AccPerplex:
    """
    AccPerplex module for calculating model's accuracy and perplexity metrics.

    Args:
        device: The GPU device.
        tp_pg: The tensor parallel process group.
        dp_pg: The data parallel process group.
        tokenizer: For calculating BPB.
        dataset_types (List[str]): Various data types that will be used in the current training process,
            such as ['en', 'cn', 'code']. The order of the List should be consistent with the type_id specified
            in the dataset. Changed parameters need to be used in conjunction with set_current_type_ids().
    """

    def __init__(self, device, tp_pg, dp_pg, tokenizer=None, dataset_types: List[str] = None):
        self.device = device
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_log_probs = torch.Tensor([0]).to(device=device)
        self.tp_pg = tp_pg
        self.dp_pg = dp_pg
        self.tp_local_rank = torch.distributed.get_rank(self.tp_pg)
        self.tokenizer = tokenizer
        self.total_bytes = torch.Tensor([0]).to(device=device).view(1)
        self.batch_shift = 0
        self.type_ids = None
        if dataset_types is not None:
            self.dataset_types = dataset_types
            self.total_type_count = len(dataset_types)
            self.ds_right = torch.zeros(self.total_type_count, dtype=torch.long, device=device)
            self.ds_tokens = torch.zeros(self.total_type_count, dtype=torch.long, device=device)

        self.loss_with_type_id = LossWithTypeId(device, dp_pg, dataset_types)

    def set_current_type_ids(self, type_ids: torch.Tensor):
        self.batch_shift = 0
        self.type_ids = type_ids.cuda()

    def __call__(self, logits, labels):
        return self.update(logits, labels, type_ids=self.type_ids)

    def update(self, logits, labels, type_ids=None):
        if gpc.config.model.use_flash_attn:
            micro_bsz = labels.size(0)
        else:
            micro_bsz = 1
        if type_ids is not None:
            type_ids = type_ids[self.batch_shift * micro_bsz : (self.batch_shift + 1) * micro_bsz].view(-1)
            self.batch_shift += 1
        self.loss_with_type_id.update(logits, labels, type_ids)

        with torch.no_grad():
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            logits = logits.detach().clone()
            labels = labels.detach().clone()

            if self.tokenizer:  # need to calculate bits per bytes
                sequences = self.tokenizer.decode_ids(labels.tolist())
                self.total_bytes += sum(map(lambda x: len(x.encode("utf-8")), sequences))

            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
            # There is a shift according to the current rank, because the logits are split
            pred_shift = self.tp_local_rank * logits.shape[-1]

            logits_max = torch.max(shift_logits, dim=-1)[0]
            torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=self.tp_pg)
            # Determine whether the maximum value of the current local tensor is the global maximum value
            logits_global = logits_max == torch.max(shift_logits, dim=-1)[0]

            corrects = torch.logical_and(
                (shift_labels == (shift_logits.argmax(dim=-1) + pred_shift)), logits_global
            ).long()
            mask = shift_labels.ne(-100).long()
            if hasattr(self, "total_type_count"):
                ds_acc = scatter(corrects, type_ids, dim=0, reduce="sum")
                token_num_type = scatter(mask, type_ids, dim=0, reduce="sum")
                if len(ds_acc) < self.total_type_count:
                    ds_acc = torch.cat([ds_acc, ds_acc.new_zeros(self.total_type_count - len(ds_acc))])
                    token_num_type = torch.cat(
                        [token_num_type, token_num_type.new_zeros(self.total_type_count - len(token_num_type))]
                    )
                self.ds_tokens += token_num_type
                sync_tensor = ds_acc
                torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)
                self.ds_right += sync_tensor.view(-1)

            acc = corrects.sum()
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)
            self.right += acc  # Masked_fill is not needed here because -100 is not available anyway
            self.total += mask.sum()

            # Subtract the maximum value.
            shift_logits = shift_logits.sub(logits_max.unsqueeze(dim=-1))

            # Get the partition's vocab indecies
            partition_vocab_size = shift_logits.size()[-1]
            vocab_start_index = partition_vocab_size * self.tp_local_rank
            vocab_end_index = vocab_start_index + partition_vocab_size

            # Create a mask of valid vocab ids (1 means it needs to be masked).
            target_mask = (shift_labels < vocab_start_index) | (shift_labels >= vocab_end_index)
            masked_target = shift_labels - vocab_start_index
            masked_target[target_mask] = 0

            # Get predicted-logits = logits[target].
            # For Simplicity, we convert logits to a 2-D tensor with size
            # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
            logits_2d = shift_logits.view(-1, partition_vocab_size)
            masked_target_1d = masked_target.view(-1)
            arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
            predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
            predicted_logits_1d = predicted_logits_1d.clone().contiguous()
            predicted_logits = predicted_logits_1d.view_as(shift_labels)  # bsz x max_len
            predicted_logits[target_mask] = 0.0
            # All reduce is needed to get the chunks from other GPUs.
            torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)

            pred_exp_logits = torch.exp(predicted_logits)
            # Sum of exponential of logits along vocab dimension across all GPUs.
            sum_exp_logits = torch.exp(shift_logits).sum(dim=-1)
            torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)

            total_log_probs = -(pred_exp_logits / sum_exp_logits).log().masked_fill(shift_labels.eq(-100), 0).sum()
            self.total_log_probs += total_log_probs

    def get_metric(self, reset=True):
        if is_no_pp_or_last_stage() and self.dp_pg is not None:
            torch.distributed.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            torch.distributed.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            torch.distributed.all_reduce(self.total_log_probs, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            if hasattr(self, "total_type_count"):
                torch.distributed.all_reduce(self.ds_right, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
                torch.distributed.all_reduce(self.ds_tokens, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            if self.tokenizer:
                torch.distributed.all_reduce(self.total_bytes, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)

        acc = round((self.right / self.total).item(), 4)
        perplexity = round(torch.exp(self.total_log_probs / self.total).item(), 4)
        bits_per_bytes = round((self.total_log_probs / self.total_bytes).item(), 4) if self.tokenizer else 0

        if hasattr(self, "total_type_count"):
            ds_acc = {}
            ds_tokens = {}
            for i in range(self.total_type_count):
                ds_acc[f"acc/{self.dataset_types[i]}"] = round(
                    (self.ds_right[i].float() / (self.ds_tokens[i].float() + 1e-5)).item(), 4
                )
                ds_tokens[f"tokens/{self.dataset_types[i]}"] = self.ds_tokens[i].item()
        if reset:
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_log_probs.fill_(0)
            self.total_bytes.fill_(0)
            if hasattr(self, "total_type_count"):
                self.ds_right.fill_(0)
                self.ds_tokens.fill_(0)
        if self.tokenizer is not None:
            res = {"acc": acc, "perplexity": perplexity, "BPB": bits_per_bytes}
        else:
            res = {"acc": acc, "perplexity": perplexity}
        if hasattr(self, "total_type_count"):
            res.update(ds_acc)
            res.update(ds_tokens)

        loss_res = self.loss_with_type_id.get_metric(reset)
        res.update(loss_res)

        return res


class LossWithTypeId:
    """
    Notice the loss value computed here may be not the same with the main info loss,
    cause loss here is the reduced result of the data parallel.
    """

    def __init__(self, device, dp_pg, dataset_types: List[str] = None) -> None:
        self.device = device
        self.dp_pg = dp_pg

        self.loss = torch.Tensor([0.0]).to(device=device)
        self.token_num = torch.Tensor([0.0]).to(device=device)

        if dataset_types is not None:
            self.dataset_types = dataset_types
            self.total_type_count = len(dataset_types)
            self.ds_loss = torch.zeros(self.total_type_count, dtype=torch.float, device=device)
            self.ds_token_num = torch.zeros(self.total_type_count, dtype=torch.float, device=device)

        self.loss_fn = FlashCrossEntropyLoss(
            reduction="none", inplace_backward=True, process_group=gpc.get_group(ParallelMode.TENSOR)
        )

    def update(self, logits, labels, type_ids=None):
        with torch.no_grad():
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = logits.contiguous().view(-1, logits.size(-1))
            labels = labels.contiguous().view(-1)
            loss_list = self.loss_fn(logits, labels)

            cond = labels != -100
            real_loss_list = loss_list[cond]
            self.loss += real_loss_list.sum()
            self.token_num += real_loss_list.numel()

            if hasattr(self, "total_type_count"):
                type_ids = type_ids.contiguous().view(-1).to(self.device)
                real_type_ids = type_ids[cond]

                loss_list_type = scatter(real_loss_list, real_type_ids, dim=0, reduce="sum")
                token_num_type = scatter(torch.ones_like(real_loss_list), real_type_ids, dim=0, reduce="sum")

                if len(loss_list_type) < self.total_type_count:
                    loss_list_type = torch.cat(
                        [loss_list_type, loss_list_type.new_zeros(self.total_type_count - len(loss_list_type))]
                    )
                    token_num_type = torch.cat(
                        [token_num_type, token_num_type.new_zeros(self.total_type_count - len(token_num_type))]
                    )
                self.ds_loss += loss_list_type
                self.ds_token_num += token_num_type

    def get_metric(self, reset=True):
        if is_no_pp_or_last_stage() and self.dp_pg is not None:
            torch.distributed.all_reduce(self.loss, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            torch.distributed.all_reduce(self.token_num, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            if hasattr(self, "total_type_count"):
                torch.distributed.all_reduce(self.ds_loss, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
                torch.distributed.all_reduce(self.ds_token_num, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)

        loss = round((self.loss / self.token_num).item(), 4)
        res = {
            "loss_from_metric": loss,
        }
        if hasattr(self, "total_type_count"):
            ds_loss = {}
            for i in range(self.total_type_count):
                ds_loss[f"loss/{self.dataset_types[i]}"] = round((self.ds_loss[i] / self.ds_token_num[i]).item(), 4)
            res.update(ds_loss)

        if reset:
            self.loss.fill_(0.0)
            self.token_num.fill_(0.0)
            if hasattr(self, "total_type_count"):
                self.ds_loss.fill_(0.0)
                self.ds_token_num.fill_(0.0)

        return res

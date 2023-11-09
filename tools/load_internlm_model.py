import inspect
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Union

import torch

from internlm.apis.inference import SequenceGenerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.initialize.launch import launch_from_torch
from internlm.train import initialize_model
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.utils.storage_manager import get_fns, init_storage_manager, llm_load
from tools.transformers.interface import GenerationConfig

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def merge_pp_within_tp(folder, del_model_prefix: bool = False) -> Dict[str, torch.Tensor]:
    """Merge checkpoints from different pipelines of the model.

    Args:
        folder (str): checkpoint directory.
        del_model_prefix (bool, optional): Whether to remove the "model." string in the key in state_dict. Defaults
            to False.

    Returns:
        Dict[str, torch.Tensor]: A state_dict that belongs to the model of the current tensor parallel rank.
    """
    assert folder is not None, "Please specify the folder of the pretrained model"
    fns = get_fns(folder)

    model_fns = []
    for fn in fns:
        # filter with `_t` is for avoiding conflict with model_config.py
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        max_pp = max(max_pp, int(pp[2:]) + 1)
        max_tp = max(max_tp, int(tp[2:]) + 1)

    if max_tp < 0 or max_pp < 0:
        raise RuntimeError(f"Could not find any model checkpoints starting with 'model_t' in the folder: {folder}")

    assert max_tp == gpc.get_world_size(ParallelMode.TENSOR), (
        f"Expected TP size in loaded checkpoint to be equal to TP size in current config, but got {max_tp} in "
        f"checkpoint and {gpc.get_world_size(ParallelMode.TENSOR)} in current config"
    )
    tp = gpc.get_local_rank(ParallelMode.TENSOR)

    layer_shift = 0

    tp_states = {}
    for pp in range(max_pp):
        _layer_shift = 0
        model_name = f"model_tp{tp}_pp{pp}.pt"
        states = llm_load(os.path.join(folder, model_name), map_location="cpu")

        keys = list(states.keys())
        for key in keys:
            match = re.search(r"\.\d+\.", key)
            if match is not None:  # Indicate layer related. shift is required.
                s, e = match.span()
                layer_idx = int(key[s + 1 : e - 1]) + layer_shift
                _layer_shift = max(_layer_shift, int(key[s + 1 : e - 1]))
                name = key[:s] + f".{layer_idx}." + key[e:]
                if del_model_prefix:
                    name = name[6:] if name.startswith("model.") else name
                tp_states[name] = states[key]
            else:
                name = key
                if del_model_prefix:
                    name = name[6:] if name.startswith("model.") else name
                tp_states[name] = states[key]
        layer_shift += _layer_shift + 1

    return tp_states


def match_fn_signature(func: Callable, args_dict: Dict) -> None:
    """Matches the parameters of a function.

    Given a function and a parameter dictionary, remove key-value pairs in the dictionary
     that do not match the parameters of the function.

    Args:
        func (Callable): specific function.
        args_dict (Dict): parameter dictionary.
    """
    params = inspect.signature(func).parameters
    params = set(params)
    args_set = set(args_dict).difference(params)
    for _name in args_set:
        args_dict.pop(_name)
    if len(args_set) and gpc.is_rank_for_log():
        logger.warning(f"These args:{args_set} are popped for func:{func.__name__}.")


def get_tp_rank() -> int:
    """Get the tensor parallel rank.
    This script uses torchrun to initialize the environment, so RANK in the environment variable is the tensor
     parallel rank.

    Returns:
        int: The tensor parallel rank to which the current process belongs.
    """
    return int(os.environ.get("RANK", 0))


def get_tp_world_size() -> int:
    """Get the tensor parallel world size.

    Returns:
        int: The tensor parallel world size to which the current process belongs.
    """
    return int(os.environ.get("WORLD_SIZE", 0))


def initialize_internlm_model(
    model_type: str,
    ckpt_dir: str,
    model_config: Dict,
    del_model_prefix: bool = False,
    param_dtype: torch.dtype = torch.bfloat16,
    training: bool = False,
    seed: int = 1024,
) -> torch.nn.Module:
    """Initialize internlm model.

    Args:
        model_type (str): The types of models supported by train_internlm, such as "INTERNLM".
        ckpt_dir (str): Directory where model checkpoints are stored. Its format needs to be like this:
            (a) local path, such as: "local:{your local path}";
            (b) boto3 path, such as: "boto3:s3://{bucket name}.{ip}/{your ceph path}".
        model_config (Optional[Union[Dict, str]], optional): Configuration of models. Defaults to None.
        del_model_prefix (bool, optional):  Whether to remove the "model." string in the key in state_dict.
            Defaults to False.
        param_dtype (torch.dtype, optional): The dtype of the model at inference time. This value can be a string.
            Use "torch.tf32" when you want to use tf32 to do the inference. Defaults to torch.bfloat16.
        training (bool, optional): model.train() or model.eval(). Defaults to False.
        seed (int, optional): Defaults to 1024.
    """

    if gpc.is_rank_for_log():
        logger.info(f"tp world size: {get_tp_world_size()}.")

    if isinstance(param_dtype, str):
        try:
            param_dtype = eval(param_dtype)  # pylint: disable=W0123
        finally:
            pass
    if param_dtype == "torch.tf32":
        param_dtype = torch.float32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if not isinstance(param_dtype, torch.dtype):
        raise ValueError("Parameter ``param_dtype`` is not right.")

    try:
        init_storage_manager(False, None, None)
    except AssertionError:
        pass
    except Exception as e:
        raise e

    model_config["dtype"] = param_dtype
    model_config["parallel_output"] = False
    match_fn_signature(MODEL_INITIALIZER.get_module(model_type), model_config)
    if gpc.is_rank_for_log():
        logger.info(f"model_config: {model_config}.")
    launch_from_torch(
        config=dict(
            model_type=model_type,
            model=model_config,
            parallel=dict(
                zero1=dict(size=1, fsdp=False),
                pipeline=dict(size=1, interleaved_overlap=True),
                tensor=get_tp_world_size(),
                sequence_parallel=0,
            ),
        ),
        seed=seed,
    )
    model = initialize_model()
    # Directly get the origin model without NativeAMP wrapper.
    model = model.model

    state_dict = merge_pp_within_tp(ckpt_dir, del_model_prefix=del_model_prefix)
    load_info = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Rank:{gpc.get_local_rank(ParallelMode.TENSOR)}. Load info: {load_info}.")

    model.to(param_dtype)
    if training:
        model.train()
    else:
        model.eval()
    model.cuda()
    torch.distributed.barrier()
    return model


def get_model_device(model):
    for param in model.parameters():
        device = param.device
        break
    return device


def internlm_interactive_generation(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    additional_eos_token_list: Optional[Union[int, List[int]]] = None,
):
    sequenece_generator = SequenceGenerator(
        decoder=model,
        eos_token_id=tokenizer.eos_id(),
        pad_token_id=tokenizer.eos_id(),
        bos_token_id=tokenizer.bos_id(),
        additional_eos_token_list=additional_eos_token_list,
    )
    additional_eos_token_list = torch.LongTensor(additional_eos_token_list)
    input_ids = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    input_ids = torch.LongTensor([input_ids]).to(get_model_device(model))
    output_generator = sequenece_generator.streaming_generate(
        tokens=input_ids,
        max_length=generation_config.max_length,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature,
        top_k=40,
        top_p=generation_config.top_p,
        repetition_penalty=generation_config.repetition_penalty,
        length_penalty=1.0,
    )
    for token_ids in output_generator:
        token_ids = token_ids.cpu().tolist()[0]
        if torch.any(additional_eos_token_list == token_ids[-1]):
            token_ids.pop()
        cur_output = tokenizer.decode(token_ids)
        cur_output = cur_output[len(prompt) :]
        yield cur_output


if __name__ == "__main__":
    """
    Here is a simple example to generate with origin internlm model architecture.
    Use the following command to run:
    >>> torchrun --master_port 12331 --nnodes=1 --node_rank=0 --nproc_per_node=1 tools/load_internlm_model.py
    """
    model = initialize_internlm_model(
        model_type="INTERNLM",
        ckpt_dir="[Please replace this with the directory where the internlm model weights are stored]",
        model_config=dict(
            checkpoint=False,
            num_attention_heads=32,
            embed_split_hidden=True,
            vocab_size=103168,
            embed_grad_scale=1,
            parallel_output=False,
            hidden_size=4096,
            num_layers=32,
            mlp_ratio=8 / 3,
            apply_post_layer_norm=False,
            dtype="torch.bfloat16",
            norm_type="rmsnorm",
            layer_norm_epsilon=1e-5,
            use_flash_attn=True,
            num_chunks=1,
            use_dynamic_ntk_rope=True,
        ),
        del_model_prefix=True,
    )

    from sentencepiece import SentencePieceProcessor

    prompt = """<|User|>:{query}<eoh>\n<|Bot|>:"""
    prompt = prompt.replace("{query}", "hello")
    tokenizer = SentencePieceProcessor("tools/V7_sft.model")  # pylint: disable=E1121

    generation_config = GenerationConfig()
    output_generator = internlm_interactive_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        generation_config=generation_config,
        additional_eos_token_list=[103028],
    )

    for text in output_generator:
        print(text)

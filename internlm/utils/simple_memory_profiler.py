import os
import time
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Tuple

import pyecharts
import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.solver.pipeline_utils import partition_uniform

mb = 1024 * 1024


class SimpleMemState:
    """
    A class to represent the memory state of a model layer.

    Args:
        layer_name (str): The name of the layer.
        layer_mem (int): The memory usage of the layer in bytes.
    """

    def __init__(self, layer_name: str, layer_mem: int = 0) -> None:
        self.layer_name = layer_name

        # Memory status of the current model layer.
        self._layer_mem: int = layer_mem
        # Total memory status of the model and sub-models, initialized with layer memory.
        self._total_mem: int = self._layer_mem
        # SimpleMemState of sub-models.
        self.sub_model_stats = OrderedDict()

    @property
    def layer_mem(self) -> int:
        """
        Get the memory usage of the layer.

        Returns:
            int: The memory usage of the layer in bytes.
        """
        return self._layer_mem

    @layer_mem.setter
    def layer_mem(self, new_layer_mem: int) -> None:
        """
        Set the memory usage of the layer.

        Args:
            new_layer_mem (int): The new memory usage of the layer in bytes.
        """
        diff = new_layer_mem - self._layer_mem
        self._layer_mem = new_layer_mem
        self._total_mem += diff

    @property
    def total_mem(self) -> int:
        """
        Get the total memory usage of the model and sub-models.

        Returns:
            int: The total memory usage in bytes.
        """
        return self._total_mem

    def add(self, layer_name: str, layer_mem: int = 0, flush: bool = True) -> None:
        """
        Add a layer to the memory state.

        Args:
            layer_name (str): The name of the layer.
            layer_mem (int, optional): The memory usage of the layer in bytes. Defaults to 0.
            flush (bool, optional): Whether to update the total memory usage. Defaults to True.
        """
        path = layer_name.split(".")

        target = self.find_layer_state(path, create=True)
        target.layer_mem = layer_mem

        if flush:
            self.update_total_memory()

    def delete(self, layer_name: str, flush: bool = True) -> None:
        """
        Delete a layer from the memory state.

        Args:
            layer_name (str): The name of the layer.
            flush (bool, optional): Whether to update the total memory usage. Defaults to True.
        """
        path = layer_name.split(".")
        assert len(path) >= 2, f"Only support deleting non-root layers, layer_name: {layer_name}"

        parent_path = path[0:-1]
        layer = path[-1]
        parent = self.find_layer_state(parent_path)

        if parent is not None and layer in parent.sub_model_stats:
            del parent.sub_model_stats[layer]

        if flush:
            self.update_total_memory()

    def update_total_memory(self) -> None:
        """
        Update the total memory usage of the model and sub-models.
        """
        for stat in self.sub_model_stats.values():
            # Update sub-model status first.
            stat.update_total_memory()
            # Add sub-model total_mem to model total_mem.
            self._total_mem += stat._total_mem

    def find_layer_state(self, path: Tuple[str], create: bool = False) -> "SimpleMemState":
        """
        Find the memory state of a layer.

        Args:
            path (Tuple[str]): The path to the layer.
            create (bool, optional): Whether to create the layer if it doesn't exist. Defaults to False.

        Returns:
            SimpleMemState: The memory state of the layer.
        """
        current_node = self

        for _node in path:
            if _node not in current_node.sub_model_stats:
                if not create:
                    return None
                # Create a layer node.
                current_node.sub_model_stats[_node] = SimpleMemState(_node)

            current_node = current_node.sub_model_stats[_node]

        return current_node

    def dump(self, prefix: str = "") -> str:
        """
        Dump the memory state of the model and sub-models.

        Args:
            prefix (str, optional): The prefix to add to the layer names. Defaults to "".

        Returns:
            str: The memory state information.
        """
        cur_prefix = prefix + "." + self.layer_name if prefix != "" else self.layer_name
        res = f"layer: {cur_prefix}, layer_mem: {self.layer_mem / mb:.2f} MB, total_mem: {self.total_mem / mb:.2f} MB\n"

        for sub_layer in self.sub_model_stats.values():
            res += sub_layer.dump(cur_prefix)

        return res

    def to_json(self, base: int = 1024 * 1024) -> dict:
        """
        Convert the memory state to a JSON structure.

        Returns:
            dict: The JSON structure of the memory state.
        """
        children = [child.to_json() for child in self.sub_model_stats.values()]
        if len(children) == 0:
            return {"name": self.layer_name, "value": self.layer_mem // base}
        else:
            return {"name": self.layer_name, "children": children}


class SimpleMemoryProfiler:
    """
    A memory profiler for a llm model.

    Args:
        model (torch.nn.Module): The model to profile.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        log_file (str): The file to write the memory state information to.
        activation_config (List[str], optional): The list of activation layers to track. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        log_folder: str,
        total_steps: int = 5,
        activation_config: List[str] = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._log_folder = log_folder
        self._remaining_steps = total_steps

        self._stoped = False
        self._record_start_time = time.time()

        # For activation memory state.
        self._activation_config = activation_config
        self._activation_mem_inited: bool = False
        self._activation_mem: int = 0
        self._activation_max_count = 0
        self._activation_base_mem: SimpleMemState = SimpleMemState("activations")

        # Check or create log folder
        os.makedirs(self._log_folder, exist_ok=True)

        # Register activation memory tracking hooks
        self._register_activation_trace_hooks()

        # Calculate static parameter cuda memory
        self._param_mem_state = SimpleMemState("param_mem")
        self._calc_tensor_memory(self._param_mem_state, self._model.named_parameters())
        # Calculate static grad cuda memory
        self._grad_mem_state = SimpleMemState("grad_mem")
        self._calc_tensor_memory(self._grad_mem_state, self._model.named_parameters(), True)
        # Calculate static optimizer state cuda memory
        self._os_params_mem_state = SimpleMemState("os_params_mem")
        self._os_state_mem_state = SimpleMemState("os_state_mem")
        self._calc_tensor_group_memory(
            self._os_params_mem_state, [(k, v) for k, v in enumerate(self._optimizer.param_groups)]
        )

        # Generate the first memory record
        self.point(create=True)

    def point(self, with_options: str = "", create: bool = False) -> None:
        """
        Record the memory state.

        Args:
            with_options (str, optional): The options to include in the memory state. Defaults to "".
            create (bool, optional): Whether to create a new memory record file. Defaults to False.

        Returns:
            None
        """
        now = time.time()
        file = f"{self._log_folder}/memory.log"

        if with_options == "all":
            options = ["params", "grads", "os_params", "os_state", "activation_base"]
        else:
            options = with_options.split(",")

        total_mem = (
            self._param_mem_state.total_mem
            + self._grad_mem_state.total_mem
            + self._os_params_mem_state.total_mem
            + self._os_state_mem_state.total_mem
            + self._activation_mem
        ) / mb

        # Generate summary information for memory state
        summary_info = (
            f"total_memory: {total_mem:.2f} MB"
            + "\n"
            + f"params_memory: {self._param_mem_state.total_mem / mb:.2f} MB, "
            + f"grads_memory: {self._grad_mem_state.total_mem / mb:.2f} MB, "
            + f"os_params_memory: {self._os_params_mem_state.total_mem / mb:.2f} MB, "
            + f"os_state_memory: {self._os_state_mem_state.total_mem / mb:.2f} MB, "
            + f"activation_memory: {self._activation_mem / mb:.2f} MB"
        )

        # Generate layout information based on selected options
        layout_info = ""
        if "params" in options:
            layout_info += "params_layout:\n" + self._param_mem_state.dump()
        if "grads" in options:
            layout_info += "grads_layout:\n" + self._grad_mem_state.dump()
        if "os_params" in options:
            layout_info += "os_params_layout:\n" + self._os_params_mem_state.dump()
        if "os_state" in options:
            layout_info += "os_state_layout:\n" + self._os_state_mem_state.dump()
        if "activation_base" in options:
            layout_info += "activation_base_layout:\n" + self._activation_base_mem.dump()

        # Write memory state information to log file
        file_mode = "w" if create else "a"
        with open(file, file_mode, encoding="utf-8") as writer:
            writer.write(
                "Memory State:\n" + f"time: {now - self._record_start_time}\n" + "---summary---\n" + summary_info + "\n"
            )
            if layout_info != "":
                writer.write("---Layout---\n" + layout_info)
            writer.write("\n")

    def step(self) -> None:
        """
        Update the memory state of the optimizer state.

        Returns:
            None
        """
        if self._stoped:
            return

        self._remaining_steps -= 1
        if self._remaining_steps == 0:
            self._stoped = True

        # Update os state memory usage
        self._os_state_mem_state = SimpleMemState("os_state_mem")
        self._calc_tensor_group_memory(
            self._os_state_mem_state, [(k, v) for k, v in self._optimizer.state_dict()["state"].items()]
        )

        if not self._stoped:
            # Do we need to print os_state_layout every time? Is it always constant?
            self.point(with_options="os_state")
        else:
            # Dump memory layout
            self.point(with_options="all")
            # Generate sunburst charts
            self._render_sunburst_chart(self._param_mem_state.to_json()["children"], "params_memory_sunburst")
            self._render_sunburst_chart(self._grad_mem_state.to_json()["children"], "grads_memory_sunburst")
            self._render_sunburst_chart(
                [self._os_params_mem_state.to_json(), self._os_state_mem_state.to_json()],
                "os_memory_sunburst",
            )
            self._render_sunburst_chart(self._activation_base_mem.to_json()["children"], "activation_memory_sunburst")
            # Generate summary sunburst chart
            summary_sunburst_data = [
                {"name": "params", "value": self._param_mem_state.total_mem // mb},
                {"name": "grads", "value": self._grad_mem_state.total_mem // mb},
                {"name": "os_params", "value": self._os_params_mem_state.total_mem // mb},
                {"name": "os_state", "value": self._os_state_mem_state.total_mem // mb},
                {"name": "activation", "value": self._activation_base_mem.total_mem // mb},
            ]

            self._render_sunburst_chart(summary_sunburst_data, "summary_sunburst")

    def _render_sunburst_chart(self, data: Any, name: str) -> None:
        pyecharts.charts.Sunburst(init_opts=pyecharts.options.InitOpts(width="1000px", height="1000px")).add(
            name,
            data_pair=data,
            highlight_policy="ancestor",
            radius=[0, "95%"],
            levels=[
                {},
                {
                    "r0": "10%",
                    "r": "40%",
                    "itemStyle": {"borderWidth": 3},
                    "label": {"align": "left"},
                },
                {"r0": "40%", "r": "65%", "label": {"align": "left"}},
                {"r0": "65%", "r": "80%", "label": {"align": "left"}},
                {"r0": "80%", "r": "90%", "label": {"align": "left"}},
                {
                    "r0": "90%",
                    "r": "92%",
                    "label": {"position": "outside", "padding": 3, "silent": False},
                    "itemStyle": {"borderWidth": 3},
                },
            ],
        ).set_global_opts(title_opts=pyecharts.options.TitleOpts(title="CUDA Memory")).set_series_opts(
            label_opts=pyecharts.options.LabelOpts(formatter="{b}")
        ).render(
            f"{self._log_folder}/{name}.html"
        )

    def _inner_activation_trace_hook(self, layer_name: str, model: Any, inputs: Any, output: torch.Tensor) -> None:
        """
        Hook function to trace the activation memory usage for a inner layer.

        Args:
            layer_name (str): The name of the layer.
            model (Any): The model.
            inputs (Any): The inputs to the layer.
            output (torch.Tensor): The output tensor.

        Returns:
            None
        """
        del model, inputs
        assert isinstance(output, torch.Tensor), f"Invalid output type: {type(output)}"

        if self._stoped or self._activation_mem_inited:
            return

        # Delay updating the total_mem of activation_base_mem here, it will be handled in the forward ending hook.
        self._activation_base_mem.add(layer_name, output.element_size() * output.nelement(), flush=False)

    def _activation_trace_hook_forward(self, model: Any, inputs: Any, output: torch.Tensor) -> None:
        """
        Hook function to trace the activation memory usage for a forward pass.

        Args:
            model (Any): The model.
            inputs (Any): The inputs to the model.
            output (torch.Tensor): The output tensor.

        Returns:
            None
        """
        del model, inputs
        assert isinstance(output, torch.Tensor), f"invalid output type: {type(output)}"

        if self._stoped:
            return

        # Check if the activation memory has been initialized
        if self._activation_mem_inited is False:
            # Update the total memory of the activation base memory state
            self._activation_base_mem.update_total_memory()
            # Set with_options to "activation_base" to include activation_base_layout in the memory dump
            self._activation_mem_inited = True

        # Accumulate activation memory usage for each forward pass
        self._activation_mem += self._activation_base_mem.total_mem

        # Update activation max count
        if self._activation_mem // self._activation_base_mem.total_mem > self._activation_max_count:
            self._activation_max_count = self._activation_mem // self._activation_base_mem.total_mem

        # Trigger a memory record
        self.point()

    def _activation_tarce_hook_backward(self, model: Any, inputs: Any, grad_outputs: Any) -> None:
        """
        Hook function to trace the activation memory usage for a backward pass.

        Args:
            model (Any): The model.
            inputs (Any): The inputs to the model.
            grad_outputs (Any): The gradients of the outputs.

        Returns:
            None
        """
        del model, inputs, grad_outputs

        if self._stoped:
            return

        # Release activation memory usage for each backward pass
        self._activation_mem -= self._activation_base_mem.total_mem

        # Trigger a memory record
        self.point()

    def _register_activation_trace_hooks(self) -> None:
        """
        Register activation trace hooks for the model and each submodule in the model.
        """

        # Register inner activation trace hooks for each submodule in the model
        for layer_name in self._activation_config:
            # Register a hook for every activation
            model = self._model
            sub_models = layer_name.split(".")
            # Get the target sub-model
            for sub_model_name in sub_models:
                try:
                    model = model.get_submodule(sub_model_name)
                except AttributeError:
                    model = None
                    break

            # Register the hook
            if model is not None:
                model.register_forward_hook(partial(self._inner_activation_trace_hook, layer_name))

        # Register a forward hook for the main model to track activation memory usage
        self._model.register_forward_hook(self._activation_trace_hook_forward)
        # Register a backward hook for the main model to release activation memory usage
        self._model.register_full_backward_hook(self._activation_tarce_hook_backward)

    def _calc_tensor_memory(
        self, root_stat: SimpleMemState, named_tensors: Dict[str, torch.Tensor], require_grad: bool = False
    ) -> None:
        """
        Calculate the memory usage of tensors and update the memory state.

        Args:
            root_stat (SimpleMemState): The root memory state.
            named_tensors (Dict[str, torch.Tensor]): A dictionary containing the named tensors.
            require_grad (bool, optional): Whether to consider tensors with gradients. Defaults to False.

        Returns:
            None
        """
        for name, tensor in named_tensors:
            if require_grad and not tensor.requires_grad:
                continue

            layer_splits = name.split(sep=".")
            layer_stat = root_stat.find_layer_state(layer_splits, create=True)
            layer_stat.layer_mem = tensor.element_size() * tensor.nelement()

        root_stat.update_total_memory()

    def _calc_tensor_group_memory(self, root_stat: SimpleMemState, tensor_groups: List[Tuple[int, torch.Tensor]]):
        """
        Calculate the memory usage of a group of tensors.

        Args:
            root_stat (SimpleMemState): The root memory state.
            tensor_groups (List[Tuple[int, torch.Tensor]]): A list of tuples containing the tensor groups.

        Returns:
            None
        """

        def _normalize_helper(named_tensors: Dict[str, Any]) -> List[Tuple[str, Any]]:
            """
            Normalize the named tensors.

            Args:
                named_tensors (Dict[str, Any]): The named tensors to normalize.

            Returns:
                List[Tuple[str, Any]]: The normalized named tensors.
            """
            res = {}

            for name, tensors in named_tensors.items():
                if isinstance(tensors, torch.Tensor):
                    res[name] = tensors
                elif isinstance(tensors, (list, tuple)):
                    for index, tensor in enumerate(tensors):
                        res[f"{name}.{index}"] = tensor
                elif isinstance(tensors, dict):
                    for subname, tensor in tensors.items():
                        res[f"{name}.{subname}"] = tensor
                else:
                    raise TypeError(f"unsupported normalize value type: {type(tensors)}")

            return list(res.items())

        def _value_check(tensor_or_tensors):
            """
            Check if the input is a tensor or a collection of tensors.

            Args:
                tensor_or_tensors (Any): The input to check.

            Returns:
                bool: True if the input is a tensor or a collection of tensors, False otherwise.
            """
            if torch.is_tensor(tensor_or_tensors):
                return True
            elif isinstance(tensor_or_tensors, (list, tuple)) and all(torch.is_tensor(x) for x in tensor_or_tensors):
                return True
            elif isinstance(tensor_or_tensors, dict) and all(torch.is_tensor(x) for x in tensor_or_tensors.values()):
                return True
            else:
                return False

        # Calculate the memory usage of a group of tensors.
        for idx, tensors in tensor_groups:
            # Normalize the named tensors
            named_tensors = {f"{idx}.{k}": v for k, v in tensors.items() if _value_check(v)}
            named_tensors = _normalize_helper(named_tensors)
            # Calculate the memory usage of the tensors and update the memory state
            self._calc_tensor_memory(root_stat, named_tensors)


def build_activation_config(num_layers: int, num_chunks: int = 1) -> List[str]:
    # TODO: support interleaved pipeline scheduling.
    assert num_chunks == 1, "Only support num_chunks == 1"

    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    start, end = parts[0]
    num_blocks = end - start

    block_conf_tmpl = [
        "mixer.rotary_emb",
        "mixer.Wqkv",
        "mixer.inner_attn",
        "mixer.inner_cross_attn",
        "mixer.out_proj",
        # "dropout1", # skip when dropout_selective_checkpoint is True
        # "dropout2", # skip when dropout_selective_checkpoint is True
        "norm1",
        "norm2",
        "mlp.w1",
        "mlp.w2",
        "mlp.w3",
    ]

    block_conf = []
    for block_id in range(num_blocks):
        block_conf += [f"blocks.{block_id}.{layer}" for layer in block_conf_tmpl]

    # We don't need to care about whether the embedding, norm, and head layers exist in the model after partitioning.
    # If they don't exist, they will be automatically ignored when registering activation trace hooks.
    activation_conf = ["embedding", "norm", "head"] + block_conf

    return activation_conf


if __name__ == "__main__":

    class SimpleModel(torch.nn.Module):
        """
        A simple model with three linear layers.

        Args:
            skip_layer2 (bool, optional): Whether to skip layer2. Defaults to False.
        """

        def __init__(self, skip_layer2: bool = False):
            super().__init__()
            self.layer1 = torch.nn.Linear(5120, 5120, True)
            self.layer3 = torch.nn.Linear(5120, 5120, False)

            if skip_layer2:
                self.layer2 = None
            else:
                self.layer2 = SimpleModel(skip_layer2=True)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the model.

            Args:
                inputs (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
            """
            output1 = self.layer1(inputs)
            if self.layer2 is not None:
                output2 = self.layer2(output1)
            else:
                output2 = output1
            output = self.layer3(output2)

            return output

    # init model and optimizer
    _model: torch.nn.Module = SimpleModel()
    _optimizer = torch.optim.Adam(_model.parameters())

    # create activation config for simple model layer by layer.
    activation_configs = [
        # model level 0
        "layer1",
        "layer2",
        "layer3",
        # model level 1
        "layer2.layer1",
        "layer2.layer3",
    ]

    _model.modules()

    # init profiler
    profiler = SimpleMemoryProfiler(_model, _optimizer, "./test_simple_memory_profiler.log", activation_configs)

    _optimizer.zero_grad()

    x1 = torch.randn((128, 5120))
    x2 = torch.randn((128, 5120))
    out1 = _model(x1)
    out2 = _model(x2)
    out1.mean().backward()
    out2.mean().backward()

    _optimizer.step()

    # Update the optimizer state memory usage and record the memory state
    profiler.step()
import os
import time
from collections import OrderedDict
from functools import partial, reduce
from typing import Any, Dict, List, Tuple

import pyecharts
import torch

from internlm.core.naive_amp import NaiveAMPModel

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
        self._total_mem = self._layer_mem

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


class ActivationMemState:
    """
    Activation Memory State
    """

    def __init__(self, num_chunks: int) -> None:
        self._num_chunks = num_chunks

        self.inited: List[bool] = [False for _ in range(num_chunks)]
        self.states: List[SimpleMemState] = [SimpleMemState(f"activations_{idx}") for idx in range(num_chunks)]

    @property
    def total_mem(self) -> int:
        return sum(state.total_mem for state in self.states)

    def dump(self, prefix: str = "") -> str:
        return reduce(lambda x, y: x + y, [state.dump(prefix) for state in self.states])

    def to_json(self, base: int = 1024 * 1024) -> List:
        return [state.to_json(base) for state in self.states]


def _unpack_naive_wrapper(model: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
    num_chunks = len(model) if isinstance(model, torch.nn.ModuleList) else 1

    if num_chunks > 1:
        model = torch.nn.ModuleList([_model.model if isinstance(_model, NaiveAMPModel) else _model for _model in model])
    else:
        model = model.model if isinstance(model, NaiveAMPModel) else model

    return model, num_chunks


class SimpleMemoryProfiler:
    """
    A memory profiler for a llm model.

    Args:
        model (torch.nn.Module): The model to profile.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        log_file (str): The file to write the memory state information to.
        total_steps: number of steps to trace.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        log_folder: str,
        total_steps: int = 5,
    ):
        self._model, self._num_model_chunks = _unpack_naive_wrapper(model)
        self._optimizer = optimizer
        self._log_folder = log_folder
        self._remaining_steps = total_steps

        self._stoped = False
        self._record_start_time = time.time()

        # For activation memory state.

        self._activation_mem: int = 0
        self._activation_mem_max: int = 0
        self._activation_base_mems = ActivationMemState(self._num_model_chunks)

        # Check or create log folder
        os.makedirs(self._log_folder, exist_ok=True)

        # Register activation memory tracking hooks
        if self._num_model_chunks > 1:
            for chunk_id in range(self._num_model_chunks):
                self._register_activation_trace_hooks(chunk_id, self._model[chunk_id])
        else:
            self._register_activation_trace_hooks(0, self._model)

        # Calculate static parameter cuda memory
        self._param_mem_state = SimpleMemState("param_mem")
        self._calc_tensor_memory(self._param_mem_state, self._model.named_parameters())
        # Calculate static grad cuda memory
        self._grad_mem_state = SimpleMemState("grad_mem")
        self._calc_tensor_memory(self._grad_mem_state, self._model.named_parameters(), True)
        # Calculate static optimizer state cuda memory
        self._os_params_mem_state = SimpleMemState("os_params_mem")
        self._os_state_mem_state = SimpleMemState("os_state_mem")
        self._calc_tensor_group_memory(self._os_params_mem_state, list(enumerate(self._optimizer.param_groups)))

        # Generate the first memory record
        self.point(with_options="params,grads,os_params", create=True)

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
            layout_info += "activation_base_layout:\n" + self._activation_base_mems.dump()

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
        self._calc_tensor_group_memory(self._os_state_mem_state, list(self._optimizer.state_dict()["state"].items()))

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
            self._render_sunburst_chart(self._activation_base_mems.to_json(), "activation_memory_sunburst")
            # Generate summary sunburst chart
            summary_sunburst_data = [
                {"name": "params", "value": self._param_mem_state.total_mem // mb},
                {"name": "grads", "value": self._grad_mem_state.total_mem // mb},
                {"name": "os_params", "value": self._os_params_mem_state.total_mem // mb},
                {"name": "os_state", "value": self._os_state_mem_state.total_mem // mb},
                {"name": "activation", "value": self._activation_mem_max // mb},
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
                    "r": "35%",
                    "itemStyle": {"borderWidth": 3},
                    "label": {"align": "left"},
                },
                {"r0": "35%", "r": "55%", "label": {"align": "left"}},
                {"r0": "55%", "r": "70%", "label": {"align": "left"}},
                {"r0": "70%", "r": "80%", "label": {"align": "left"}},
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

    def _inner_activation_trace_hook(
        self,
        chunk_id: int,
        layer_name: str,
        model: Any,
        inputs: Any,
        output: torch.Tensor,
    ) -> None:
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

        if self._stoped or self._activation_base_mems.inited[chunk_id]:
            return

        # Delay updating the total_mem of activation_base_mem here, it will be handled in the forward ending hook.
        self._activation_base_mems.states[chunk_id].add(
            layer_name, output.element_size() * output.nelement(), flush=False
        )

    def _activation_trace_hook_forward(self, chunk_id: int, model: Any, inputs: Any, output: torch.Tensor) -> None:
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
        if self._activation_base_mems.inited[chunk_id] is False:
            self._activation_base_mems.inited[chunk_id] = True
            # Update the total memory of the activation base memory state
            self._activation_base_mems.states[chunk_id].update_total_memory()
            # Set with_options to "activation_base" to include activation_base_layout in the memory dump
            with_options = "activation_base"
        else:
            with_options = ""

        # Accumulate activation memory usage for each forward pass
        self._activation_mem += self._activation_base_mems.states[chunk_id].total_mem
        if self._activation_mem > self._activation_mem_max:
            self._activation_mem_max = self._activation_mem

        # Trigger a memory record
        self.point(with_options)

    def _activation_tarce_hook_backward(self, chunk_id: int, model: Any, inputs: Any, grad_outputs: Any) -> None:
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
        self._activation_mem -= self._activation_base_mems.states[chunk_id].total_mem

        # Trigger a memory record
        self.point()

    def _register_activation_trace_hooks(self, chunk_id: int, model_chunk: torch.nn.Module) -> None:
        """
        Register activation trace hooks for the model and each submodule in the model.
        """

        # Register inner activation trace hooks for each submodule in the model
        for layer_name, sub_model in model_chunk.named_modules():
            # Register the hook
            if len(sub_model._modules) != 0:
                continue  # TODO: in some special cases, we may need some additional configuration to correct

            sub_model.register_forward_hook(partial(self._inner_activation_trace_hook, chunk_id, layer_name))

        # Register a forward hook for the main model to track activation memory usage
        model_chunk.register_forward_hook(partial(self._activation_trace_hook_forward, chunk_id))
        # Register a backward hook for the main model to release activation memory usage
        model_chunk.register_full_backward_hook(partial(self._activation_tarce_hook_backward, chunk_id))

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

    def _simple_schedule(_num_chunks, _model_chunks, _input) -> torch.Tensor:
        if _num_chunks > 1:
            _output = _input
            for _model_chunk in _model_chunks:
                _output = _model_chunk(_output)
        else:
            _output = _model_chunks(_input)

        return _output

    # num_chunks config
    _num_chunks = 1

    # init model and optimizer
    if _num_chunks > 1:
        _chunks = [SimpleModel(skip_layer2=idx % 2 == 0) for idx in range(_num_chunks)]
        _model = torch.nn.ModuleList(_chunks).cuda()
    else:
        _model: torch.nn.Module = SimpleModel().cuda()
    _optimizer = torch.optim.Adam(_model.parameters())

    # init profiler
    profiler = SimpleMemoryProfiler(_model, _optimizer, "./test_simple_memory_profiler", total_steps=1)

    _optimizer.zero_grad()

    # inputs
    x1 = torch.randn((128, 5120)).cuda()
    x2 = torch.randn((128, 5120)).cuda()
    # forward
    out1 = _simple_schedule(_num_chunks, _model, x1)
    out2 = _simple_schedule(_num_chunks, _model, x2)
    # backward
    out1.mean().backward()
    out2.mean().backward()

    _optimizer.step()

    # Update the optimizer state memory usage and record the memory state
    profiler.step()

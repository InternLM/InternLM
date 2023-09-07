性能分析
========

.. Mainly about the usage of torch profiler and memory profiler

Torch Profiler
-----------------

InternLM 使用 ``internlm.train.initialize_llm_profile()`` 来收集和分析模型训练或推理期间的性能数据，如 CPU/CUDA/memory 等性能数据。这个实现基于 `torch.profiler <https://pytorch.org/docs/stable/profiler.html>`_ ，输出的性能分析 trace 文件可以使用 `tensorboard <https://www.tensorflow.org>`_ 进行可视化。

用户如果想使用这个 torch 性能分析工具，需要在启动训练时传递 ``--profiling`` 参数以启用性能分析。完成 torch 性能分析后，用户可以在 ``{JOB_NAME}/{start_time}/traces/rank{}_dp{}_tp{}_pp{}`` 文件夹中看到性能分析结果。

.. autofunction:: internlm.train.initialize_llm_profile

Memory Profiler
-----------------

InternLM 提供了一个实用的内存分析工具 ``internlm.utils.simple_memory_profiler.SimpleMemoryProfiler`` 来监控实际的 GPU 内存使用情况。在实现中，会对模型数据（包括模型参数、模型梯度和优化器状态）和非模型数据（包括激活值）分别进行详细的统计。

要使用这个内存分析工具，用户需要在启动训练时传递 ``--profiling`` 参数以启用内存分析。完成内存分析后，用户可以在 ``memory_trace/rank{}_dp{}_tp{}`` 文件夹中找到特定 rank 对应的内存分析结果（包括不同时间点的内存使用日志和显示总体内存使用情况的太阳图表）。

.. autoclass:: internlm.utils.simple_memory_profiler.SimpleMemoryProfiler
    :members:

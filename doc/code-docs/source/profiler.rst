Profiler
========

.. Mainly about the usage of torch profiler and memory profiler

Torch Profiler
-----------------
InternLM uses ``internlm.train.initialize_llm_profile()`` to profile performance data, execution time duration and breakdown analysis of 
step time. The implementation is based on `torch.profiler <https://pytorch.org/docs/stable/profiler.html>`_ and output tracing files can 
be visualized with `tensorboard <https://www.tensorflow.org>`_.

To use this torch profiler tool, you need to enable profiling by passing the ``--profiling`` flag when starting training. After torch 
profiling is completed, you can find the profiling results in the ``{JOB_NAME}/{start_time}/traces/rank{}_dp{}_tp{}_pp{}`` folder.

.. autofunction:: internlm.train.initialize_llm_profile

Memory Profiler
-----------------

InternLM provides a practical solution ``internlm.utils.simple_memory_profiler.SimpleMemoryProfiler`` to monitor actual GPU memory usage.
In the implmentation, model data (including model parameters, model gradients, and optimizer states) and non-model data 
(including activations) are calculated.

To use this memory profiler tool, you need to enable profiling by passing the ``--profiling`` flag when starting training. After memory 
profiling is completed, you can find the profiling results (including logs of memory usage at different time point and sunburst charts 
showing overall memory usage) for a specific rank device in the ``memory_trace/rank{}_dp{}_tp{}`` folder.

.. autoclass:: internlm.utils.simple_memory_profiler.SimpleMemoryProfiler
    :members:

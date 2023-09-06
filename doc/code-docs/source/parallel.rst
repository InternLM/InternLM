Parallel Training
==================

.. Brief introduction to training parallelism, and how-to guide about config setting

InternLM supports tensor parallel, pipeline parallel, sequence parallel, data parallel, and ZeRO1.5 to parallelize the training pipeline. 
When initializing the distributed environment, we need to specify tensor parallel size, pipeline parallel size, data parallel size, 
and ZeRO1.5 strategy. 

The parallel setting of InternLM is fully config-driven, and you can change the parallelism by modifying 
`config file <https://github.com/InternLM/InternLM/blob/main/configs/7B_sft.py>`_. An exmaple parallel training configuration can be defined as follows:

.. code-block:: python

    parallel = dict(
        zero1=8,
        tensor=1,
        pipeline=dict(size=1, interleaved_overlap=True),
        sequence_parallel=False,
    )

- zero1: zero parallel strategy, divided into the following three cases, the default value is -1

    - When ``size <= 0``, the size of the zero1 process group is equal to the size of the data parallel process group, so the optimizer state parameters will be split within the data parallel range.
    - When ``size == 1``, zero1 is not used, and all data parallel groups retain the complete optimizer state parameters.
    - When ``size > 1`` and ``size <= data_parallel_world_size``, the zero1 process group is a subset of the data parallel process group.

- tensor: tensor parallel size, usually the number of GPUs per node, the default value is 1
- pipeline: pipeline parallel strategy

    - size: pipeline parallel size, the default value is 1
    - interleaved_overlap: bool type, when interleaved scheduling, enable or disable communication optimization, the default value is False

- sequence_parallel: whether to enable sequence parallelism, the default value is False

Note: `Data parallel size = Total number of GPUs / Pipeline parallel size / Tensor parallel size`

Tensor Parallel
-----------------

The implementation of tensor parallel for InternLM is based on `flash attention <https://github.com/Dao-AILab/flash-attention>`_, which has tensor 
parallel extensions to parallelize `attention <https://github.com/InternLM/InternLM/blob/main/internlm/model/multi_head_attention.py>`_ and 
`linear <https://github.com/InternLM/InternLM/blob/main/internlm/model/linear.py>`_ blocks in InternLM model. 

To use tensor parallel, you need to set the value of tensor parallel size ``parallel.tensor`` in the config file, which is usually the number of GPUs per node.

.. figure:: ../../imgs/tensor_parallel.png
  :scale: 50%
  :class: with-border

  Tensor parallel, adopted from `flash-attention <https://arxiv.org/pdf/2205.14135.pdf>`_

Pipeline Parallel
-----------------

InternLM uses `1F1B <https://arxiv.org/pdf/2104.04473.pdf>`_ (one forward pass followed by one backward pass) for pipeline parallel. For 1F1B strategy, there are two implementations: 
(1) non-interleaved scheduler, which is memory-efficient (2) interleaved scheduler, which is both memory-efficient and time-efficient.

.. figure:: ../../imgs/pipeline_schedule.png
  :scale: 45%
  :class: with-border

  Non-interleaved and interleaved scheduler for 1F1B pipeline parallelism, adopted from `Megatron-LM <https://arxiv.org/pdf/2104.04473.pdf>`_

scheduler for non-interleaved 1F1B strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To use non-interleaved pipeline scheduler, you need to set ``model.num_chunks = 1`` in the config file.

.. autoclass:: internlm.core.scheduler.pipeline_scheduler.PipelineScheduler
    :members:

scheduler for interleaved 1F1B strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To use interleaved pipeline scheduler, you need to set ``model.num_chunks > 1`` in the config file.

.. autoclass:: internlm.core.scheduler.pipeline_scheduler.InterleavedPipelineScheduler
    :members:

Also, to enable communication overlap when using interleaved pipeline scheduler, you need to set ``parallel.pipeline.interleaved_overlap = True`` 
in the config file.

When ``parallel.pipeline.interleaved_overlap = True``, function ``InterleavedPipelineScheduler._run_1f1b_loop_with_overlap`` will be called and 
``internlm.core.communication.AsynCommunicator`` will be created for managing async communication. Asynchronous communication will be enabled in 1F1B stage to make full 
use of uplink/downlink bandwidth and achieve communication overlap. 

The difference between 1F1B stage without overlap and 1F1B stage with overlap is shown as follows:

The 1F1B stage without overlap consists of the following steps:

.. code-block:: bash

    1. Perform the forward pass.
    2. Perform the backward pass.
    3. Send the forward output of this iteration to the next stage, and send the backward output of this iteration to the previous stage, and receive the forward and backward inputs for the next iteration.

The 1F1B stage with overlap consists of the following steps:

.. code-block:: bash

    1. Perform the forward pass.
    2. Check if the backward input is ready.
    3. Send the forward output and receive the forward input for the next iteration.
    4. Perform the backward pass.
    5. Check if the forward input is ready.
    6. Send the backward output and receive the backward input for the next iteration.


Sequence Parallel
-----------------

Sequence parallel is a technique to reduce activation memory in layer norm and dropout without additional computation, communication or memory overhead.
The implementation of sequence parallel for InternLM is based on `flash attention <https://github.com/Dao-AILab/flash-attention>`_. 

To enable sequence parallel, you need to set ``parallel.sequence_parallel = True`` in the config file.

.. figure:: ../../imgs/sequence_parallel.png
  :scale: 50%
  :class: with-border

  Sequence parallel, adopted from flash-attention

Data Parallel
-----------------

InternLM supports data parallel. For data parallel:

`Data parallel size = Total number of GPUs / Pipeline parallel size / Tensor parallel size`

ZeRO1.5
-----------------

The implementation of ZeRO1.5 uses the concept of hierarchical sharding via config value ``parallel.zero1``, which enables sharding within local nodes.

1. If ``parallel.zero1 <= 0``, the size of the zero process group is equal to the size of the dp process group, so parameters will be divided within the range of dp.
2. If ``parallel.zero1 == 1``, zero is not used, and all dp groups retain the full amount of model parameters.
3. If ``parallel.zero1 > 1`` and ``parallel.zero1 <= dp world size``, the world size of zero is a subset of dp world size. For smaller models, it is usually a better choice to split the parameters within nodes with a setting ``parallel.zero1 <= 8``.

Furthermore, you can enable communication-computation overlap, set bucket reduce size, gradient clipping parameters in the config file.

.. code-block:: python

    hybrid_zero_optimizer = dict(
        # Enable low_level_optimzer overlap_communication
        overlap_sync_grad=True,  
        overlap_sync_param=True,
        # bucket size for nccl communication params
        reduce_bucket_size=512 * 1024 * 1024,
        # grad clipping
        clip_grad_norm=1.0,
    )

There are two communication optimizations worth paying attention to here:

- ``overlap_sync_grad``: If set True, overlapping training backward pass with gradients' all-reduce communication
- ``overlap_sync_param``: If set True, overlapping parameters' broadcast communication with next step's forward pass

.. autoclass:: internlm.solver.optimizer.hybrid_zero_optim.HybridZeroOptimizer
    :members:

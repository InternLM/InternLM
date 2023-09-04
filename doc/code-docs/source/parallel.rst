Parallel Training
=================

.. Brief introduction to training parallelism, and how-to guide about config setting

InternLM supports tensor parallel, pipeline parallel, sequence parallel, data parallel, and ZeRO1.5 to parallelize the training pipeline. 
When initializing the distributed environment, we need to specify tensor parallel size, pipeline parallel size, data parallel size, 
and ZeRO1.5 strategy. 

The parallel setting of InternLM is fully config-driven, and you can change the parallelism by modifying 
`config file <https://github.com/InternLM/InternLM/blob/main/configs/7B_sft.py>`_.

.. code-block:: python

    """
    zero1 parallel:
        1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
            For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
    pipeline parallel (dict):
        1. size: int, the size of pipeline parallel.
        2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
    tensor parallel: tensor parallel size, usually the number of GPUs per node.
    """
    parallel = dict(
        zero1=8,
        pipeline=dict(size=1, interleaved_overlap=True),
        sequence_parallel=False,
    )

Note: `Total number of GPUs = tensor parallel size * pipeline parallel size * data parallel size`

Tensor Parallel
-----------------

The implementation of tensor parallel for InternLM is based on `flash attention <https://github.com/Dao-AILab/flash-attention>`_, which has tensor 
parallel extensions to parallelize `attention <https://github.com/InternLM/InternLM/blob/main/internlm/model/multi_head_attention.py>`_ and 
`linear <https://github.com/InternLM/InternLM/blob/main/internlm/model/linear.py>`_ blocks in InternLM model. 

To use tensor parallel, you need to set the value of tensor parallel size ``parallel.tensor`` in the config file, which is usually the number of GPUs per node.

.. figure:: ../../imgs/tensor_parallel.png
  :scale: 50%
  :class: with-border

  Tensor parallel, adopted from flash-attention original paper

Pipeline Parallel
-----------------

InternLM uses `1F1B <https://arxiv.org/pdf/2104.04473.pdf>`_ (one forward pass followed by one backward pass) for pipeline parallel. For 1F1B strategy, there are two implementations: 
(1) non-interleaved scheduler, which is memory-efficient (2) interleaved scheduler, which is both memory-efficient and time-efficient.

.. figure:: ../../imgs/pipeline_schedule.png
  :scale: 45%
  :class: with-border

  Non-interleaved and interleaved scheduler for 1F1B pipeline parallelism, adopted from Megatron-LM original paper

scheduler for non-interleaved 1F1B strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To use non-interleaved pipeline scheduler, you need to set ``model.num_chunks = 1`` in the config file.

.. autoclass:: internlm.core.scheduler.pipeline_scheduler.PipelineScheduler
    :members:

scheduler for interleaved 1F1B strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To use interleaved pipeline scheduler, you need to set ``model.num_chunks > 1`` in the config file.

Also, to enable communication overlap when using interleaved pipeline scheduler, you need to set ``parallel.pipeline.interleaved_overlap = True`` 
in the config file.

.. autoclass:: internlm.core.scheduler.pipeline_scheduler.InterleavedPipelineScheduler
    :members:

Sequence Parallel
-----------------

Sequence parallel is a technique to reduce activation memory in layer norm and dropout without additional computation, communication or memory overhead.
The implementation of sequence parallel for InternLM is based on `flash attention <https://github.com/Dao-AILab/flash-attention>`_. 

To enable sequence parallel, you need to set ``parallel.sequence_parallel = True`` in the config file.

.. figure:: ../../imgs/sequence_parallel.png
  :scale: 50%
  :class: with-border

  Sequence parallel, adopted from flash-attention original paper

Data Parallel
-----------------

InternLM supports both vanilla data parallel and ZeRO1.5 (an optimized implementation of ZeRO).

To enable vanilla data parallel, you need to set ``parallel.zero1 == 1`` in the config file.

ZeRO1.5
-----------------

The implementation of ZeRO1.5 uses the concept of hierarchical sharding via config value ``parallel.zero1``, which enables sharding within local nodes.

1. If ``parallel.zero1 <= 0``, the size of the zero process group is equal to the size of the dp process group, so parameters will be divided within the range of dp.
2. If ``parallel.zero1 == 1``, zero is not used, and all dp groups retain the full amount of model parameters.
3. If ``parallel.zero1 > 1`` and ``parallel.zero1 <= dp world size``, the world size of zero is a subset of dp world size. For smaller models, it is usually a better choice to split the parameters within nodes with a setting ``parallel.zero1 <= 8``.

Furthermore, you can enable communication-computation overlap, bucket reduce operation, gradient clipping in the config file.

.. code-block:: python

    hybrid_zero_optimizer = dict(
        # Enable low_level_optimzer overlap_communication
        zero_overlap_communication=True,
        # bucket size for nccl communication params
        reduce_bucket_size=512 * 1024 * 1024,
        # grad clipping
        clip_grad_norm=1.0,
    )

.. autoclass:: internlm.solver.optimizer.hybrid_zero_optim.HybridZeroOptimizer
    :members:
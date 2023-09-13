训练 API
============

InternLM 的训练 API 由 ``internlm.core.trainer.Trainer`` 管理。在定义了训练引擎和调度器之后，我们可以调用 Trainer API 来执行模型训练、评估、梯度清零和参数更新等。

InternLM 的训练流程可以归纳为两个步骤：

1. 初始化

    * 初始化模型、优化器、数据加载器、Trainer，生成不同种类的进程组，为混合并行的迭代训练做准备。
    * 初始化Logger、Checkpoint管理器、Monitor管理器、Profiler，对迭代训练的过程观察、预警、记录。

2. 迭代训练
   
    * 根据配置文件定义的张量并行、流水线并行、数据并行的大小，加载训练引擎和调度器进行混合并行训练。
    * 在迭代训练中，调用 Trainer API 进行梯度置零，前向传播计算损失并反向传播，参数更新。

.. figure:: ../../imgs/hybrid_parallel_training.png
  :scale: 45%
  :class: with-border

  InternLM训练流程图

有关详细用法，请参阅 Trainer API 文档和示例。

.. autoclass:: internlm.core.trainer.Trainer
    :members:
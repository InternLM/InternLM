训练 API
============

InternLM 的训练 API 由 ``internlm.core.trainer.Trainer`` 管理。在定义了训练引擎和调度器之后，我们可以调用 Trainer API 来执行模型训练、评估、梯度清零和参数更新等。

有关详细用法，请参阅 Trainer API 文档和示例。

.. autoclass:: internlm.core.trainer.Trainer
    :members:
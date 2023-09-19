模型保存
===================

InternLM 使用 ``internlm.utils.model_checkpoint.CheckpointManager`` 来管理模型保存。其中，可以使用 ``CheckpointManager.try_save_checkpoint(train_state)`` 来保存指定 step 的模型状态。

InternLM支持启动时自动加载最新的模型备份，并在接收信号退出训练时自动进行模型备份。

Checkpointing
-------------

.. autoclass:: internlm.utils.model_checkpoint.CheckpointManager
    :members:

Model Checkpointing
===================

InternLM uses ``internlm.utils.model_checkpoint.CheckpointManager`` to manage model checkpointing. In the implementation, 
we use ``CheckpointManager.try_save_checkpoint(train_state)`` to checkpoint training states at specific steps. InternLM supports 
automatic loading of latest ckpt at startup and automatic model checkpointing at signal quit.

Checkpointing
-------------

.. autoclass:: internlm.utils.model_checkpoint.CheckpointManager
    :members:

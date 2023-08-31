Training API
============

InternLM training API is managed in ``internlm.core.engine.Engine``. After defining the model, optimizer, criterion, train_dataloader, and lr_scheduler, 
we can call training engine API to perform zero gradients, forward pass, backward pass, and parameter update.

For detailed usage, please refer to API documentation and examples.

.. autoclass:: internlm.core.engine.Engine
    :members:

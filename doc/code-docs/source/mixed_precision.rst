混合精度
-----------------
混合精度是指在模型训练的过程中同时使用16位和32位浮点数类型，是一种在最小化精度损失的前提下加速模型训练的方法。
混合精度通过让模型的某些部分使用32位浮点数以保持数值稳定性，并在其余部分利用半精度浮点数加速训练并可以减少内存使用，在评估指标（如准确率）方面仍可以获得同等的训练效果。

.. autoclass:: internlm.core.naive_amp.NaiveAMPModel

InternLM默认将模型转换为16位浮点数类型进行训练（在配置文件中可以设置默认类型为其他数据类型）。在使用混合精度时，需要在构建模型时使用

.. code-block:: python

    set_fp32_attr_to_module(/*fp32 module*/)

将模型的某个子模块设置为32位浮点数类型进行训练，InternLM会在模型训练时自动将数据类型转换成需要的精度。

例如：

.. code-block:: python

    class MlpModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 1, bias=False)
            self.linear2 = nn.Linear(1, 4, bias=False)

    model = MlpModel()
    # set model.linear2 as fp32 module
    set_fp32_attr_to_module(model.linear2)

    # apply mixed precision
    model = NaiveAMPModel(
        model=model,
        output_to_fp32=True,
        dtype=torch.bfloat16(),
        sync_buffer=False,
    )

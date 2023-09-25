监控和告警
=================

监控
-----------------

InternLM 使用 ``internlm.monitor.monitor.initialize_monitor_manager()`` 来初始化上下文监控管理。其中，一个实例化的单例对象 ``internlm.monitor.monitor.MonitorManager`` 将管理监控线程并使用 ``internlm.monitor.monitor.MonitorTracker`` 来跟踪模型训练生命周期和训练状态。

.. autofunction:: internlm.monitor.monitor.initialize_monitor_manager

.. autoclass:: internlm.monitor.monitor.MonitorManager
    :members:

.. autoclass:: internlm.monitor.monitor.MonitorTracker
    :members:

告警
-----------------

InternLM 监控线程会周期性地检查模型训练过程中是否出现 loss spike、潜在的 training stuck、运行时异常等，并捕获 SIGTERM 异常信号。当出现上述情况时，将触发警报，并通过调用 ``internlm.monitor.alert.send_feishu_msg_with_webhook()`` 向飞书的 Webhook 地址发送报警消息。

.. autofunction:: internlm.monitor.alert.send_feishu_msg_with_webhook

轻量监控
-----------------

InternLM轻量级监控工具采用心跳机制实时监测训练过程中的各项指标，如loss、grad_norm、训练阶段的耗时等。同时，InternLM还可以通过 `grafana dashboard <https://grafana.com/grafana/dashboards/>`_ 直观地呈现这些指标信息，以便用户进行更加全面和深入的训练分析。

轻量监控的配置由配置文件中的 ``monitor`` 字段指定， 用户可以通过修改配置文件 `config file <https://github.com/InternLM/InternLM/blob/main/configs/7B_sft.py>`_ 来更改监控配置。以下是一个监控配置的示例：

.. code-block:: python

    monitor = dict(
        alert=dict(
            enable_feishu_alert=False,
            feishu_alert_address=None,
            light_monitor_address=None,
        ),
    )

- enable_feishu_alert：是否启用飞书告警
- feishu_alert_address：飞书告警的 Webhook 地址
- light_monitor_address：轻量监控的地址

InternLM 使用 ``internlm.monitor.alert.initialize_light_monitor`` 来初始化轻量监控客户端。一旦初始化完成，它会建立与监控服务器的连接。在训练过程中，使用 ``internlm.monitor.alert.send_heartbeat`` 来发送不同类型的心跳信息至监控服务器。监控服务器会根据这些心跳信息来检测训练是否出现异常，并在需要时发送警报消息。

.. autofunction:: internlm.monitor.alert.initialize_light_monitor

.. autofunction:: internlm.monitor.alert.send_heartbeat

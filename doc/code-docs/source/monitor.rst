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

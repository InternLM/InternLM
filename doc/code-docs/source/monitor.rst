Monitor and Alert
=================

Monitoring
-----------------

InternLM uses ``internlm.monitor.monitor.initialize_monitor_manager()`` to initialize context monitor. During this time, 
a singleton ``internlm.monitor.monitor.MonitorManager`` will manage monitoring thread and track training status 
with ``internlm.monitor.monitor.MonitorTracker``. 

.. autofunction:: internlm.monitor.monitor.initialize_monitor_manager

.. autoclass:: internlm.monitor.monitor.MonitorManager
    :members:

.. autoclass:: internlm.monitor.monitor.MonitorTracker
    :members:

Alerting
-----------------

InternLM monitor thread periodically tracks loss spike, potential stuck condition, runtime exception, and SIGTERM signal.
When above situation occurs, an alert will be triggered and a message will be sent to the Feishu webhook address by calling
``internlm.monitor.alert.send_feishu_msg_with_webhook()``

.. autofunction:: internlm.monitor.alert.send_feishu_msg_with_webhook

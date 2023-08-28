## InternLM系统结构
本项目系统代码文件结构如下所示：
```bash
├── configs                                  # 配置模块，管理模型和训练相关参数
│   └── 7B_sft.py                            # 7B_sft.py 是系统 demo 的配置文件样例
├── internlm                                 # 系统代码的主目录
│   ├── apis                                 # 接口模块，包含一些关于推理等的接口函数
│   ├── core                                 # 核心模块，管理用于训练和推理的 parallel context 和训练调度引擎
│   │   ├── communication                    # 通信模块，负责流水线并行调度中的p2p通信
│   │   ├── context                          # context 模块，主要负责初始化并行进程组，并管理 parallel context
│   │   │   ├── parallel_context.py
│   │   │   └── process_group_initializer.py
│   │   ├── scheduler                        # 调度模块，管理并行训练的调度器，包括非流水线并行调度器和流水线并行调度器
│   │   │   ├── no_pipeline_scheduler.py
│   │   │   └── pipeline_scheduler.py
│   │   ├── engine.py                        # 负责管理模型的训练和评估过程
│   │   └── trainer.py                       # 负责管理训练引擎和调度器
│   ├── data                                 # 数据模块，负责管理数据集生成和处理
│   ├── initialize                           # 初始化模块，负责管理分布式环境启动和训练器初始化
│   ├── model                                # 模型模块，负责管理模型结构定义和实现
│   ├── solver                               # 负责管理 optimizer 和 lr_scheduler 等的实现
│   └── utils                                # 辅助模块，负责管理日志、存储、模型注册等
├── train.py                                 # 模型训练的主函数入口文件
├── requirements                             # 系统运行的依赖包列表
├── third_party                              # 系统所依赖的第三方模块，包括 apex 和 flash-attention 等
├── tools                                    # 一些脚本工具，用于原始数据集处理和转换，模型 checkpoint 转换等
└── version.txt                              # 系统版本号
```

## InternLM System Structure
The system code file structure is shown below:
```bash
├── configs                                  # Configuration module, managing model and training-related parameters
│   └── 7B_sft.py                            # 7B_sft.py is a sample configuration file for the system demo
├── internlm                                 # Main directory of the system code
│   ├── apis                                 # Interface module, containing some interface functions related to inference, etc.
│   ├── core                                 # Core module, managing parallel context and training scheduling engine for training and inference
│   │   ├── communication                    # Communication module, responsible for p2p communication in pipeline parallel scheduling
│   │   ├── context                          # Context module, mainly responsible for initializing parallel process groups and managing parallel context
│   │   │   ├── parallel_context.py
│   │   │   └── process_group_initializer.py
│   │   ├── scheduler                        # Scheduling module, which manages schedulers for parallel training, including non-pipeline and pipeline parallel schedulers
│   │   │   ├── no_pipeline_scheduler.py
│   │   │   └── pipeline_scheduler.py
│   │   ├── engine.py                        # Responsible for managing the training and evaluation process of the model
│   │   └── trainer.py                       # Responsible for managing the training engine and scheduler
│   ├── data                                 # Data module, responsible for managing dataset generation and processing
│   ├── initialize                           # Initialization module, responsible for managing distributed environment startup and trainer initialization
│   ├── model                                # Model module, responsible for managing model structure definition and implementation
│   ├── solver                               # Responsible for managing the implementation of optimizer and lr_scheduler, etc.
│   └── utils                                # Auxiliary module, responsible for managing logs, storage, model registration, etc.
├── train.py                                 # Main function entry file for model training
├── requirements                             # List of dependent packages for system running
├── third_party                              # Third-party modules on which the system depends, including apex and flash-attention, etc.
├── tools                                    # Some script tools for processing and converting raw datasets, model checkpoint conversion, etc.
└── version.txt                              # System version number
```

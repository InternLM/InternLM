20B Demo
================

训练配置
----------------

20B demo 训练配置文件样例如下:

.. code-block:: python

    JOB_NAME = "20b_train"

    SEQ_LEN = 2048
    HIDDEN_SIZE = 5120
    NUM_ATTENTION_HEAD = 40
    MLP_RATIO = 8 / 3
    NUM_LAYER = 60
    VOCAB_SIZE = 103168

    MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
    # Ckpt folder format:
    # fs: 'local:/mnt/nfs/XXX'
    SAVE_CKPT_FOLDER = "local:llm_ckpts"
    LOAD_CKPT_FOLDER = "local:llm_ckpts/49"

    # boto3 Ckpt folder format:
    # import os
    # BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
    # SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
    # LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
    CHECKPOINT_EVERY = 50
    ckpt = dict(
        enable_save_ckpt=False,  # enable ckpt save.
        save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
        # load_ckpt_folder=LOAD_CKPT_FOLDER, # Ckpt path to resume training(load weights and scheduler/context states).
        # load_model_only_folder=MODEL_ONLY_FOLDER, # Path to initialize with given model weights.
        load_optimizer=True,  # Wheter to load optimizer states when continuing training.
        checkpoint_every=CHECKPOINT_EVERY,
        async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
        async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
        snapshot_ckpt_folder="/".join([SAVE_CKPT_FOLDER, "snapshot"]),  # directory for snapshot ckpt storage path.
        oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
    )

    TRAIN_FOLDER = "/path/to/dataset"
    VALID_FOLDER = "/path/to/dataset"
    data = dict(
        seq_len=SEQ_LEN,
        # micro_num means the number of micro_batch contained in one gradient update
        micro_num=4,
        # packed_length = micro_bsz * SEQ_LEN
        micro_bsz=2,
        # defaults to the value of micro_num
        valid_micro_num=4,
        # defaults to 0, means disable evaluate
        valid_every=50,
        pack_sample_into_one=False,
        total_steps=50000,
        skip_batches="",
        rampup_batch_size="",
        # Datasets with less than 50 rows will be discarded
        min_length=50,
        # train_folder=TRAIN_FOLDER,
        # valid_folder=VALID_FOLDER,
    )

    grad_scaler = dict(
        fp16=dict(
            # the initial loss scale, defaults to 2**16
            initial_scale=2**16,
            # the minimum loss scale, defaults to None
            min_scale=1,
            # the number of steps to increase loss scale when no overflow occurs
            growth_interval=1000,
        ),
        # the multiplication factor for increasing loss scale, defaults to 2
        growth_factor=2,
        # the multiplication factor for decreasing loss scale, defaults to 0.5
        backoff_factor=0.5,
        # the maximum loss scale, defaults to None
        max_scale=2**24,
        # the number of overflows before decreasing loss scale, defaults to 2
        hysteresis=2,
    )

    hybrid_zero_optimizer = dict(
        # Enable low_level_optimzer overlap_communication
        overlap_sync_grad=True,
        overlap_sync_param=True,
        # bucket size for nccl communication params
        reduce_bucket_size=512 * 1024 * 1024,
        # grad clipping
        clip_grad_norm=1.0,
    )

    loss = dict(
        label_smoothing=0,
    )

    adam = dict(
        lr=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_beta2_c=0,
        adam_eps=1e-8,
        weight_decay=0.01,
    )

    lr_scheduler = dict(
        total_steps=data["total_steps"],
        init_steps=0,  # optimizer_warmup_step
        warmup_ratio=0.01,
        eta_min=1e-5,
        last_epoch=-1,
    )

    beta2_scheduler = dict(
        init_beta2=adam["adam_beta2"],
        c=adam["adam_beta2_c"],
        cur_iter=-1,
    )

    model = dict(
        checkpoint=False,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
        num_attention_heads=NUM_ATTENTION_HEAD,
        embed_split_hidden=True,
        vocab_size=VOCAB_SIZE,
        embed_grad_scale=1,
        parallel_output=True,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYER,
        mlp_ratio=MLP_RATIO,
        apply_post_layer_norm=False,
        dtype="torch.float16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
        norm_type="rmsnorm",
        layer_norm_epsilon=1e-5,
        use_flash_attn=True,
        num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    )
    """
    zero1 parallel:
        1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
            For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
    pipeline parallel (dict):
        1. size: int, the size of pipeline parallel.
        2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
    tensor parallel: tensor parallel size, usually the number of GPUs per node.
    """
    parallel = dict(
        zero1=-1,
        tensor=4,
        pipeline=dict(size=1, interleaved_overlap=True),
        sequence_parallel=False,
    )

    cudnn_deterministic = False
    cudnn_benchmark = False


启动训练
----------------

完成以上训练配置后，可启动模型训练，以在 ``slurm`` 平台上为例，启动两节点 16GPU 的训练命令如下所示：

.. code-block:: bash

    srun -p internllm -N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/20B_sft.py

训练结果
----------------

基于以上训练配置和启动命令，两节点 16GPU 下的模型训练部分日志展示如下：

.. code-block:: bash

    2023-11-10 11:45:20,248 INFO parallel_context.py:555 in set_device -- process rank 0 is bound to host:HOST-10-140-60-69 device: 0
    2023-11-10 11:45:20,287 INFO parallel_context.py:555 in set_device -- process rank 10 is bound to host:HOST-10-140-60-95 device: 2
    2023-11-10 11:45:20,289 INFO parallel_context.py:555 in set_device -- process rank 12 is bound to host:HOST-10-140-60-95 device: 4
    2023-11-10 11:45:20,291 INFO parallel_context.py:555 in set_device -- process rank 9 is bound to host:HOST-10-140-60-95 device: 1
    2023-11-10 11:45:20,291 INFO parallel_context.py:555 in set_device -- process rank 13 is bound to host:HOST-10-140-60-95 device: 5
    2023-11-10 11:45:20,292 INFO parallel_context.py:555 in set_device -- process rank 8 is bound to host:HOST-10-140-60-95 device: 0
    2023-11-10 11:45:20,292 INFO parallel_context.py:555 in set_device -- process rank 15 is bound to host:HOST-10-140-60-95 device: 7
    2023-11-10 11:45:20,292 INFO parallel_context.py:555 in set_device -- process rank 14 is bound to host:HOST-10-140-60-95 device: 6
    2023-11-10 11:45:20,292 INFO parallel_context.py:555 in set_device -- process rank 11 is bound to host:HOST-10-140-60-95 device: 3
    2023-11-10 11:45:20,298 INFO parallel_context.py:555 in set_device -- process rank 6 is bound to host:HOST-10-140-60-69 device: 6
    2023-11-10 11:45:20,340 INFO parallel_context.py:555 in set_device -- process rank 7 is bound to host:HOST-10-140-60-69 device: 7
    2023-11-10 11:45:20,387 INFO parallel_context.py:555 in set_device -- process rank 2 is bound to host:HOST-10-140-60-69 device: 2
    2023-11-10 11:45:20,387 INFO parallel_context.py:555 in set_device -- process rank 5 is bound to host:HOST-10-140-60-69 device: 5
    2023-11-10 11:45:20,388 INFO parallel_context.py:555 in set_device -- process rank 1 is bound to host:HOST-10-140-60-69 device: 1
    2023-11-10 11:45:20,390 INFO parallel_context.py:555 in set_device -- process rank 4 is bound to host:HOST-10-140-60-69 device: 4
    2023-11-10 11:45:20,463 INFO parallel_context.py:555 in set_device -- process rank 3 is bound to host:HOST-10-140-60-69 device: 3
    2023-11-10 11:45:25,162 INFO launch.py:409 in launch -- Distributed environment is initialized, data parallel size: 4, pipeline parallel size: 1, tensor parallel size: 4
    2023-11-10 11:45:40,621 INFO hybrid_zero_optim.py:268 in _partition_param_list -- Number of elements on ranks: [1262168320, 1269084160, 1269084160, 1222844160], rank:0
    2023-11-10T11:46:16.409+08:00 INFO [training_internlm.py, line 600, in record_current_batch_training_metrics] - pid=117775 : tflops=30.535171880622176 step=0 loss=11.542577743530273 tgs (tokens/gpu/second)=246.32 tgs/last_tgs_1=246.32 tgs/tgs_all=246.32 tgs/tgs_avg=246.32 tgs/tgs_SMA=246.32 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=4.0000000000000003e-07 loss_scale=65536.0 grad_norm={'0_default': 87.3189924662012, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=65536 inf_nan_skip_batches=0 num_samples_in_batch=18 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=13.47 acc=0.0 perplexity=104321.0312 acc/.git=0.0 acc/.github=0.0 acc/.gitignore=0.0 acc/.gitmodules=0.0 acc/.owners.yml=0.0 acc/.pre-commit-config.yaml=0.0 acc/.pylintrc=0.0 acc/.pytest_cache=0.0 acc/.readthedocs.yml=0.0 acc/.vscode=0.0 acc/7b_train=0.0 acc/CHANGE_LOG.md=0.0 acc/LICENSE=0.0 acc/README-ja-JP.md=0.0 acc/README-zh-Hans.md=0.0 acc/README.md=0.0 acc/RUN=0.0 acc/ci_scripts=0.0 acc/configs=0.0 acc/doc=0.0 acc/docker=0.0 acc/docker.Makefile=0.0 acc/experiment=0.0 acc/internlm=0.0 acc/requirements=0.0 acc/sonar-project.properties=0.0 acc/tests=0.0 acc/third_party=0.0 acc/tools=0.0 acc/train.py=0.0 acc/version.txt=0.0 acc/web_demo.py=0.0 acc/web_demo_internlm.py=0.0 tokens/.git=60571 tokens/.github=0 tokens/.gitignore=0 tokens/.gitmodules=0 tokens/.owners.yml=0 tokens/.pre-commit-config.yaml=0 tokens/.pylintrc=0 tokens/.pytest_cache=0 tokens/.readthedocs.yml=0 tokens/.vscode=0 tokens/7b_train=0 tokens/CHANGE_LOG.md=0 tokens/LICENSE=0 tokens/README-ja-JP.md=0 tokens/README-zh-Hans.md=0 tokens/README.md=0 tokens/RUN=0 tokens/ci_scripts=0 tokens/configs=0 tokens/doc=0 tokens/docker=0 tokens/docker.Makefile=0 tokens/experiment=0 tokens/internlm=0 tokens/requirements=0 tokens/sonar-project.properties=0 tokens/tests=0 tokens/third_party=0 tokens/tools=0 tokens/train.py=0 tokens/version.txt=0 tokens/web_demo.py=0 tokens/web_demo_internlm.py=0 loss_from_metric=11.5552 loss/.git=11.5552 loss/.github=nan loss/.gitignore=nan loss/.gitmodules=nan loss/.owners.yml=nan loss/.pre-commit-config.yaml=nan loss/.pylintrc=nan loss/.pytest_cache=nan loss/.readthedocs.yml=nan loss/.vscode=nan loss/7b_train=nan loss/CHANGE_LOG.md=nan loss/LICENSE=nan loss/README-ja-JP.md=nan loss/README-zh-Hans.md=nan loss/README.md=nan loss/RUN=nan loss/ci_scripts=nan loss/configs=nan loss/doc=nan loss/docker=nan loss/docker.Makefile=nan loss/experiment=nan loss/internlm=nan loss/requirements=nan loss/sonar-project.properties=nan loss/tests=nan loss/third_party=nan loss/tools=nan loss/train.py=nan loss/version.txt=nan loss/web_demo.py=nan loss/web_demo_internlm.py=nan 
    2023-11-10T11:46:20.794+08:00 INFO [training_internlm.py, line 600, in record_current_batch_training_metrics] - pid=117775 : tflops=119.67196090960911 step=1 loss=11.337997436523438 tgs (tokens/gpu/second)=965.36 tgs/last_tgs_1=965.37 tgs/tgs_all=392.49 tgs/tgs_avg=605.85 tgs/tgs_SMA=392.49 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=6.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 90.85007610412333, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=131072 inf_nan_skip_batches=0 num_samples_in_batch=19 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.7 acc=0.0 perplexity=81555.5 acc/.git=0.0 acc/.github=0.0 acc/.gitignore=0.0 acc/.gitmodules=0.0 acc/.owners.yml=0.0 acc/.pre-commit-config.yaml=0.0 acc/.pylintrc=0.0 acc/.pytest_cache=0.0 acc/.readthedocs.yml=0.0 acc/.vscode=0.0 acc/7b_train=0.0 acc/CHANGE_LOG.md=0.0 acc/LICENSE=0.0 acc/README-ja-JP.md=0.0 acc/README-zh-Hans.md=0.0 acc/README.md=0.0 acc/RUN=0.0 acc/ci_scripts=0.0 acc/configs=0.0 acc/doc=0.0 acc/docker=0.0 acc/docker.Makefile=0.0 acc/experiment=0.0 acc/internlm=0.0 acc/requirements=0.0 acc/sonar-project.properties=0.0 acc/tests=0.0 acc/third_party=0.0 acc/tools=0.0 acc/train.py=0.0 acc/version.txt=0.0 acc/web_demo.py=0.0 acc/web_demo_internlm.py=0.0 tokens/.git=60265 tokens/.github=0 tokens/.gitignore=0 tokens/.gitmodules=0 tokens/.owners.yml=0 tokens/.pre-commit-config.yaml=0 tokens/.pylintrc=0 tokens/.pytest_cache=0 tokens/.readthedocs.yml=0 tokens/.vscode=0 tokens/7b_train=0 tokens/CHANGE_LOG.md=0 tokens/LICENSE=0 tokens/README-ja-JP.md=0 tokens/README-zh-Hans.md=0 tokens/README.md=0 tokens/RUN=0 tokens/ci_scripts=0 tokens/configs=0 tokens/doc=0 tokens/docker=0 tokens/docker.Makefile=0 tokens/experiment=0 tokens/internlm=0 tokens/requirements=0 tokens/sonar-project.properties=0 tokens/tests=0 tokens/third_party=0 tokens/tools=0 tokens/train.py=0 tokens/version.txt=0 tokens/web_demo.py=0 tokens/web_demo_internlm.py=0 loss_from_metric=11.309 loss/.git=11.309 loss/.github=nan loss/.gitignore=nan loss/.gitmodules=nan loss/.owners.yml=nan loss/.pre-commit-config.yaml=nan loss/.pylintrc=nan loss/.pytest_cache=nan loss/.readthedocs.yml=nan loss/.vscode=nan loss/7b_train=nan loss/CHANGE_LOG.md=nan loss/LICENSE=nan loss/README-ja-JP.md=nan loss/README-zh-Hans.md=nan loss/README.md=nan loss/RUN=nan loss/ci_scripts=nan loss/configs=nan loss/doc=nan loss/docker=nan loss/docker.Makefile=nan loss/experiment=nan loss/internlm=nan loss/requirements=nan loss/sonar-project.properties=nan loss/tests=nan loss/third_party=nan loss/tools=nan loss/train.py=nan loss/version.txt=nan loss/web_demo.py=nan loss/web_demo_internlm.py=nan 
    2023-11-10T11:46:24.921+08:00 INFO [training_internlm.py, line 600, in record_current_batch_training_metrics] - pid=117775 : tflops=127.02177898638753 step=2 loss=10.111491203308105 tgs (tokens/gpu/second)=1024.65 tgs/last_tgs_1=1024.66 tgs/tgs_all=494.11 tgs/tgs_avg=745.45 tgs/tgs_SMA=494.11 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=8.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 76.99316692997016, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=196608 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.43 acc=0.0704 perplexity=25907.498 acc/.git=0.0704 acc/.github=0.0 acc/.gitignore=0.0 acc/.gitmodules=0.0 acc/.owners.yml=0.0 acc/.pre-commit-config.yaml=0.0 acc/.pylintrc=0.0 acc/.pytest_cache=0.0 acc/.readthedocs.yml=0.0 acc/.vscode=0.0 acc/7b_train=0.0 acc/CHANGE_LOG.md=0.0 acc/LICENSE=0.0 acc/README-ja-JP.md=0.0 acc/README-zh-Hans.md=0.0 acc/README.md=0.0 acc/RUN=0.0 acc/ci_scripts=0.0 acc/configs=0.0 acc/doc=0.0 acc/docker=0.0 acc/docker.Makefile=0.0 acc/experiment=0.0 acc/internlm=0.0 acc/requirements=0.0 acc/sonar-project.properties=0.0 acc/tests=0.0 acc/third_party=0.0 acc/tools=0.0 acc/train.py=0.0 acc/version.txt=0.0 acc/web_demo.py=0.0 acc/web_demo_internlm.py=0.0 tokens/.git=60244 tokens/.github=0 tokens/.gitignore=0 tokens/.gitmodules=0 tokens/.owners.yml=0 tokens/.pre-commit-config.yaml=0 tokens/.pylintrc=0 tokens/.pytest_cache=0 tokens/.readthedocs.yml=0 tokens/.vscode=0 tokens/7b_train=0 tokens/CHANGE_LOG.md=0 tokens/LICENSE=0 tokens/README-ja-JP.md=0 tokens/README-zh-Hans.md=0 tokens/README.md=0 tokens/RUN=0 tokens/ci_scripts=0 tokens/configs=0 tokens/doc=0 tokens/docker=0 tokens/docker.Makefile=0 tokens/experiment=0 tokens/internlm=0 tokens/requirements=0 tokens/sonar-project.properties=0 tokens/tests=0 tokens/third_party=0 tokens/tools=0 tokens/train.py=0 tokens/version.txt=0 tokens/web_demo.py=0 tokens/web_demo_internlm.py=0 loss_from_metric=10.1623 loss/.git=10.1623 loss/.github=nan loss/.gitignore=nan loss/.gitmodules=nan loss/.owners.yml=nan loss/.pre-commit-config.yaml=nan loss/.pylintrc=nan loss/.pytest_cache=nan loss/.readthedocs.yml=nan loss/.vscode=nan loss/7b_train=nan loss/CHANGE_LOG.md=nan loss/LICENSE=nan loss/README-ja-JP.md=nan loss/README-zh-Hans.md=nan loss/README.md=nan loss/RUN=nan loss/ci_scripts=nan loss/configs=nan loss/doc=nan loss/docker=nan loss/docker.Makefile=nan loss/experiment=nan loss/internlm=nan loss/requirements=nan loss/sonar-project.properties=nan loss/tests=nan loss/third_party=nan loss/tools=nan loss/train.py=nan loss/version.txt=nan loss/web_demo.py=nan loss/web_demo_internlm.py=nan 
    2023-11-10T11:46:29.389+08:00 INFO [training_internlm.py, line 600, in record_current_batch_training_metrics] - pid=117775 : tflops=127.11695859262743 step=3 loss=8.848428726196289 tgs (tokens/gpu/second)=1025.42 tgs/last_tgs_1=1025.43 tgs/tgs_all=567.64 tgs/tgs_avg=815.45 tgs/tgs_SMA=567.64 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=1.0000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 60.47096249182329, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=262144 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.44 acc=0.0782 perplexity=7380.2217 acc/.git=0.0782 acc/.github=0.0 acc/.gitignore=0.0 acc/.gitmodules=0.0 acc/.owners.yml=0.0 acc/.pre-commit-config.yaml=0.0 acc/.pylintrc=0.0 acc/.pytest_cache=0.0 acc/.readthedocs.yml=0.0 acc/.vscode=0.0 acc/7b_train=0.0 acc/CHANGE_LOG.md=0.0 acc/LICENSE=0.0 acc/README-ja-JP.md=0.0 acc/README-zh-Hans.md=0.0 acc/README.md=0.0 acc/RUN=0.0 acc/ci_scripts=0.0 acc/configs=0.0 acc/doc=0.0 acc/docker=0.0 acc/docker.Makefile=0.0 acc/experiment=0.0 acc/internlm=0.0 acc/requirements=0.0 acc/sonar-project.properties=0.0 acc/tests=0.0 acc/third_party=0.0 acc/tools=0.0 acc/train.py=0.0 acc/version.txt=0.0 acc/web_demo.py=0.0 acc/web_demo_internlm.py=0.0 tokens/.git=60328 tokens/.github=0 tokens/.gitignore=0 tokens/.gitmodules=0 tokens/.owners.yml=0 tokens/.pre-commit-config.yaml=0 tokens/.pylintrc=0 tokens/.pytest_cache=0 tokens/.readthedocs.yml=0 tokens/.vscode=0 tokens/7b_train=0 tokens/CHANGE_LOG.md=0 tokens/LICENSE=0 tokens/README-ja-JP.md=0 tokens/README-zh-Hans.md=0 tokens/README.md=0 tokens/RUN=0 tokens/ci_scripts=0 tokens/configs=0 tokens/doc=0 tokens/docker=0 tokens/docker.Makefile=0 tokens/experiment=0 tokens/internlm=0 tokens/requirements=0 tokens/sonar-project.properties=0 tokens/tests=0 tokens/third_party=0 tokens/tools=0 tokens/train.py=0 tokens/version.txt=0 tokens/web_demo.py=0 tokens/web_demo_internlm.py=0 loss_from_metric=8.9066 loss/.git=8.9066 loss/.github=nan loss/.gitignore=nan loss/.gitmodules=nan loss/.owners.yml=nan loss/.pre-commit-config.yaml=nan loss/.pylintrc=nan loss/.pytest_cache=nan loss/.readthedocs.yml=nan loss/.vscode=nan loss/7b_train=nan loss/CHANGE_LOG.md=nan loss/LICENSE=nan loss/README-ja-JP.md=nan loss/README-zh-Hans.md=nan loss/README.md=nan loss/RUN=nan loss/ci_scripts=nan loss/configs=nan loss/doc=nan loss/docker=nan loss/docker.Makefile=nan loss/experiment=nan loss/internlm=nan loss/requirements=nan loss/sonar-project.properties=nan loss/tests=nan loss/third_party=nan loss/tools=nan loss/train.py=nan loss/version.txt=nan loss/web_demo.py=nan loss/web_demo_internlm.py=nan 
    2023-11-10T11:46:33.512+08:00 INFO [training_internlm.py, line 600, in record_current_batch_training_metrics] - pid=117775 : tflops=127.046731454726 step=4 loss=7.509818077087402 tgs (tokens/gpu/second)=1024.85 tgs/last_tgs_1=1024.86 tgs/tgs_all=623.25 tgs/tgs_avg=857.33 tgs/tgs_SMA=623.25 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=1.2000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 42.36598096083032, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=327680 inf_nan_skip_batches=0 num_samples_in_batch=22 largest_length=1893 largest_batch=8 smallest_batch=4 adam_beta2=0.95 fwd_bwd_time=3.44 acc=0.0706 perplexity=2728.5999 acc/.git=0.0706 acc/.github=0.0 acc/.gitignore=0.0 acc/.gitmodules=0.0 acc/.owners.yml=0.0 acc/.pre-commit-config.yaml=0.0 acc/.pylintrc=0.0 acc/.pytest_cache=0.0 acc/.readthedocs.yml=0.0 acc/.vscode=0.0 acc/7b_train=0.0 acc/CHANGE_LOG.md=0.0 acc/LICENSE=0.0 acc/README-ja-JP.md=0.0 acc/README-zh-Hans.md=0.0 acc/README.md=0.0 acc/RUN=0.0 acc/ci_scripts=0.0 acc/configs=0.0 acc/doc=0.0 acc/docker=0.0 acc/docker.Makefile=0.0 acc/experiment=0.0 acc/internlm=0.0 acc/requirements=0.0 acc/sonar-project.properties=0.0 acc/tests=0.0 acc/third_party=0.0 acc/tools=0.0 acc/train.py=0.0 acc/version.txt=0.0 acc/web_demo.py=0.0 acc/web_demo_internlm.py=0.0 tokens/.git=61028 tokens/.github=0 tokens/.gitignore=0 tokens/.gitmodules=0 tokens/.owners.yml=0 tokens/.pre-commit-config.yaml=0 tokens/.pylintrc=0 tokens/.pytest_cache=0 tokens/.readthedocs.yml=0 tokens/.vscode=0 tokens/7b_train=0 tokens/CHANGE_LOG.md=0 tokens/LICENSE=0 tokens/README-ja-JP.md=0 tokens/README-zh-Hans.md=0 tokens/README.md=0 tokens/RUN=0 tokens/ci_scripts=0 tokens/configs=0 tokens/doc=0 tokens/docker=0 tokens/docker.Makefile=0 tokens/experiment=0 tokens/internlm=0 tokens/requirements=0 tokens/sonar-project.properties=0 tokens/tests=0 tokens/third_party=0 tokens/tools=0 tokens/train.py=0 tokens/version.txt=0 tokens/web_demo.py=0 tokens/web_demo_internlm.py=0 loss_from_metric=7.9115 loss/.git=7.9115 loss/.github=nan loss/.gitignore=nan loss/.gitmodules=nan loss/.owners.yml=nan loss/.pre-commit-config.yaml=nan loss/.pylintrc=nan loss/.pytest_cache=nan loss/.readthedocs.yml=nan loss/.vscode=nan loss/7b_train=nan loss/CHANGE_LOG.md=nan loss/LICENSE=nan loss/README-ja-JP.md=nan loss/README-zh-Hans.md=nan loss/README.md=nan loss/RUN=nan loss/ci_scripts=nan loss/configs=nan loss/doc=nan loss/docker=nan loss/docker.Makefile=nan loss/experiment=nan loss/internlm=nan loss/requirements=nan loss/sonar-project.properties=nan loss/tests=nan loss/third_party=nan loss/tools=nan loss/train.py=nan loss/version.txt=nan loss/web_demo.py=nan loss/web_demo_internlm.py=nan 
    2023-11-10T11:46:37.686+08:00 INFO [training_internlm.py, line 600, in record_current_batch_training_metrics] - pid=117775 : tflops=125.95244539756375 step=5 loss=7.049615859985352 tgs (tokens/gpu/second)=1016.03 tgs/last_tgs_1=1016.04 tgs/tgs_all=666.17 tgs/tgs_avg=883.78 tgs/tgs_SMA=666.17 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=1.4000000000000001e-06 loss_scale=65536.0 grad_norm={'0_default': 32.49300931426443, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=393216 inf_nan_skip_batches=0 num_samples_in_batch=13 largest_length=2048 largest_batch=4 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.48 acc=0.0726 perplexity=1169.7832 acc/.git=0.0726 acc/.github=0.0 acc/.gitignore=0.0 acc/.gitmodules=0.0 acc/.owners.yml=0.0 acc/.pre-commit-config.yaml=0.0 acc/.pylintrc=0.0 acc/.pytest_cache=0.0 acc/.readthedocs.yml=0.0 acc/.vscode=0.0 acc/7b_train=0.0 acc/CHANGE_LOG.md=0.0 acc/LICENSE=0.0 acc/README-ja-JP.md=0.0 acc/README-zh-Hans.md=0.0 acc/README.md=0.0 acc/RUN=0.0 acc/ci_scripts=0.0 acc/configs=0.0 acc/doc=0.0 acc/docker=0.0 acc/docker.Makefile=0.0 acc/experiment=0.0 acc/internlm=0.0 acc/requirements=0.0 acc/sonar-project.properties=0.0 acc/tests=0.0 acc/third_party=0.0 acc/tools=0.0 acc/train.py=0.0 acc/version.txt=0.0 acc/web_demo.py=0.0 acc/web_demo_internlm.py=0.0 tokens/.git=61004 tokens/.github=0 tokens/.gitignore=0 tokens/.gitmodules=0 tokens/.owners.yml=0 tokens/.pre-commit-config.yaml=0 tokens/.pylintrc=0 tokens/.pytest_cache=0 tokens/.readthedocs.yml=0 tokens/.vscode=0 tokens/7b_train=0 tokens/CHANGE_LOG.md=0 tokens/LICENSE=0 tokens/README-ja-JP.md=0 tokens/README-zh-Hans.md=0 tokens/README.md=0 tokens/RUN=0 tokens/ci_scripts=0 tokens/configs=0 tokens/doc=0 tokens/docker=0 tokens/docker.Makefile=0 tokens/experiment=0 tokens/internlm=0 tokens/requirements=0 tokens/sonar-project.properties=0 tokens/tests=0 tokens/third_party=0 tokens/tools=0 tokens/train.py=0 tokens/version.txt=0 tokens/web_demo.py=0 tokens/web_demo_internlm.py=0 loss_from_metric=7.0646 loss/.git=7.0646 loss/.github=nan loss/.gitignore=nan loss/.gitmodules=nan loss/.owners.yml=nan loss/.pre-commit-config.yaml=nan loss/.pylintrc=nan loss/.pytest_cache=nan loss/.readthedocs.yml=nan loss/.vscode=nan loss/7b_train=nan loss/CHANGE_LOG.md=nan loss/LICENSE=nan loss/README-ja-JP.md=nan loss/README-zh-Hans.md=nan loss/README.md=nan loss/RUN=nan loss/ci_scripts=nan loss/configs=nan loss/doc=nan loss/docker=nan loss/docker.Makefile=nan loss/experiment=nan loss/internlm=nan loss/requirements=nan loss/sonar-project.properties=nan loss/tests=nan loss/third_party=nan loss/tools=nan loss/train.py=nan loss/version.txt=nan loss/web_demo.py=nan loss/web_demo_internlm.py=nan 
    
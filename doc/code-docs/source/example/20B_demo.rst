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

    2023-11-10 15:05:04,535 INFO parallel_context.py:555 in set_device -- process rank 9 is bound to host:HOST-10-140-60-90 device: 1
    2023-11-10 15:05:04,518 INFO parallel_context.py:555 in set_device -- process rank 6 is bound to host:HOST-10-140-60-14 device: 6
    2023-11-10 15:05:04,523 INFO parallel_context.py:555 in set_device -- process rank 0 is bound to host:HOST-10-140-60-14 device: 0
    2023-11-10 15:05:04,524 INFO parallel_context.py:555 in set_device -- process rank 3 is bound to host:HOST-10-140-60-14 device: 3
    2023-11-10 15:05:04,575 INFO parallel_context.py:555 in set_device -- process rank 15 is bound to host:HOST-10-140-60-90 device: 7
    2023-11-10 15:05:04,576 INFO parallel_context.py:555 in set_device -- process rank 12 is bound to host:HOST-10-140-60-90 device: 4
    2023-11-10 15:05:04,577 INFO parallel_context.py:555 in set_device -- process rank 11 is bound to host:HOST-10-140-60-90 device: 3
    2023-11-10 15:05:04,582 INFO parallel_context.py:555 in set_device -- process rank 10 is bound to host:HOST-10-140-60-90 device: 2
    2023-11-10 15:05:04,560 INFO parallel_context.py:555 in set_device -- process rank 5 is bound to host:HOST-10-140-60-14 device: 5
    2023-11-10 15:05:04,592 INFO parallel_context.py:555 in set_device -- process rank 4 is bound to host:HOST-10-140-60-14 device: 4
    2023-11-10 15:05:04,593 INFO parallel_context.py:555 in set_device -- process rank 7 is bound to host:HOST-10-140-60-14 device: 7
    2023-11-10 15:05:04,624 INFO parallel_context.py:555 in set_device -- process rank 1 is bound to host:HOST-10-140-60-14 device: 1
    2023-11-10 15:05:04,683 INFO parallel_context.py:555 in set_device -- process rank 8 is bound to host:HOST-10-140-60-90 device: 0
    2023-11-10 15:05:04,718 INFO parallel_context.py:555 in set_device -- process rank 14 is bound to host:HOST-10-140-60-90 device: 6
    2023-11-10 15:05:04,718 INFO parallel_context.py:555 in set_device -- process rank 13 is bound to host:HOST-10-140-60-90 device: 5
    2023-11-10 15:05:04,723 INFO parallel_context.py:555 in set_device -- process rank 2 is bound to host:HOST-10-140-60-14 device: 2
    2023-11-10 15:05:07,912 INFO launch.py:409 in launch -- Distributed environment is initialized, data parallel size: 4, pipeline parallel size: 1, tensor parallel size: 4
    2023-11-10 15:05:24,106 INFO hybrid_zero_optim.py:268 in _partition_param_list -- Number of elements on ranks: [1262168320, 1269084160, 1269084160, 1222844160], rank:0
    2023-11-10T15:05:58.540+08:00 INFO [training_internlm.py, line 601, in record_current_batch_training_metrics] - pid=78690 : tflops=41.977599404049684 step=0 loss=11.542577743530273 tgs (tokens/gpu/second)=338.62 tgs/last_tgs_1=338.62 tgs/tgs_all=338.62 tgs/tgs_avg=338.62 tgs/tgs_SMA=338.62 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=4.0000000000000003e-07 loss_scale=65536.0 grad_norm={'0_default': 87.3189617106087, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=65536 inf_nan_skip_batches=0 num_samples_in_batch=18 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=8.94 acc=0.0 perplexity=104321.0312 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=60571 tokens/cn=0 tokens/code=0 loss_from_metric=11.5552 loss/en=11.5552 loss/cn=nan loss/code=nan 
    2023-11-10T15:06:02.978+08:00 INFO [training_internlm.py, line 601, in record_current_batch_training_metrics] - pid=78690 : tflops=115.41412094522278 step=1 loss=11.33798599243164 tgs (tokens/gpu/second)=931.02 tgs/last_tgs_1=931.03 tgs/tgs_all=496.62 tgs/tgs_avg=634.83 tgs/tgs_SMA=496.62 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=6.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 90.85008685328815, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=131072 inf_nan_skip_batches=0 num_samples_in_batch=19 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.85 acc=0.0 perplexity=81555.5 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=60265 tokens/cn=0 tokens/code=0 loss_from_metric=11.309 loss/en=11.309 loss/cn=nan loss/code=nan 
    2023-11-10T15:06:06.988+08:00 INFO [training_internlm.py, line 601, in record_current_batch_training_metrics] - pid=78690 : tflops=127.89743136367036 step=2 loss=10.111495971679688 tgs (tokens/gpu/second)=1031.72 tgs/last_tgs_1=1031.72 tgs/tgs_all=600.43 tgs/tgs_avg=767.12 tgs/tgs_SMA=600.43 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=8.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 76.99318912653898, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=196608 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.4 acc=0.0704 perplexity=25907.623 acc/en=0.0704 acc/cn=0.0 acc/code=0.0 tokens/en=60244 tokens/cn=0 tokens/code=0 loss_from_metric=10.1623 loss/en=10.1623 loss/cn=nan loss/code=nan 
    2023-11-10T15:06:10.994+08:00 INFO [training_internlm.py, line 601, in record_current_batch_training_metrics] - pid=78690 : tflops=127.89845291183941 step=3 loss=8.848427772521973 tgs (tokens/gpu/second)=1031.73 tgs/last_tgs_1=1031.73 tgs/tgs_all=670.5 tgs/tgs_avg=833.27 tgs/tgs_SMA=670.5 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=1.0000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 60.47092413727133, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=262144 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.41 acc=0.0783 perplexity=7380.229 acc/en=0.0783 acc/cn=0.0 acc/code=0.0 tokens/en=60328 tokens/cn=0 tokens/code=0 loss_from_metric=8.9066 loss/en=8.9066 loss/cn=nan loss/code=nan 
    2023-11-10T15:06:15.041+08:00 INFO [training_internlm.py, line 601, in record_current_batch_training_metrics] - pid=78690 : tflops=126.55593705224216 step=4 loss=7.509810924530029 tgs (tokens/gpu/second)=1020.9 tgs/last_tgs_1=1020.9 tgs/tgs_all=719.92 tgs/tgs_avg=870.8 tgs/tgs_SMA=719.92 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=1.2000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 42.36608180721121, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=327680 inf_nan_skip_batches=0 num_samples_in_batch=22 largest_length=1893 largest_batch=8 smallest_batch=4 adam_beta2=0.95 fwd_bwd_time=3.43 acc=0.0706 perplexity=2728.5764 acc/en=0.0706 acc/cn=0.0 acc/code=0.0 tokens/en=61028 tokens/cn=0 tokens/code=0 loss_from_metric=7.9115 loss/en=7.9115 loss/cn=nan loss/code=nan 
    2023-11-10T15:06:19.051+08:00 INFO [training_internlm.py, line 601, in record_current_batch_training_metrics] - pid=78690 : tflops=127.79902453659938 step=5 loss=7.049621105194092 tgs (tokens/gpu/second)=1030.92 tgs/last_tgs_1=1030.93 tgs/tgs_all=758.03 tgs/tgs_avg=897.49 tgs/tgs_SMA=758.03 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=1.4000000000000001e-06 loss_scale=65536.0 grad_norm={'0_default': 32.49298677335042, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=393216 inf_nan_skip_batches=0 num_samples_in_batch=13 largest_length=2048 largest_batch=4 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.42 acc=0.0726 perplexity=1169.7916 acc/en=0.0726 acc/cn=0.0 acc/code=0.0 tokens/en=61004 tokens/cn=0 tokens/code=0 loss_from_metric=7.0646 loss/en=7.0646 loss/cn=nan loss/code=nan
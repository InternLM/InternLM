7B Demo
================

训练配置
----------------

7B demo 的训练配置文件样例如下:

.. code-block:: python

    JOB_NAME = "7b_train"

    SEQ_LEN = 2048
    HIDDEN_SIZE = 4096
    NUM_ATTENTION_HEAD = 32
    MLP_RATIO = 8 / 3
    NUM_LAYER = 32
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
        zero1=8,
        pipeline=dict(size=1, interleaved_overlap=True),
        sequence_parallel=False,
    )

    cudnn_deterministic = False
    cudnn_benchmark = False

启动训练
----------------

完成以上训练配置后，可启动模型训练，以在 ``slurm`` 平台上为例，启动单节点 8GPU 的训练命令如下所示：

.. code-block:: bash

    srun -p internllm -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py

训练结果
----------------

基于以上训练配置和启动命令，单节点 8GPU 下的模型训练部分日志展示如下：

.. code-block:: bash

    2023-09-05 11:47:44,649 INFO parallel_context.py:508 in set_device -- process rank 4 is bound to host:SH-IDC1-10-140-1-110 device: 4
    2023-09-05 11:47:44,650 INFO parallel_context.py:508 in set_device -- process rank 3 is bound to host:SH-IDC1-10-140-1-110 device: 3
    2023-09-05 11:47:44,651 INFO parallel_context.py:508 in set_device -- process rank 6 is bound to host:SH-IDC1-10-140-1-110 device: 6
    2023-09-05 11:47:44,652 INFO parallel_context.py:508 in set_device -- process rank 7 is bound to host:SH-IDC1-10-140-1-110 device: 7
    2023-09-05 11:47:44,652 INFO parallel_context.py:508 in set_device -- process rank 5 is bound to host:SH-IDC1-10-140-1-110 device: 5
    2023-09-05 11:47:44,652 INFO parallel_context.py:508 in set_device -- process rank 1 is bound to host:SH-IDC1-10-140-1-110 device: 1
    2023-09-05 11:47:44,652 INFO parallel_context.py:508 in set_device -- process rank 2 is bound to host:SH-IDC1-10-140-1-110 device: 2
    2023-09-05 11:47:44,652 INFO parallel_context.py:508 in set_device -- process rank 0 is bound to host:SH-IDC1-10-140-1-110 device: 0
    2023-09-05 11:47:51,006 INFO launch.py:354 in launch -- Distributed environment is initialized, data parallel size: 8, pipeline parallel size: 1, tensor parallel size: 1
    2023-09-05 11:49:09,855 INFO hybrid_zero_optim.py:294 in _partition_param_list -- Number of elements on ranks: [894509056, 944865280, 966909952, 966909952, 966909952, 944865280, 966909952, 670068736], rank:0
    2023-09-05T11:49:58.225+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=6794 : tflops=63.283263603947816 step=0 loss=11.641494750976562 tgs (tokens/gpu/second)=1424.93 lr=4.0000000000000003e-07 loss_scale=65536.0 grad_norm={'0_default': 66.51907327507652} micro_num=4 num_consumed_tokens=131072 inf_nan_skip_batches=0 num_samples_in_batch=19 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=6.87 acc=0.0 perplexity=112181.7188 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=120836 tokens/cn=0 tokens/code=0 loss_from_metric=11.6279 loss/en=11.6279 loss/cn=nan loss/code=nan 
    2023-09-05T11:50:02.553+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=6794 : tflops=171.92140761933035 step=1 loss=11.546792984008789 tgs (tokens/gpu/second)=3871.11 lr=6.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 64.47430144542088} micro_num=4 num_consumed_tokens=262144 inf_nan_skip_batches=0 num_samples_in_batch=16 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=4.14 acc=0.0 perplexity=103779.1406 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=120572 tokens/cn=0 tokens/code=0 loss_from_metric=11.55 loss/en=11.55 loss/cn=nan loss/code=nan 
    2023-09-05T11:50:06.504+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=6794 : tflops=186.0565203348341 step=2 loss=11.106071472167969 tgs (tokens/gpu/second)=4189.39 lr=8.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 62.520055376005146} micro_num=4 num_consumed_tokens=393216 inf_nan_skip_batches=0 num_samples_in_batch=16 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.82 acc=0.0001 perplexity=71139.6797 acc/en=0.0001 acc/cn=0.0 acc/code=0.0 tokens/en=122032 tokens/cn=0 tokens/code=0 loss_from_metric=11.1724 loss/en=11.1724 loss/cn=nan loss/code=nan 
    2023-09-05T11:50:10.487+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=6794 : tflops=185.48897918112567 step=3 loss=10.444510459899902 tgs (tokens/gpu/second)=4176.61 lr=1.0000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 57.91057980979166} micro_num=4 num_consumed_tokens=524288 inf_nan_skip_batches=0 num_samples_in_batch=18 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.83 acc=0.0705 perplexity=39851.1289 acc/en=0.0705 acc/cn=0.0 acc/code=0.0 tokens/en=121125 tokens/cn=0 tokens/code=0 loss_from_metric=10.5929 loss/en=10.5929 loss/cn=nan loss/code=nan 
    2023-09-05T11:50:14.476+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=6794 : tflops=185.8751803758398 step=4 loss=9.798665046691895 tgs (tokens/gpu/second)=4185.31 lr=1.2000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 48.1136933755285} micro_num=4 num_consumed_tokens=655360 inf_nan_skip_batches=0 num_samples_in_batch=14 largest_length=2048 largest_batch=4 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.82 acc=0.076 perplexity=18045.6699 acc/en=0.076 acc/cn=0.0 acc/code=0.0 tokens/en=121365 tokens/cn=0 tokens/code=0 loss_from_metric=9.8007 loss/en=9.8007 loss/cn=nan loss/code=nan 
    2023-09-05T11:50:18.442+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=6794 : tflops=185.6236609556878 step=5 loss=9.215429306030273 tgs (tokens/gpu/second)=4179.64 lr=1.4000000000000001e-06 loss_scale=65536.0 grad_norm={'0_default': 36.95489557069029} micro_num=4 num_consumed_tokens=786432 inf_nan_skip_batches=0 num_samples_in_batch=14 largest_length=2048 largest_batch=4 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=3.82 acc=0.0767 perplexity=8999.0869 acc/en=0.0767 acc/cn=0.0 acc/code=0.0 tokens/en=121223 tokens/cn=0 tokens/code=0 loss_from_metric=9.1049 loss/en=9.1049 loss/cn=nan loss/code=nan 

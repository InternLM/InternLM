30B Demo
================

Training Config
----------------

30B demo config file example:

.. code-block:: python

    JOB_NAME = "30b_train"

    SEQ_LEN = 2048
    HIDDEN_SIZE = 6144
    NUM_ATTENTION_HEAD = 48
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
        checkpoint=True,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
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
        tensor=8,
        pipeline=dict(size=2, interleaved_overlap=True),
        sequence_parallel=True,
    )

    cudnn_deterministic = False
    cudnn_benchmark = False

Start Training
----------------

After completing the data preparation and relevant training configurations, you can start the demo training.
The following example shows how to start distributed training in ``slurm`` environments with 16 GPUs.

.. code-block:: bash

    srun -p internllm -N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/30B_sft.py

Training Results
----------------

Taking the configuration of the demo training on two nodes with 16 GPUs on slurm as an example, the training result log is shown below:

.. code-block:: bash

    2023-09-05 12:50:09,682 INFO parallel_context.py:508 in set_device -- process rank 3 is bound to host:SH-IDC1-10-140-1-110 device: 3
    2023-09-05 12:50:09,685 INFO parallel_context.py:508 in set_device -- process rank 7 is bound to host:SH-IDC1-10-140-1-110 device: 7
    2023-09-05 12:50:09,686 INFO parallel_context.py:508 in set_device -- process rank 6 is bound to host:SH-IDC1-10-140-1-110 device: 6
    2023-09-05 12:50:09,688 INFO parallel_context.py:508 in set_device -- process rank 5 is bound to host:SH-IDC1-10-140-1-110 device: 5
    2023-09-05 12:50:09,689 INFO parallel_context.py:508 in set_device -- process rank 1 is bound to host:SH-IDC1-10-140-1-110 device: 1
    2023-09-05 12:50:09,689 INFO parallel_context.py:508 in set_device -- process rank 4 is bound to host:SH-IDC1-10-140-1-110 device: 4
    2023-09-05 12:50:09,689 INFO parallel_context.py:508 in set_device -- process rank 0 is bound to host:SH-IDC1-10-140-1-110 device: 0
    2023-09-05 12:50:09,689 INFO parallel_context.py:508 in set_device -- process rank 2 is bound to host:SH-IDC1-10-140-1-110 device: 2
    2023-09-05 12:50:09,696 INFO parallel_context.py:508 in set_device -- process rank 13 is bound to host:SH-IDC1-10-140-1-138 device: 5
    2023-09-05 12:50:09,699 INFO parallel_context.py:508 in set_device -- process rank 9 is bound to host:SH-IDC1-10-140-1-138 device: 1
    2023-09-05 12:50:09,700 INFO parallel_context.py:508 in set_device -- process rank 14 is bound to host:SH-IDC1-10-140-1-138 device: 6
    2023-09-05 12:50:09,701 INFO parallel_context.py:508 in set_device -- process rank 15 is bound to host:SH-IDC1-10-140-1-138 device: 7
    2023-09-05 12:50:09,702 INFO parallel_context.py:508 in set_device -- process rank 12 is bound to host:SH-IDC1-10-140-1-138 device: 4
    2023-09-05 12:50:09,703 INFO parallel_context.py:508 in set_device -- process rank 8 is bound to host:SH-IDC1-10-140-1-138 device: 0
    2023-09-05 12:50:09,704 INFO parallel_context.py:508 in set_device -- process rank 10 is bound to host:SH-IDC1-10-140-1-138 device: 2
    2023-09-05 12:50:09,704 INFO parallel_context.py:508 in set_device -- process rank 11 is bound to host:SH-IDC1-10-140-1-138 device: 3
    2023-09-05 12:50:16,744 INFO launch.py:354 in launch -- Distributed environment is initialized, data parallel size: 1, pipeline parallel size: 2, tensor parallel size: 8
    2023-09-05 12:51:35,106 INFO hybrid_zero_optim.py:294 in _partition_param_list -- Number of elements on ranks: [1778554368], rank:8
    2023-09-05T12:52:09.502+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=187985 : tflops=20.13930514195207 step=0 loss=11.515681266784668 tgs (tokens/gpu/second)=86.32 lr=4.0000000000000003e-07 loss_scale=65536.0 grad_norm={'0_default': 44.66850557859531} micro_num=4 num_consumed_tokens=16384 inf_nan_skip_batches=0 num_samples_in_batch=15 largest_length=2048 largest_batch=4 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=7.77 acc=0.0 perplexity=100246.0391 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=15092 tokens/cn=0 tokens/code=0 loss_from_metric=11.5154 loss/en=11.5154 loss/cn=nan loss/code=nan 
    2023-09-05T12:52:12.267+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=187985 : tflops=87.4814514237506 step=1 loss=11.463700294494629 tgs (tokens/gpu/second)=374.95 lr=6.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 37.74545806441071} micro_num=4 num_consumed_tokens=32768 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=2.2 acc=0.0 perplexity=95221.6484 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=15317 tokens/cn=0 tokens/code=0 loss_from_metric=11.464 loss/en=11.464 loss/cn=nan loss/code=nan 
    2023-09-05T12:52:15.044+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=187985 : tflops=89.13317930104051 step=2 loss=10.821619033813477 tgs (tokens/gpu/second)=382.03 lr=8.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 43.673985302322706} micro_num=4 num_consumed_tokens=49152 inf_nan_skip_batches=0 num_samples_in_batch=22 largest_length=2048 largest_batch=7 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=2.15 acc=0.0446 perplexity=50201.3984 acc/en=0.0446 acc/cn=0.0 acc/code=0.0 tokens/en=14783 tokens/cn=0 tokens/code=0 loss_from_metric=10.8238 loss/en=10.8238 loss/cn=nan loss/code=nan 
    2023-09-05T12:52:17.716+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=187985 : tflops=91.82722820664878 step=3 loss=10.437983512878418 tgs (tokens/gpu/second)=393.58 lr=1.0000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 35.13228697210907} micro_num=4 num_consumed_tokens=65536 inf_nan_skip_batches=0 num_samples_in_batch=15 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=2.08 acc=0.0608 perplexity=34208.2695 acc/en=0.0608 acc/cn=0.0 acc/code=0.0 tokens/en=15379 tokens/cn=0 tokens/code=0 loss_from_metric=10.4402 loss/en=10.4403 loss/cn=nan loss/code=nan 
    2023-09-05T12:52:20.400+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=187985 : tflops=91.19641511880559 step=4 loss=9.160537719726562 tgs (tokens/gpu/second)=390.87 lr=1.2000000000000002e-06 loss_scale=65536.0 grad_norm={'0_default': 36.88318235490262} micro_num=4 num_consumed_tokens=81920 inf_nan_skip_batches=0 num_samples_in_batch=16 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=2.1 acc=0.0889 perplexity=9540.1846 acc/en=0.0889 acc/cn=0.0 acc/code=0.0 tokens/en=14903 tokens/cn=0 tokens/code=0 loss_from_metric=9.1633 loss/en=9.1633 loss/cn=nan loss/code=nan 
    2023-09-05T12:52:23.088+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=187985 : tflops=90.62865083014503 step=5 loss=8.400506019592285 tgs (tokens/gpu/second)=388.44 lr=1.4000000000000001e-06 loss_scale=65536.0 grad_norm={'0_default': 30.51502295553065} micro_num=4 num_consumed_tokens=98304 inf_nan_skip_batches=0 num_samples_in_batch=20 largest_length=2048 largest_batch=6 smallest_batch=4 adam_beta2=0.95 fwd_bwd_time=2.12 acc=0.0884 perplexity=4532.5967 acc/en=0.0884 acc/cn=0.0 acc/code=0.0 tokens/en=14972 tokens/cn=0 tokens/code=0 loss_from_metric=8.4191 loss/en=8.419 loss/cn=nan loss/code=nan

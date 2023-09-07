30B Demo
================

训练配置
----------------

30B demo 训练配置文件样例如下:

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

    srun -p internllm -N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/30B_sft.py

训练结果
----------------

基于以上训练配置和启动命令，两节点 16GPU 下的模型训练部分日志展示如下：

.. code-block:: bash

    2023-09-06 10:29:26,629 INFO parallel_context.py:508 in set_device -- process rank 10 is bound to host:HOST-10-140-66-20 device: 2
    2023-09-06 10:29:26,632 INFO parallel_context.py:508 in set_device -- process rank 11 is bound to host:HOST-10-140-66-20 device: 3
    2023-09-06 10:29:26,634 INFO parallel_context.py:508 in set_device -- process rank 12 is bound to host:HOST-10-140-66-20 device: 4
    2023-09-06 10:29:26,636 INFO parallel_context.py:508 in set_device -- process rank 9 is bound to host:HOST-10-140-66-20 device: 1
    2023-09-06 10:29:26,640 INFO parallel_context.py:508 in set_device -- process rank 15 is bound to host:HOST-10-140-66-20 device: 7
    2023-09-06 10:29:26,639 INFO parallel_context.py:508 in set_device -- process rank 0 is bound to host:HOST-10-140-66-9 device: 0
    2023-09-06 10:29:26,641 INFO parallel_context.py:508 in set_device -- process rank 2 is bound to host:HOST-10-140-66-9 device: 2
    2023-09-06 10:29:26,643 INFO parallel_context.py:508 in set_device -- process rank 5 is bound to host:HOST-10-140-66-9 device: 5
    2023-09-06 10:29:26,645 INFO parallel_context.py:508 in set_device -- process rank 6 is bound to host:HOST-10-140-66-9 device: 6
    2023-09-06 10:29:26,661 INFO parallel_context.py:508 in set_device -- process rank 13 is bound to host:HOST-10-140-66-20 device: 5
    2023-09-06 10:29:26,707 INFO parallel_context.py:508 in set_device -- process rank 1 is bound to host:HOST-10-140-66-9 device: 1
    2023-09-06 10:29:26,826 INFO parallel_context.py:508 in set_device -- process rank 4 is bound to host:HOST-10-140-66-9 device: 4
    2023-09-06 10:29:26,871 INFO parallel_context.py:508 in set_device -- process rank 7 is bound to host:HOST-10-140-66-9 device: 7
    2023-09-06 10:29:26,932 INFO parallel_context.py:508 in set_device -- process rank 3 is bound to host:HOST-10-140-66-9 device: 3
    2023-09-06 10:29:27,156 INFO parallel_context.py:508 in set_device -- process rank 14 is bound to host:HOST-10-140-66-20 device: 6
    2023-09-06 10:29:27,271 INFO parallel_context.py:508 in set_device -- process rank 8 is bound to host:HOST-10-140-66-20 device: 0
    2023-09-06 10:29:32,060 INFO launch.py:329 in launch -- Distributed environment is initialized, data parallel size: 4, pipeline parallel size: 1, tensor parallel size: 4
    2023-09-06 10:30:06,141 INFO hybrid_zero_optim.py:291 in _partition_param_list -- Number of elements on ranks: [1782007296, 1812307968, 1812307968, 1706469888], rank:0
    2023-09-06T10:30:38.216+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=15224 : tflops=40.00268401421643 step=0 loss=11.548227310180664 tgs (tokens/gpu/second)=227.37 lr=9.779754323328192e-05 loss_scale=65536.0 grad_norm={'0_default': 61.5836932112004} micro_num=4 num_consumed_tokens=65536 inf_nan_skip_batches=0 num_samples_in_batch=18 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=12.51 acc=0.0 perplexity=104121.5547 acc/en=0.0 acc/cn=0.0 acc/code=0.0 tokens/en=60571 tokens/cn=0 tokens/code=0 loss_from_metric=11.5533 loss/en=11.5533 loss/cn=nan loss/code=nan 
    2023-09-06T10:30:46.343+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=15224 : tflops=89.00005814543725 step=1 loss=6.05580997467041 tgs (tokens/gpu/second)=505.86 lr=9.140576474687264e-05 loss_scale=65536.0 grad_norm={'0_default': 27.397946290506887} micro_num=4 num_consumed_tokens=131072 inf_nan_skip_batches=0 num_samples_in_batch=19 largest_length=2048 largest_batch=6 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=7.91 acc=0.0885 perplexity=405.4076 acc/en=0.0885 acc/cn=0.0 acc/code=0.0 tokens/en=60265 tokens/cn=0 tokens/code=0 loss_from_metric=6.0049 loss/en=6.0049 loss/cn=nan loss/code=nan 
    2023-09-06T10:30:51.443+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=15224 : tflops=142.5138940898651 step=2 loss=5.054169654846191 tgs (tokens/gpu/second)=810.03 lr=8.14503363531613e-05 loss_scale=65536.0 grad_norm={'0_default': 10.438111430093606} micro_num=4 num_consumed_tokens=196608 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=4.87 acc=0.0715 perplexity=184.2986 acc/en=0.0715 acc/cn=0.0 acc/code=0.0 tokens/en=60244 tokens/cn=0 tokens/code=0 loss_from_metric=5.2166 loss/en=5.2166 loss/cn=nan loss/code=nan 
    2023-09-06T10:30:56.509+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=15224 : tflops=143.56131674769466 step=3 loss=4.662276268005371 tgs (tokens/gpu/second)=815.98 lr=6.890576474687264e-05 loss_scale=65536.0 grad_norm={'0_default': 9.15959986316653} micro_num=4 num_consumed_tokens=262144 inf_nan_skip_batches=0 num_samples_in_batch=17 largest_length=2048 largest_batch=5 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=4.83 acc=0.0775 perplexity=102.6568 acc/en=0.0775 acc/cn=0.0 acc/code=0.0 tokens/en=60328 tokens/cn=0 tokens/code=0 loss_from_metric=4.6314 loss/en=4.6314 loss/cn=nan loss/code=nan 
    2023-09-06T10:31:01.552+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=15224 : tflops=143.85087291011183 step=4 loss=4.020431041717529 tgs (tokens/gpu/second)=817.63 lr=5.500000000000001e-05 loss_scale=65536.0 grad_norm={'0_default': 6.873464794412589} micro_num=4 num_consumed_tokens=327680 inf_nan_skip_batches=0 num_samples_in_batch=22 largest_length=1893 largest_batch=8 smallest_batch=4 adam_beta2=0.95 fwd_bwd_time=4.82 acc=0.0701 perplexity=69.1167 acc/en=0.0701 acc/cn=0.0 acc/code=0.0 tokens/en=61028 tokens/cn=0 tokens/code=0 loss_from_metric=4.2358 loss/en=4.2358 loss/cn=nan loss/code=nan 
    2023-09-06T10:31:06.830+08:00 INFO [training_internlm.py, line 413, in record_current_batch_training_metrics] - pid=15224 : tflops=142.8966468353613 step=5 loss=3.733311891555786 tgs (tokens/gpu/second)=812.2 lr=4.109423525312737e-05 loss_scale=65536.0 grad_norm={'0_default': 5.811005102730085} micro_num=4 num_consumed_tokens=393216 inf_nan_skip_batches=0 num_samples_in_batch=13 largest_length=2048 largest_batch=4 smallest_batch=3 adam_beta2=0.95 fwd_bwd_time=4.85 acc=0.0688 perplexity=46.298 acc/en=0.0688 acc/cn=0.0 acc/code=0.0 tokens/en=61004 tokens/cn=0 tokens/code=0 loss_from_metric=3.8351 loss/en=3.8351 loss/cn=nan loss/code=nan
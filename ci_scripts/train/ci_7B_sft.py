JOB_NAME = "7b_train"

SEQ_LEN = 1024
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 16
MLP_RATIO = 8 / 3
NUM_LAYER = 16
VOCAB_SIZE = 103168

# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
# oss: 'boto3:s3://model_weights/XXX'
# MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
# SAVE_CKPT_FOLDER = "local:llm_ckpts"
SAVE_CKPT_FOLDER = "local:llm_ckpts"
# LOAD_CKPT_FOLDER = "local:llm_ckpts/49"
ckpt = dict(
    enable_save_ckpt=True,
    # Path to save training ckpt.
    save_ckpt_folder=SAVE_CKPT_FOLDER,
    # Path to continue training ckpt (load model weights and scheduler/context states).
    # load_ckpt_folder=LOAD_CKPT_FOLDER,
    # Path to initialize with given model weights.
    # load_model_only_folder=MODEL_ONLY_FOLDER,
    checkpoint_every=20,
    # Wheter to load optimizer states when continuing training.
    load_optimizer=True,
)

TRAIN_FOLDER = "local:../lm_data/alpaca_data/train/en"
data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=4,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=2,
    pack_sample_into_one=False,
    total_steps=20,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    # train_folder=TRAIN_FOLDER,
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
    zero_overlap_communication=True,
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
    checkpoint=False,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
)
"""
zero1 parallel:
    1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
        so parameters will be divided within the range of dp.
    2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
    3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
pipeline parallel: pipeline parallel size, only 1 is accepted currently.
tensor parallel: tensor parallel size, usually the number of GPUs per node, only 1 is accepted currently.
"""
parallel = dict(
    zero1=8,
)

cudnn_deterministic = False
cudnn_benchmark = False

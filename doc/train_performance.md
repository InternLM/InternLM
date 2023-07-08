## 训练性能

InternLM 深度整合了 Flash-Attention, Apex 等高性能模型算子，提高了训练效率。通过构建 Hybrid Zero 技术，实现计算和通信的高效重叠，大幅降低了训练过程中的跨节点通信流量。InternLM 支持 7B 模型从 8 卡扩展到 1024 卡，千卡规模下加速效率可高达 90%，训练吞吐超过 180TFLOPS，平均单卡每秒处理的 token 数量超过3600。下表为 InternLM 在不同配置下的扩展性测试数据：

| GPU Number         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS (Tokens/GPU/Second) | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |


我们在GPU集群上测试了多种并行配置下，InternLM训练7B模型的性能。在每组测试中，每张GPU在单次迭代中处理的token数量一致。测试使用的硬件和参数配置如下表所示：

| 硬件                    | 硬件型号                      |
| ----------------------- | ----------------------------- |
| GPU                     | nvidia_a100-sxm4-80gb         |
| Memory                  | 2TB                           |
| Inter-machine bandwidth | 4 * 100Gb RoCE                |
| CPU                     | 128 core Intel(R) Xeon(R) CPU |

| 超参      | tp=1 | tp=2 |
| --------- | ---- | ---- |
| micro_num | 4    | 4    |
| micro_bsz | 2    | 4    |
| seq_len   | 2048 | 2048 |

InternLM中`zero1`的配置决定了优化器状态的分配范围。
- `zero1=-1`表明优化器状态分布在全部数据并行节点（等同于Deepspeed Zero-1的效果）
- `zero1=8，tp=1`的情况下，优化器状态分布在单节点8张GPU内，并且不同节点上的优化器状态保持一致。

### 吞吐量测量

吞吐量定义为TGS，平均每GPU每秒处理的token的数量（Tokens per GPU per Second）。在该项测试的训练配置中，`pack_sample_into_one=False`，`checkpoint=False`, `dtype=torch.bfloat16`。测试结果如下表所示。采用`zero1=8，tp=1`，InternLM针对7B模型训练的扩展性，在千卡训练的加速效率可以达到`88%`。

| 并行配置         | 8卡  | 16卡 | 32卡 | 64卡 | 128卡 | 256卡 | 512卡 | 1024卡 |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| (tp=1, zero1=-1) | 4062 | 3842 | 3752 | 3690 | 3571  | 3209  | 2861  | 2271   |
| (tp=1, zero1=8)  | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| (tp=2, zero1=-1) | 3822 | 3595 | 3475 | 3438 | 3308  | 3094  | 2992  | 2785   |
| (tp=2, zero1=4)  | 3761 | 3658 | 3655 | 3650 | 3651  | 3653  | 3589  | 3486   |


<div align="left">
    <img src="../doc/imgs/train_performance.png" width="580"/>
</div>

### FLOPS测试
模型训练的计算量参考 [Megatron](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf) 论文中FLOPS计算方式。为了保证训练过程中的FLOPS恒定，在该项测试的训练配置中，`pack_sample_into_one=True`，`dtype=torch.bfloat16`。


当开启 Activation Ckpt后，测试结果如下表所示，InternLM针对7B模型的千卡训练，可以达到 `>180 TFLOPS`：

- TGS: Tokens per GPU per Second

- Global Bsz: 一个step中所有GPU处理的token数量


| TP | Zero1 | Pack Sample Into One | Activation Ckpt | GPU Num | Seq Len | Micro Bsz | Micro Num | Global Bsz | TGS | TFLOPS |
|-|-|-|-|-|-|-|-|-|-|-|
| 1 | 8 | TRUE | TRUE | 8 | 2048 | 8 | 1 | 0.125M | 3314 | 193 |
| 1 | 8 | TRUE | TRUE | 16 | 2048 | 8 | 1 | 0.25M | 3268 | 191 |  
| 1 | 8 | TRUE | TRUE | 32 | 2048 | 8 | 1 | 0.5M | 3323 | 188 |
| 1 | 8 | TRUE | TRUE | 64 | 2048 | 8 | 1 | 1M | 3217 | 188 |
| 1 | 8 | TRUE | TRUE | 128 | 2048 | 8 | 1 | 2M | 3260 | 187 |
| 1 | 8 | TRUE | TRUE | 256 | 2048 | 8 | 1 | 4M | 3215 | 187 |
| 1 | 8 | TRUE | TRUE | 512 | 2048 | 8 | 1 | 8M | 3199 | 186 |  
| 1 | 8 | TRUE | TRUE | 1024 | 2048 | 8 | 1 | 16M | 3163 | 184 |
| 1 | 8 | TRUE | TRUE | 512 | 2048 | 4 | 1 | 4M | 2963 | 173 |
| 1 | 8 | TRUE | TRUE | 1024 | 2048 | 2 | 1 | 4M | 2341 | 136 |
| 1 | 8 | TRUE | TRUE | 1024 | 2048 | 4 | 1 | 8M | 2796 | 160 |

当关闭 Activation Ckpt后，测试结果如下表所示：

| TP | Zero1 | Pack Sample Into One | Activation Ckpt | GPU Num | Seq Len | Micro Bsz | Micro Num | Global Bsz | TGS | TFLOPS |
|-|-|-|-|-|-|-|-|-|-|-|
| 1 | 8 | TRUE | FALSE | 8 | 2048 | 2 | 4 | 0.125M | 4103 | 183 |
| 1 | 8 | TRUE | FALSE | 16 | 2048 | 2 | 4 | 0.25M | 3939 | 177 |
| 1 | 8 | TRUE | FALSE | 32 | 2048 | 2 | 4 | 0.5M | 3919 | 176 |
| 1 | 8 | TRUE | FALSE | 64 | 2048 | 2 | 4 | 1M | 3944 | 174 |
| 1 | 8 | TRUE | FALSE | 128 | 2048 | 2 | 4 | 2M | 3928 | 173 |
| 1 | 8 | TRUE | FALSE | 256 | 2048 | 2 | 4 | 4M | 3920 | 173 |
| 1 | 8 | TRUE | FALSE | 512 | 2048 | 2 | 4 | 8M | 3900 | 173 |
| 1 | 8 | TRUE | FALSE | 1024 | 2048 | 2 | 4 | 16M | 3625 | 160 |
| 1 | 8 | TRUE | FALSE | 512 | 2048 | 2 | 2 | 4M | 3084 | 139 |  
| 1 | 8 | TRUE | FALSE | 1024 | 2048 | 2 | 1 | 4M | 2346 | 105 |
| 1 | 8 | TRUE | FALSE | 1024 | 2048 | 2 | 2 | 8M | 2817 | 124 |

<div align="left">
    <img src="../doc/imgs/flops.png" width="580"/>
</div>


# 模型与算法评测清单 (Model & Algorithm Inventory for Benchmarking)

本文档旨在为 PyTorch 到 MindSpore/Ascend 的兼容性基准测试（TorchBridgeBench）提供**全局视角的模型与算法候选池**。
该清单不仅用于指导后续基准测试用例的编写，也是撰写学术论文（如 ASPLOS/OSDI/ATC）时，论证框架兼容性广度与深度的重要参考素材。

清单分为三大类：**一、经典热门模型**；**二、经典算法流程**；**三、新型/前沿算法架构**。

***

## 一、 经典热门模型 (Classic & Popular Models)

这一部分主要包含工业界和学术界最广泛使用的模型，用于验证兼容性框架在“主流场景”下的可用性与性能。

### 1. 大语言模型 (Large Language Models, LLMs)

验证大规模 Transformer 架构的自回归生成能力、KV Cache 管理及显存优化。

- **LLaMA 系列 (LLaMA-2 / LLaMA-3)**
  - **关键 API/算子**: `torch.nn.functional.scaled_dot_product_attention` (SDPA), `torch.matmul`, `torch.nn.Embedding`, `RMSNorm` (通常基于 `torch.rsqrt` 和 `torch.mul` 实现), `RoPE` (旋转位置编码, 涉及 `torch.polar` 或 `torch.cos`/`torch.sin` 及切片操作)。
  - **难点**: KV-Cache 的 In-place 更新 (`tensor.copy_` 或 `index_put_`), 动态 Shape (Prefill vs Decode)。
- **Qwen / ChatGLM**
  - **关键 API/算子**: 类似于 LLaMA，额外关注 `torch.cat` (拼接 Past Key Values), `torch.masked_fill` (Causal Masking)。
- **BERT / RoBERTa**
  - **关键 API/算子**: `torch.nn.LayerNorm`, `torch.nn.MultiheadAttention`, `torch.nn.GELU`, `torch.bmm` (Batch MatMul)。
  - **难点**: 传统的 Encoder-only 架构，通常作为最基础的 Baseline 验证。

### 2. 计算机视觉模型 (Computer Vision Models)

验证 2D/3D 空间特征提取算子和传统 CNN 结构的兼容性。

- **ResNet 系列 (ResNet-50 / ResNet-101)**
  - **关键 API/算子**: `torch.nn.Conv2d`, `torch.nn.BatchNorm2d`, `torch.nn.MaxPool2d`, `torch.nn.AdaptiveAvgPool2d`, `torch.add` (残差连接)。
  - **难点**: BatchNorm 的 running mean/var 状态同步，inplace 操作 (`relu_()`, `add_()`)。
- **YOLO 系列 (YOLOv8 / YOLOv10)**
  - **关键 API/算子**: `torch.nn.Upsample` (或 `torch.nn.functional.interpolate`), `torch.split`, `torch.cat`, `torch.sigmoid`, `torch.exp` (Anchor Box 计算)。
  - **难点**: 复杂的张量切片与拼接，非极大值抑制 (NMS) 的算子支持 (`torchvision.ops.nms` 或自定义基于 `torch.where` 的过滤)。
- **ViT (Vision Transformer)**
  - **关键 API/算子**: `torch.nn.Conv2d` (Patch Embedding), `torch.einsum`, `torch.nn.LayerNorm`。

### 3. 生成式模型与多模态 (Generative & Multi-modal)

验证扩散模型及图文对齐能力。

- **Stable Diffusion (SD 1.5 / SDXL)**
  - **关键 API/算子**: `torch.nn.Conv2d`, `torch.nn.GroupNorm`, `torch.nn.functional.silu`, `torch.chunk`, 跨注意力机制 (Cross-Attention)。
  - **难点**: U-Net 结构中的密集内存分配与释放，CFG (Classifier-Free Guidance) 带来的大 Batch Size 计算。
- **CLIP**
  - **关键 API/算子**: `torch.cosine_similarity`, `torch.diag`, `torch.matmul`。

***

## 二、 经典算法与训练范式 (Classic Algorithms & Training Pipelines)

这一部分关注于验证在特定算子映射和执行机制下，框架能否稳定支撑业界标准的“微调-对齐”训练流水线。这些算法决定了框架是否具备真正的生产落地能力（特别是 Autograd 和优化器兼容性）。

### 1. LLM 全生命周期训练流水线 (Pretrain-SFT-RL)

验证从海量文本预训练到基于偏好对齐的端到端能力。

- **预训练 (Pre-training) / 监督微调 (SFT)**
  - **核心逻辑**: 自回归语言模型损失计算。
  - **关键 API/算子**: `torch.nn.CrossEntropyLoss` (配合 `ignore_index=-100`), `torch.roll` (或 `tensor[:-1], tensor[1:]` 位移对齐), `optimizer.step()`, `loss.backward()`。
  - **难点**: 梯度的正确累加 (`loss / gradient_accumulation_steps`)、大 Batch 下交叉熵数值稳定性（依赖 `log_softmax`）。
- **强化学习人类反馈对齐 (RLHF / PPO)**
  - **核心逻辑**: 策略模型 (Policy)、奖励模型 (Reward) 与价值评估 (Value) 之间的复杂梯度流动。
  - **关键 API/算子**: `torch.gather` (获取特定 Token 的 log\_prob), `torch.clamp` (Clipped Surrogate Objective), `torch.mean`, `torch.var` (Advantage Normalization)。
  - **难点**: 多个模型在同一进程内同时存在，梯度流图异常复杂，内存碎片化严重。
- **直接偏好优化 (DPO)**
  - **核心逻辑**: 替代 PPO 的轻量级对齐算法，通过计算 Policy 与 Reference 模型对 Chosen/Rejected 样本的 Log Prob 差异构建损失。
  - **关键 API/算子**: `torch.nn.functional.log_softmax`, `torch.exp`, `torch.sigmoid` (或 `torch.nn.functional.logsigmoid`), 复杂的张量广播运算。

### 2. 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)

验证框架能否通过拦截或替换原生层，在极低资源下完成微调。

- **LoRA (Low-Rank Adaptation)**
  - **核心逻辑**: 在预训练冻结权重的旁路添加降维与升维矩阵。
  - **关键 API/算子**: `torch.nn.Linear`, `torch.baddbmm` 或常规 `matmul`, `torch.nn.Parameter` (requires\_grad=True 与 False 混合)。
  - **难点**: 梯度在部分子图流动，部分阻断，考验 Autograd 引擎对动态图的支持能力。

### 3. 高性能与内存优化算法 (Performance & Memory Optimization)

验证兼容框架是否支持最新的加速特性，或者提供同等语义的 Lowering。

- **FlashAttention (v1 / v2)**
  - **核心逻辑**: 融合的注意力机制 (Fused Attention)，极大减少 HBM 读写。
  - **关键 API/算子**: `torch.nn.functional.scaled_dot_product_attention` (通常在后端被映射到硬件特定的 Fused Kernel，如 NPU 的 `npu_flash_attention`)。
- **混合精度训练 (AMP) / Zero Redundancy Optimizer**
  - **核心逻辑**: FP32 主权重 + FP16/BF16 激活与梯度。
  - **关键 API/算子**: `torch.autocast`, `torch.cuda.amp.GradScaler` (对应 NPU/MindSpore 的 loss scaling 机制)。

***

## 三、 新型与前沿算法架构 (Novel & Cutting-Edge Architectures)

这一部分主要包含尚未成为绝对主流，但代表了下一代模型架构趋势的算法。它们通常引入了新的数学原语、控制流或张量变换，是测试“兼容性系统是否具备前瞻性和鲁棒性”的极佳靶点。

### 1. 状态空间模型 (State Space Models, SSMs)

打破 Transformer 长度限制，利用状态空间机制实现线性时间复杂度的序列建模。

- **Mamba / Mamba-2**
  - **核心逻辑**: 硬件感知的高效状态转移扫描 (Selective Scan)，以及依赖输入的依赖门控。
  - **关键 API/算子**: `torch.nn.functional.linear`, `torch.cumsum`, `torch.exp`, 复杂的 Einsum 和 Cumprod 操作 (通常依赖高度定制的 CUDA/NPU C++ Kernel)。
  - **难点**: 极强的序列依赖性，如果缺乏底层融合算子 (Fused Kernel) 支持，用纯 PyTorch API 拼装会导致极高的性能退化。

### 2. 稀疏专家混合模型 (Mixture of Experts, MoE)

以较小的推理计算代价扩大模型参数规模。

- **MoE 路由与负载均衡 (Router & Load Balancing)**
  - **核心逻辑**: 每个 Token 根据 Router 网络计算出的分数，被分配给 Top-K 个专家进行处理。
  - **关键 API/算子**: `torch.topk`, `torch.gather`, `torch.scatter_add_`, `torch.bincount`, `torch.cumsum`。
  - **难点**: 动态控制流 (Dynamic Control Flow)。每个批次中分配给不同专家的 Token 数量是不确定的 (Dynamic Shape)，这给静态图编译 (Graph Mode) 带来了巨大挑战。同时需要额外的辅助损失 (Auxiliary Loss) 来实现负载均衡。

### 3. 新型注意力与上下文架构 (Novel Attention & Context)

探索提升长上下文处理能力的新型算法。

- **流形约束超连接 (mHC / manifold constrained Hyper Connection)**
  - **核心逻辑**: 用于残差流带宽扩展。它打破了传统 Transformer 每一层只有一个 $h_{dim}$ 维度的残差流限制，允许残差流传递 4 个甚至更多个隐向量。在每一层的输入输出进行多对一或一对多的混合与分发，并通过 Sinkhorn-Knopp 算法（流形约束）限制这四个向量之间的混合矩阵，确保特征混合时不会导致范数爆炸。
  - **关键 API/算子**: `torch.einsum` (用于执行高维的流混合), `torch.nn.functional.normalize` (L1 归一化用于 Sinkhorn 迭代), `torch.softmax`, `torch.cat`, 以及高频的动态 `nn.Parameter` 操作。
  - **难点**: Sinkhorn 算法包含大量的迭代归一化与 Softmax 操作，这在某些静态图编译器（Graph Mode）或算子拦截层容易引发数值溢出或控制流（While-loop）展开失败。且 `einsum` 字符串解析映射到 NPU 时容易出现维度对齐问题。
- **环形注意力 (Ring Attention) / 块向注意力 (Blockwise Attention)**
  - **核心逻辑**: 将极长的序列分块计算并在设备间传递 KV 块。
  - **关键 API/算子**: `torch.distributed.send` / `recv`, 或 `torch.distributed.all_gather` 等通信算子。
  - **难点**: 兼容层对分布式后端 (NCCL -> HCCL) 的支持与映射，以及通信与计算的重叠调度。


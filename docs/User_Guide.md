# TorchBridgeBench 使用指南 (User Guide)

TorchBridgeBench 是一个用于系统化评测 **PyTorch 到 MindSpore/Ascend 兼容层方案**（如 `torch4ms`, `mindnlp_patch`, `mindtorch`, `torch-npu` 等）的统一基准测试框架。

它不仅评估代码“能不能跑”（Compatibility），还评估“结果对不对”（Correctness）以及“性能代价有多大”（Performance），最终生成多维度的诊断报告。

---

## 1. 核心架构与功能

本系统采用模块化设计，主要包含以下几个核心部分：
*   **Adapter 机制 (`adapters/`)**: 将不同的兼容框架封装为统一的调用接口。不管底层是基于 AST 注入、Monkey Patch 还是拦截器，上层测试代码都无需修改。
*   **层次化测试套件 (`suites/`)**:
    *   `operator_smoke.py`: 基础算子测试（如 `add`, `matmul`, `softmax` 等）。
    *   `module_smoke.py`: 基础网络模块测试（如 `Linear`, `Conv2d`, `LayerNorm` 等）。
    *   `model_smoke.py`: 经典微型网络结构测试（如 `TinyResNet`, `TinyLSTM` 等）。
    *   `autograd_smoke.py`: 自动求导与反向传播一致性测试。
    *   `end2end_training.py`: 基于合成数据的 MNIST 端到端完整训练流程测试（涵盖前向、Loss 计算、反向与参数更新）。
    *   `novel_algorithm_smoke.py`: 前沿算法架构兼容性测试（如 `mHC` 流形约束超连接）。
*   **环境预检 (`preflight.py`)**: 运行前自动检查 NPU 驱动状态 (`npu-smi`) 以及关键依赖库（MindSpore, PyTorch, torch-npu）的版本匹配情况。
*   **Agent 自动诊断系统 (`agent_core/` & `agent_run.py`)**: 当测试失败时，通过大语言模型 (LLM) 结合受限环境沙箱，自动执行依赖库降级、代码追踪等操作，输出根本原因分析（如 API 版本漂移或底层 C++ 类型转换错误）。
*   **报告生成 (`report_generator.py`)**: 汇总所有后端的 JSON 结果，生成直观的 Markdown 矩阵表格和失败诊断清单。

---

## 2. 环境依赖与隔离

为了确保不同兼容层方案之间不发生依赖冲突（特别是 MindSpore 版本冲突），系统依赖于**多虚拟环境隔离**。

默认配置下（见 `run_all.sh`），系统使用以下路径：
*   **`VENV_DIR`** (`.venv-torch4ms`): 用于运行依赖最新版/特定补丁的框架（如 `mindnlp_patch`, `torch-npu`）。
*   **`CLEAN_VENV_DIR`** (`.venv-torch4ms-clean`): 包含干净的 MindSpore 2.8.0 环境，用于运行需要严格环境隔离的框架（如 `torch4ms`, `mindtorch`）。

> **前置要求**: 
> 1. Linux 操作系统（推荐 aarch64 架构以原生支持 Ascend NPU）。
> 2. 已安装正确版本的 CANN 驱动与 Ascend Toolkit。
> 3. Python 3.8+ (推荐 3.10)。
> 4. 某些前瞻性测试（如 mHC）需要额外安装 `x-transformers` 等第三方库。

---

## 3. 运行指南

### 3.1 一键运行完整基准测试

这是最常用的命令。它会依次拉起所有的虚拟环境，对所有配置的后端执行全量测试，并在最后自动聚合生成报告：

```bash
cd /root/autodl-tmp/torchbridgebench
bash run_all.sh
python report_generator.py
```

### 3.2 运行指定的后端

如果你只想测试某一个特定的后端（例如在开发调试 `torch4ms` 时），可以直接使用 CLI 工具：

```bash
# 激活对应的虚拟环境
source /root/autodl-tmp/mindnlp/.venv-torch4ms-clean/bin/activate

# 运行特定后端，结果保存为 JSON
python cli.py --backend torch4ms --output report_torch4ms.json
```

*支持的 backend 参数：`torch-npu`, `torch4ms`, `mindtorch`, `mindnlp_patch`, `torch` (基线).*

### 3.3 定向运行 suite / test / layer

CLI 支持按 suite、test、layer 缩小执行范围，便于复现失败和验证修复，不需要再手写 Python 片段。

```bash
# 单跑一个 suite
python cli.py --backend torch4ms \
  --suite module_smoke \
  --output artifacts/reports/module_smoke.json

# 单跑一个测试，suite 和 test 分开写
python cli.py --backend torch4ms \
  --suite module_smoke \
  --test test_batchnorm2d_module \
  --output artifacts/reports/batchnorm.json

# 单跑一个测试，使用 suite/test 形式
python cli.py --backend torch4ms \
  --test module_smoke/test_avgpool2d_module \
  --output artifacts/reports/avgpool.json

# 单跑一个 layer
python cli.py --backend torch4ms \
  --layer end2end \
  --output artifacts/reports/end2end.json
```

`--suite`、`--test`、`--layer` 均可重复传入，也可使用逗号分隔列表。

### 3.4 触发 Agent 自动诊断

当发现某个测试用例失败时，可以调用 Agent 进行自动化溯源与环境突变测试（例如验证是否为版本不兼容导致）。

```bash
# 示例：让 Agent 诊断 torch4ms 在 maxpool2d 模块上的失败
python agent_run.py \
  --backend torch4ms \
  --venv /root/autodl-tmp/mindnlp/.venv-torch4ms-clean \
  --suite module_smoke \
  --test test_maxpool2d_module \
  --error "module 'mindspore.ops' has no attribute 'MaxPool2D'"
```
*(注意：需要配置好相应的 LLM API Key，如 DeepSeek 或 OpenAI。)*

---

## 4. 产出物说明

运行结束后，项目根目录下会生成以下核心文件：
1.  **`report_*.json`**: 各个后端的原始详细测试结果，包含通过状态、运行耗时（毫秒）、报错栈信息等。
2.  **`benchmark_report.md`**: 最终的合并诊断报告。包含：
    *   **Summary**: 总体通过率对比。
    *   **Detailed Test Results**: 细粒度到具体用例的通过/失败矩阵及性能耗时。
    *   **Failure Diagnostics**: 将晦涩的底层 C++ 报错进行归类的错误清单。

---

## 5. 未来演进路线 (Future Roadmap)

TorchBridgeBench 是一个持续发展的生态系统，旨在跟上学术界与工业界最前沿的 AI 框架发展趋势。以下是我们正在规划或即将加入的核心特性：

### 5.1 更大规模的模型与算法验证
虽然目前系统已经包含了 MNIST 端到端训练和基础算子测试，但针对真实业务场景的大规模压力测试将是下一步的重点：
*   **大语言模型 (LLM) 推理与微调**: 引入 LLaMA-2/3、Qwen 等主流开源大模型的推理流程测试，涵盖 KV-Cache 动态更新、SDPA (Scaled Dot-Product Attention) 算子映射等高难度场景。
*   **分布式与通信算子测试**: 针对多 NPU 节点的 `torch.distributed` 兼容性测试，验证 `all_gather`, `reduce_scatter` 在 MindSpore 底层 HCCL 的映射准确性与通信开销。
*   **新型算法深度评测**: 进一步扩充 `novel_algorithm_smoke.py`，加入 Mamba (SSMs 状态空间模型)、MoE (混合专家路由) 等具有动态控制流和特殊约束的前沿算法。

### 5.2 Agent 系统的自我进化
当前的 Agent 系统主要用于诊断和环境隔离验证，未来它将具备更强的“自我修复”能力：
*   **Auto-Fix Patch 生成**: 当 LLM 识别出底层 `dispatch` 错误或 API 拼写不一致时，不仅输出分析报告，还能直接生成并应用 Git Patch，在隔离环境中进行“试错式修复”。
*   **知识库增强 (RAG)**: 接入 MindSpore 和 PyTorch 的最新官方文档向量库，让 Agent 在遇到未知的 C++ 底层报错时，能通过检索历史 Issue 和社区问答进行更精准的归因。

### 5.3 深度性能分析 (Profiling & Metrics)
除了目前的总耗时（毫秒）统计，未来将引入更细粒度的性能画像工具：
*   **显存占用峰值追踪**: 测量不同后端在相同模型结构下的显存碎片化程度和峰值占用 (Peak Memory Usage)。
*   **算子级 Profiling 导出**: 支持自动导出 Chrome Trace 格式的时间线，帮助框架开发者直观定位性能瓶颈（如 Host-Device 同步开销、算子下发延迟等）。

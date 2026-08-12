# 端到端指南：在 `vllm/vllm-openai-rocm:nightly` 容器中执行 `dsv4_fp4_mi355x_vllm_mtp.sh`

## 脚本概述

该脚本运行一个 **Agentic trace replay benchmark**，测试 DeepSeek-V4-Pro (FP4) 在 AMD MI355X GPU 上的推理性能。

核心流程：

```
下载模型 → 安装依赖 → 启动 vLLM 服务 → (可选) 启动 router → 运行 aiperf trace replay → 聚合结果
```

---

## 核心环境事实

| 项目 | 说明 |
|------|------|
| **镜像** | `vllm/vllm-openai-rocm:nightly` |
| **模型路径** | `/shared/models/deepseek-ai/DeepSeek-V4-Pro`（本地已下载） |
| **vLLM** | 已预装，ROCm 版本 |
| **Python** | 系统 Python 3.10 |
| **缺失的包** | nightly 镜像可能缺少 `Pillow`、`fastapi`、`uvicorn`，脚本会自动补装 |
| **aiperf** | 脚本创建独立 venv（Python 3.11），不污染 vLLM site-packages |

---

## 步骤 1：启动容器

```bash
docker run -it --rm \
    --name vllm-rocm-bench \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /shared/models:/shared/models \
    -v $(pwd):/workspace/InferenceX \
    -v /tmp/results:/workspace/results \
    -w /workspace \
    vllm/vllm-openai-rocm:nightly \
    /bin/bash
```

> `/dev/kfd` 和 `/dev/dri` 是 ROCm 必须的设备。模型目录 `/shared/models` 挂载为只读即可。结果目录 `/tmp/results` 可按需调整。

## 步骤 2：验证容器环境

```bash
# 确认 GPU 可见（应看到 8 个 MI355X）
rocm-smi --showmemuse

# 确认 vLLM 已安装
python3 -c "import vllm; print(vllm.__version__)"

# 确认模型文件存在
ls /shared/models/deepseek-ai/DeepSeek-V4-Pro | head -20
```

## 步骤 3：设置环境变量

### 模型

```bash
export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PATH="/shared/models/deepseek-ai/DeepSeek-V4-Pro"
```

### 并行拓扑（三选一）

并行拓扑决定了 8 张 GPU 之间如何分工。三种模式都需要设 `TP=8`，但脚本内部会根据 `DP_ATTENTION` 开关自动重解释 `$TP` 的含义：

| 模式 | TP | EP_SIZE | DP_ATTENTION | 实际传给 vLLM 的参数 |
|------|:--:|:-------:|:------------:|---------------------|
| **A** Pure TP | 8 | 1 | false | `--tensor-parallel-size 8 --data-parallel-size 1` |
| **B** TP + EP | 8 | 8 | false | `--tensor-parallel-size 8 --data-parallel-size 1 --enable-expert-parallel` |
| **C** DEP | 8 | 8 | true | `--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel` |

- **Pure TP（A）**：8 GPU 联合做 attention TP 分片，无 expert parallel。单引擎，TPOT 最低，batch 最小。
- **TP + EP（B）**：attention 继续 TP 分片，MoE expert 额外做 EP 分片。
- **DEP（C）**：TP 强制为 1，`$TP` 值转为 `--data-parallel-size`。每个 GPU 独立做 attention，MoE expert 跨 8 个 DP rank 分片。大并发下总吞吐最高。

```bash
# ====== 选项 A：Pure TP（推荐默认） ======
export TP=8
export EP_SIZE=1
export DP_ATTENTION="false"

# ====== 选项 B：TP + EP ======
# export TP=8
# export EP_SIZE=8
# export DP_ATTENTION="false"

# ====== 选项 C：DEP ======
# export TP=8
# export EP_SIZE=8
# export DP_ATTENTION="true"
```

### KV Offload（三选一）

```bash
# ====== 选项 A：vllm-native（推荐首次尝试）======
export KV_OFFLOADING="dram"
export KV_OFFLOAD_BACKEND="vllm-native"
export TOTAL_CPU_DRAM_GB=2500

# ====== 选项 B：不使用 KV offload ======
# export KV_OFFLOADING="none"

# ====== 选项 C：mooncake 或 lmcache（需要额外编译步骤）======
# export KV_OFFLOADING="dram"
# export KV_OFFLOAD_BACKEND="mooncake"   # 或 "lmcache"
```

### 运行参数

```bash
export CONC=32
export DURATION=1200
export RESULT_DIR="/workspace/results"
export PORT=8888
```

### 快速验证（smoke test）

压低并发和时长，确认链路通畅后再跑完整配置：

```bash
export CONC=4 DURATION=300
export AIPERF_UNSAFE_OVERRIDE="true"
export AIPERF_EXPERIMENTAL_FAST="1"
```

## 步骤 4：执行脚本

```bash
cd /workspace/InferenceX
mkdir -p /workspace/results
bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm_mtp.sh
```

由于 `MODEL_PATH` 已设置且目录存在，脚本会直接跳过 `hf download`。

---

## 时间线预估

| 阶段 | 时间 | 说明 |
|------|------|------|
| 依赖安装 | ~5 min | aiperf venv（uv 创建 Python 3.11 隔离环境） |
| 模型加载 | 1.5–2.5 h | 805 GiB 从本地磁盘加载 + CUDA Graph 编译 |
| 预热 | ~10 min | trace warmup requests |
| Benchmark | = `DURATION` | aiperf trace replay |
| 结果聚合 | ~2 min | JSON + 分布分析 + 验证 |

首次运行约 3–5 小时，后续运行约 30–60 分钟（模型已在 cache 中）。

---

## 运行时监控

```bash
# 终端 2：观察 vLLM 引擎日志
tail -f /workspace/results/server.log
# 看到 "Uvicorn running on http://0.0.0.0:8888" 表示引擎就绪

# 终端 3：观察 benchmark 进度
tail -f /workspace/results/benchmark.log

# 终端 4：GPU 状态
watch -n 1 rocm-smi --showmemuse
```

---

## 结果文件

运行结束后，`/workspace/results/` 下的关键产物：

```
results/
├── server.log                     # vLLM 引擎完整日志
├── benchmark.log                  # aiperf trace replay 输出
├── benchmark_command.txt          # 实际执行的 aiperf 命令
├── vllm_command.txt               # 实际执行的 vllm serve 命令
└── aiperf_artifacts/
    ├── profile_export.json        # 聚合性能数据
    ├── profile_export.jsonl       # 逐请求时序
    └── server_metrics_export.csv  # Prometheus metrics 时序
```

聚合后的最终 JSON 结果写入 `/workspace/`。

---

## 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `Server died before becoming healthy` | 权重加载 OOM | 检查 `server.log`，确认 `rocm-smi` 显示 8 卡，尝试编辑脚本将 `--gpu-memory-utilization` 从 `0.8` 降到 `0.75` |
| 加载超时（>3h） | CUDA Graph capture 卡住 | 脚本默认 `mode=3, FULL_AND_PIECEWISE`，尝试临时改为 `mode=0` |
| `hf download` 仍被触发 | `MODEL_PATH` 目录为空或不可读 | 确认 `ls /shared/models/deepseek-ai/DeepSeek-V4-Pro` 有内容 |
| `KV_OFFLOADING must be set` | 环境变量漏设 | 必须显式设置 `KV_OFFLOADING=dram` 或 `KV_OFFLOADING=none` |
| Mooncake/LMCache 编译失败 | 缺少编译依赖 | 首次运行建议先用 `vllm-native`，后续再切换 |

---

## 附录：一键 Smoke Test 脚本

```bash
#!/bin/bash
# smoke_test.sh — 最小化验证链路是否通畅

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PATH="/shared/models/deepseek-ai/DeepSeek-V4-Pro"
export TP=8
export EP_SIZE=1
export DP_ATTENTION="false"
export CONC=4
export KV_OFFLOADING="dram"
export KV_OFFLOAD_BACKEND="vllm-native"
export TOTAL_CPU_DRAM_GB=2500
export DURATION=300
export RESULT_DIR="/workspace/results_smoke"
export PORT=8888
export EVAL_ONLY="false"
export AIPERF_UNSAFE_OVERRIDE="true"
export AIPERF_EXPERIMENTAL_FAST="1"

mkdir -p "$RESULT_DIR"
cd /workspace/InferenceX
bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm_mtp.sh
```

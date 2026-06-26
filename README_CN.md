# Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic

**Timely Machine** 的官方开源代码，包含评测框架和 RL 训练代码。

[English README](README.md) · [Eval Package](src/timely_eval) · [RL Code](rl/internbootcamp_v2) · [RL README](rl/internbootcamp_v2/README.md)

## 两条代码路径

本仓库有两个相互区分的部分。复现实验评测请使用 **Timely Eval**；训练 timer-aware agent 请使用 **Timely RL**。

| 部分 | 位置 | 入口 | 用途 |
| --- | --- | --- | --- |
| **Timely Eval** | `src/timely_eval/` | `timely-eval ...` | 可直接安装运行的时间感知评测框架。 |
| **Timely RL** | `rl/internbootcamp_v2/` | `scripts/run_llm_timer_rl_example.sh` | RL 训练代码、环境 server、tool backend 和本地 verl 训练栈。 |

## Timely Eval

Timely Eval 是本仓库中轻量、可安装的评测包，支持 OpenAI-compatible API endpoint，也支持本地 vLLM/SGLang 服务。

| 评测轨道 | 命令 | 需要的资源 |
| --- | --- | --- |
| 通用推理 | `timely-eval general` | JSONL 格式的问题和答案。 |
| Agentic ML | `timely-eval agentic-ml` | 公开 train/test CSV、private labels、prompt template。 |
| Interactive Jericho | `timely-eval interactive` | 本地 Jericho/Frotz 游戏文件，例如 `zork1.z5`。 |

Interactive games time-performance 实验图：[picture_time_performance_acl.pdf](assets/picture_time_performance_acl.pdf)

## Timely RL

Timely RL 是训练侧代码，比 Eval 包更重，有独立依赖、运行时服务和启动脚本。

<p align="center">
  <img src="assets/RL_pipeline%20%281%29.png" width="92%" alt="Timely Machine RL pipeline">
</p>

RL 主入口：

```text
rl/internbootcamp_v2/internbootcamp/bootcamps/Basic_LLM_timer
```

典型启动顺序：

1. 启动一个任务环境 server：general timer、Agentic ML timer 或 Jericho。
2. 启动分布式 tool backend：`scripts/run_llm_timer_tool_server.sh`。
3. 启动训练：`scripts/run_llm_timer_rl_example.sh`。

RL 详细说明见 [rl/internbootcamp_v2/README.md](rl/internbootcamp_v2/README.md)。

## What's New

- **2026.06** 发布 Timely Eval、Timely RL、toy examples、单测和 smoke-test 说明。
- **2026.06** README 拆分为英文主页和中文说明。
- **2026.06** 文档和运行方式中明确区分 Eval 与 RL。

## 目录结构

```text
src/timely_eval/                 # Timely Eval 包和 CLI
examples/                        # 用于 smoke test 的合成 toy data/prompt
tests/                           # Eval 单测
rl/internbootcamp_v2/            # Timely RL 训练代码和本地 verl 训练栈
rl/internbootcamp_v2/README.md   # RL 安装和启动说明
assets/                          # README 图片和论文素材
```

## 安装 Eval

```bash
cd OpenSource
python -m venv .venv
source .venv/bin/activate
pip install -e ".[agentic,dev]"
```

如果需要 Jericho 交互评测，再安装：

```bash
pip install -e ".[interactive]"
python -m spacy download en_core_web_sm
```

设置 OpenAI-compatible API：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

本地服务：

```bash
export OPENAI_API_KEY="empty"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

## Eval Quick Start

### 通用推理评测

数据格式为 JSONL，每行包含：

```json
{"id": "case-1", "question": "Compute 2 + 2.", "answer": "4"}
```

测速：

```bash
timely-eval general \
  --mode speed_test \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME> \
  --workers 4
```

按相对时间预算评测：

```bash
timely-eval general \
  --mode time_test_w_tool \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME> \
  --workers 4 \
  --time-limit-probs 0.75 1.0 2.0 3.0
```

只分析已有结果：

```bash
timely-eval general \
  --mode result_analysis \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME>
```

### Agentic ML 评测

Agentic ML 需要：

- `data_dir/public/train.csv`
- `data_dir/public/test.csv`
- 评测用 private label 文件
- 告诉 agent 如何生成 `submission.csv` 的 prompt template

Toy 测速：

```bash
timely-eval agentic-ml \
  --mode speed_test \
  --benchmark-name toy_ml \
  --data-dir examples/data/agentic_ml_toy \
  --private-test-path examples/data/agentic_ml_toy/private/test.csv \
  --prompt-template examples/prompts/agentic_ml_toy.txt \
  --output-dir outputs/agentic_ml_toy \
  --model <MODEL_NAME> \
  --is-binary \
  --binary-label-column label \
  --batch-size 2 \
  --workers 2
```

完整时间预算评测：

```bash
timely-eval agentic-ml \
  --mode full_eval \
  --benchmark-name toy_ml \
  --data-dir examples/data/agentic_ml_toy \
  --private-test-path examples/data/agentic_ml_toy/private/test.csv \
  --prompt-template examples/prompts/agentic_ml_toy.txt \
  --output-dir outputs/agentic_ml_toy \
  --model <MODEL_NAME> \
  --is-binary \
  --binary-label-column label \
  --time-limit-probs 1.0 2.0 3.0
```

### Jericho 交互评测

需要本地游戏文件，例如 `zork1.z5`：

```bash
timely-eval interactive \
  --mode speed_eval \
  --game-path /path/to/zork1.z5 \
  --output-dir outputs/interactive_zork1 \
  --model <MODEL_NAME> \
  --batch-size 4 \
  --max-test-steps 64
```

```bash
timely-eval interactive \
  --mode full_eval \
  --game-path /path/to/zork1.z5 \
  --output-dir outputs/interactive_zork1 \
  --model <MODEL_NAME> \
  --batch-size 4 \
  --max-steps 30 50 100 200
```

## RL Quick Start

RL 不通过上面的 Eval 安装流程安装。完整环境请看 RL README：

```bash
cd rl/internbootcamp_v2
less README.md
```

Smoke-test 状态：

| 组件 | 状态 |
| --- | --- |
| General timer server | `/health`、`/register`、`/call` 已通过。 |
| Agentic ML server 和 `MLTimerTool` | 外部提供 ML data 后，代码执行、`submission.csv` 评测、计时已通过。 |
| Jericho server 和 tools | 外部提供 ROM 后，available actions、score、max score、`look`、end-game 已通过。 |
| 单步 RL smoke | Qwen3-8B 在一张 H200 上配合 actor/reference CPU offload 跑通。 |

## 开源清理说明

本版本已经移除原始工作目录中的敏感或机器相关内容：

- API key、服务账号 JSON、内部 IP、私有 endpoint
- 实验日志、模型输出、checkpoint、workspace、真实 private label、大型 RL 数据集、内部集群启动快照
- ML benchmark 的 `data_sources`
- Jericho game ROM 文件

`examples/` 中的数据是合成 toy data，只用于验证流程。

注意：Agentic ML 会执行模型生成的 Python。当前实现只是子进程和工作目录隔离，不是安全沙箱。运行不可信模型时，请放在容器或虚拟机中，并限制文件系统和网络访问。

## 开发

```bash
pip install -e ".[agentic,dev]"
PYTHONPATH=src pytest -q
```

发布前建议再扫一次敏感信息：

```bash
rg -n "sk-|CREDENTIALS|BEGIN .*PRIVATE KEY|/mnt/|http://10\\.|http://100\\.|http://172\\.|http://192\\.168\\." .
```

## Citation

```bibtex
@misc{timelyreasoner2026,
  title = {Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic},
  author = {Timely Machine Authors},
  year = {2026},
  note = {Code release}
}
```

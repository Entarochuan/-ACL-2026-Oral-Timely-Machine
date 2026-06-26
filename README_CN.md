# Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic

[English README](README.md) | [Timely Eval](src/timely_eval) | [RL Training](rl/internbootcamp_v2)

本目录是论文 **Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic** 的开源整理版。

代码分为两部分：

- **Timely Eval**：评测框架，位于 `src/timely_eval/`，可以直接安装运行。
- **Timely RL**：RL 训练代码，位于 `rl/internbootcamp_v2/`，入口、依赖和运行方式与 Eval 分开。

## What's New

- **[2026.06]** 发布 Timely Eval、RL 训练代码、toy examples、单测和 smoke-test 说明。
- **[2026.06]** RL 代码独立放在 `rl/internbootcamp_v2/`，与 Eval 入口区分。
- **[2026.06]** General reasoning、Agentic ML、Interactive Jericho 三类路径已完成 smoke test。

## Todo List

- ✅ Timely Eval 代码
- ✅ RL 训练代码
- ✅ Toy examples 和单测
- ✅ 英文 README + 中文 README
- ✅ RL pipeline 和 interactive-games 实验图
- ⬜ 论文链接和正式 citation
- ⬜ 模型/checkpoint 链接，如后续发布

## Highlights

- ⏱️ **时间感知评测**：评测模型是否能在显式时间预算下调整推理、写代码和交互策略。
- 🧮 **通用推理评测**：支持 AIME、MATH、GPQA 风格 JSONL 数据，两阶段测速和时间预算评测。
- 🧑‍💻 **Agentic ML 评测**：模型生成 Python 代码、执行、根据反馈迭代改进 `submission.csv`。
- 🎮 **交互环境评测**：可选 Jericho/Frotz 文本游戏交互评测。
- 🛠️ **OpenAI-compatible 后端**：支持 API 模型、本地 vLLM/SGLang 服务或其他兼容 endpoint。
- 🚀 **RL 训练代码**：包含 timer tools、tool server、local verl 和训练启动脚本。

RL pipeline：

<p align="center">
  <img src="assets/RL_pipeline%20%281%29.png" width="92%" alt="Timely Reasoner RL pipeline">
</p>

Interactive games time-performance 实验图：

- [picture_time_performance_acl.pdf](assets/picture_time_performance_acl.pdf)

## 目录结构

```text
src/timely_eval/                 # Eval 包和 CLI
examples/                        # 用于 smoke test 的合成 toy data/prompt
tests/                           # Eval 单测
rl/internbootcamp_v2/            # RL 训练代码、bootcamp tools、本地 verl 代码
rl/internbootcamp_v2/README.md   # RL 安装和启动说明
```

## 安装

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

本地服务也可以：

```bash
export OPENAI_API_KEY="empty"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

## 通用推理评测

数据格式为 JSONL，每行包含：

```json
{"id": "case-1", "question": "Compute 2 + 2.", "answer": "4"}
```

第一步，测速：

```bash
timely-eval general \
  --mode speed_test \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME> \
  --workers 4
```

第二步，按相对时间预算评测：

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

第三步，只分析已有结果：

```bash
timely-eval general \
  --mode result_analysis \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME>
```

## Agentic ML 评测

需要提供公开训练/测试数据、评测用标签文件和 prompt 模板。toy 示例：

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

## Jericho 交互评测

需要本地游戏文件，例如 `zork1.z5`：

```bash
timely-eval interactive \
  --mode speed_eval \
  --game-path /path/to/zork1.z5 \
  --output-dir outputs/interactive_zork1 \
  --model <MODEL_NAME>
```

```bash
timely-eval interactive \
  --mode full_eval \
  --game-path /path/to/zork1.z5 \
  --output-dir outputs/interactive_zork1 \
  --model <MODEL_NAME> \
  --max-steps 30 50 100 200
```

## RL 训练代码

RL 部分和 Eval 部分是分开的。RL 代码请参考：

```bash
cd rl/internbootcamp_v2
less README.md
```

主要入口：

```text
rl/internbootcamp_v2/internbootcamp/bootcamps/Basic_LLM_timer
```

启动顺序：

1. 先启动任务环境 server：general timer、Agentic ML timer 或 Jericho。
2. 再启动分布式 tool backend：`scripts/run_llm_timer_tool_server.sh`。
3. 最后启动 RL：`scripts/run_llm_timer_rl_example.sh`。

当前 smoke-test 状态：

- General timer server：`/health`、`/register`、`/call` 已通过。
- Agentic ML server 和 `MLTimerTool`：外部提供 `ML_source/data_sources` 后，代码执行、`submission.csv` 评测和计时已通过。
- Jericho server 和 tools：外部提供 ROM 后，available actions、score、max score、`look`、end-game 已通过。
- Qwen3-8B 单步 RL smoke 在一张 H200 上配合 actor/reference CPU offload 跑通。

## 数据和开源清理

本版本已经移除原始工作目录中的敏感或机器相关内容：

- 不包含 API key、服务账号 JSON、内部 IP、私有 endpoint。
- 不包含实验日志、模型输出、checkpoint、workspace、真实 private label、大型 RL 数据集、内部集群启动快照。
- 不包含 ML benchmark 的 `data_sources`。
- 不包含 Jericho game ROM 文件。

`examples/` 里的数据是合成 toy data，只用于验证流程。

注意：Agentic ML 会执行模型生成的 Python。当前实现只是子进程和工作目录隔离，不是安全沙箱。运行不可信模型时，请放在容器或虚拟机中，并限制文件系统和网络访问。

## 开发与发布前检查

```bash
pip install -e ".[agentic,dev]"
PYTHONPATH=src pytest -q
```

发布前建议再扫一次敏感信息：

```bash
rg -n "sk-|CREDENTIALS|BEGIN .*PRIVATE KEY|/mnt/|http://10\\.|http://100\\.|http://172\\.|http://192\\.168\\." .
```

预期只应命中文档中的泛化变量名，不应出现真实 key、私有路径或内部 endpoint。

## Citation

```bibtex
@misc{timelyreasoner2026,
  title = {Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic},
  author = {Timely Machine Authors},
  year = {2026},
  note = {Code release}
}
```

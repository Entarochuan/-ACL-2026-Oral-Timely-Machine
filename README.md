# Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic

[Paper](#citation) | [Project Page](#timely-machine-awareness-of-time-makes-test-time-scaling-agentic) | [中文说明](README_CN.md) | [Timely Eval](src/timely_eval) | [RL Training](rl/internbootcamp_v2)

This repository contains the official open-source evaluation and RL training code for **Timely Machine**, a framework for studying whether LLM agents can reason, code, and interact under explicit time budgets.

## What's New

- **[2026.06]** We release the open-source Timely Eval package, RL training code, toy examples, unit tests, and smoke-tested launch notes.
- **[2026.06]** The RL code is separated from the evaluation package under `rl/internbootcamp_v2/`.
- **[2026.06]** General reasoning, Agentic ML, and Interactive Jericho smoke paths have been tested with local/OpenAI-compatible model serving.

## Todo List

- ✅ Timely Eval code release
- ✅ RL training code release
- ✅ Toy examples and unit tests
- ✅ English README with Chinese README link
- ✅ RL pipeline and interactive-games experiment figures
- ⬜ Paper link and citation metadata
- ⬜ Public model/checkpoint links, if released

## Highlights

<p align="center">
  <img src="assets/RL_pipeline%20%281%29.png" width="92%" alt="Timely Reasoner RL pipeline">
</p>

Timely Machine evaluates and trains agents that are aware of elapsed time during test-time scaling. The release covers static reasoning, code-writing ML agents, and interactive text-game agents, with a unified time-tool interface and a separate RL pipeline for timer-aware training.

- ⏱️ **Time-aware test-time scaling**: agents can call a `get_duration` time tool and adapt behavior under explicit time budgets.
- 🧮 **General reasoning evaluation**: supports JSONL math/static reasoning datasets with `speed_test`, `time_test_w_tool`, and `result_analysis`.
- 🧑‍💻 **Agentic ML evaluation**: agents write Python code, execute it, inspect feedback, and iteratively improve `submission.csv`.
- 🎮 **Interactive-game evaluation**: optional Jericho/Frotz evaluation for multi-turn tool-use agents.
- 🚀 **RL training code**: timer tools, local environment servers, distributed tool backend, and verl-based training scripts are included.

Interactive games time-performance figure: [picture_time_performance_acl.pdf](assets/picture_time_performance_acl.pdf)

## Timely Tasks

| Track | Entry | What It Tests |
| --- | --- | --- |
| General Reasoning | `timely-eval general` | Time-aware static reasoning on math/QA-style JSONL data. |
| Agentic ML | `timely-eval agentic-ml` | Time-aware iterative coding and submission generation. |
| Interactive Jericho | `timely-eval interactive` | Time-aware multi-turn interaction with text-game environments. |
| RL Training | `rl/internbootcamp_v2/` | GRPO-style training with timer-aware tools and environment servers. |

## Repository Structure

```text
src/timely_eval/                 # Evaluation package and CLI
examples/                        # Synthetic toy data and prompts for smoke tests
tests/                           # Unit tests for eval utilities
rl/internbootcamp_v2/            # RL training code, timer tools, tool servers, local verl
rl/internbootcamp_v2/README.md   # RL-specific setup and launch notes
assets/                          # README figures and paper assets
```

## Setup

```bash
cd OpenSource
python -m venv .venv
source .venv/bin/activate
pip install -e ".[agentic,dev]"
```

Install optional Jericho/Frotz dependencies only for interactive-game evaluation:

```bash
pip install -e ".[interactive]"
python -m spacy download en_core_web_sm
```

Configure an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

For local model servers:

```bash
export OPENAI_API_KEY="empty"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

## Quick Start

### General Reasoning

Input JSONL:

```json
{"id": "case-1", "question": "Compute 2 + 2.", "answer": "4"}
```

Measure baseline duration:

```bash
timely-eval general \
  --mode speed_test \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME> \
  --workers 4
```

Evaluate under relative time budgets:

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

Analyze existing results:

```bash
timely-eval general \
  --mode result_analysis \
  --data-path examples/data/general_reasoning_toy.jsonl \
  --benchmark-name toy \
  --output-dir outputs/general_toy \
  --model <MODEL_NAME>
```

### Agentic ML

Agentic ML expects:

- `data_dir/public/train.csv`
- `data_dir/public/test.csv`
- a private label file for evaluation
- a prompt template that tells the agent how to produce `submission.csv`

Toy speed test:

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

Full time-budgeted evaluation:

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

### Interactive Jericho

Interactive evaluation requires a local game file such as `zork1.z5`.

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

## RL Training

The RL code has separate dependencies and runtime services from Timely Eval.

```bash
cd rl/internbootcamp_v2
less README.md
```

Main RL entrypoint:

```text
rl/internbootcamp_v2/internbootcamp/bootcamps/Basic_LLM_timer
```

Typical launch order:

1. Start the task environment server: general timer, Agentic ML timer, or Jericho.
2. Start the distributed tool backend with `scripts/run_llm_timer_tool_server.sh`.
3. Start training with `scripts/run_llm_timer_rl_example.sh`.

Smoke-test status:

- ✅ General timer server: `/health`, `/register`, and `/call`
- ✅ Agentic ML server and `MLTimerTool`: code execution, `submission.csv` evaluation, and timing with external ML data
- ✅ Jericho server and tools: available actions, score, max score, `look`, and end-game with external ROM files
- ✅ One-step Qwen3-8B RL smoke on one H200 with actor/reference CPU offload

## Release Notes

This open-source release intentionally omits private or machine-specific artifacts:

- API keys, service-account files, internal IPs, and private endpoints
- experiment logs, model outputs, checkpoints, generated workspaces, private labels, large RL datasets, and internal cluster launch snapshots
- ML benchmark `data_sources`
- Jericho game ROM files

The examples in this repository are synthetic and intended for smoke tests only.

Agentic ML executes model-generated Python code as a subprocess in an isolated working directory. This is not a security sandbox. For untrusted code, run inside a container or VM with restricted filesystem and network access.

## Development

```bash
pip install -e ".[agentic,dev]"
PYTHONPATH=src pytest -q
```

Run a lightweight privacy scan before publishing:

```bash
rg -n "sk-|CREDENTIALS|BEGIN .*PRIVATE KEY|/mnt/|http://10\\.|http://100\\.|http://172\\.|http://192\\.168\\." .
```

## Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{timelyreasoner2026,
  title = {Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic},
  author = {Timely Machine Authors},
  year = {2026},
  note = {Code release}
}
```


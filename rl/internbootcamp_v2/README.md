# Timely RL Training Code

This directory contains the RL training-side code used for Timely Reasoner.
It is separate from the evaluation package in the repository root.

## Scope

- `internbootcamp/bootcamps/Basic_LLM_timer/`: Timely RL bootcamp, reward calculators, timer tools, Jericho tools, ML tools, preprocessing scripts, and training config.
- `internbootcamp/`: shared bootcamp utilities and tool-server code.
- `verl/verl/`: the local verl training framework code required by the RL pipeline.

The open-source copy intentionally omits checkpoints, generated outputs, workspaces, large datasets, game ROMs, historical experiment launch snapshots, and internal cluster scripts.

## Main Entrypoint

The Timely RL code is centered at:

```bash
internbootcamp/bootcamps/Basic_LLM_timer
```

The most relevant files are:

- `Basic_timer_reward_calculator.py`: reward logic for general reasoning with timer-aware outputs.
- `Basic_timer_tool.py`: elapsed-time tool for general reasoning tasks.
- `Jericho_timer_tool.py`: timer-aware interactive game tools.
- `MachineLearning_timer_tool.py`: timer-aware Agentic ML execution/evaluation tool.
- `configs/llm_timer_multiturn_w_tool_grpo.yaml`: base multiturn GRPO config.
- `configs/llm_timer_tool_config.yaml`: native tool definitions.
- `scripts/run_llm_timer_tool_server.sh`: backend tool-server launcher.
- `scripts/run_llm_timer_rl_example.sh`: editable launcher template.

## Setup

Install the package from this directory:

```bash
cd OpenSource/rl/internbootcamp_v2
pip install -e . --no-deps
pip install -e ./verl --no-deps
```

Optional task dependencies:

```bash
# Agentic ML tasks.
pip install -e ".[ml]"

# Interactive game tasks.
pip install -e ".[jericho]"
python -m spacy download en_core_web_sm
```

Install the actual training dependencies according to your CUDA, PyTorch, vLLM/SGLang, Ray, and verl environment. The original internal environment is not included.

## Running A Training Job

Prepare your own model and parquet datasets.

Start the task environment server required by your dataset:

```bash
cd OpenSource/rl/internbootcamp_v2

# General reasoning timer tasks.
python -m internbootcamp.bootcamps.Basic_LLM_timer.local_timer_server

# Agentic ML timer tasks.
# Requires user-provided ML_source/data_sources files.
python -m internbootcamp.bootcamps.Basic_LLM_timer.local_timer_sever_ml_task

# Interactive game tasks. Requires `jericho`, `func-timeout`, spaCy's
# en_core_web_sm model, plus game ROM files under
# internbootcamp/bootcamps/Basic_LLM_timer/jericho_game_sources/jericho-game-suite.
python -m internbootcamp.bootcamps.Basic_LLM_timer.local_jericho_server
```

Start only the environment server needed by the task type you are training.

Then start the distributed tool backend:

```bash
cd OpenSource/rl/internbootcamp_v2
PORT=18091 \
NUM_WORKERS=1 \
bash internbootcamp/bootcamps/Basic_LLM_timer/scripts/run_llm_timer_tool_server.sh
```

The server writes a runtime config next to the native tool config:

```text
internbootcamp/bootcamps/Basic_LLM_timer/configs/llm_timer_tool_config_with_server_urls.yaml
```

Then start RL training in another shell:

```bash
cd OpenSource/rl/internbootcamp_v2
MODEL_PATH=/path/to/base_model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
bash internbootcamp/bootcamps/Basic_LLM_timer/scripts/run_llm_timer_rl_example.sh
```

A small single-GPU smoke run for an 8B model may need CPU offload:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/base_model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
bash internbootcamp/bootcamps/Basic_LLM_timer/scripts/run_llm_timer_rl_example.sh \
  trainer.total_training_steps=1 \
  data.train_batch_size=1 \
  data.train_max_samples=1 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2
```

For a native-tool debug run without the backend server, set:

```bash
TOOL_CONFIG_PATH=internbootcamp/bootcamps/Basic_LLM_timer/configs/llm_timer_tool_config.yaml
```

For judge-model based rewards, configure an OpenAI-compatible endpoint:

```bash
export TIMELY_JUDGE_MODEL=your-judge-model
export TIMELY_JUDGE_BASE_URL=http://localhost:8000/v1
export TIMELY_JUDGE_API_KEY=EMPTY
```

## Data And Artifacts

The release does not include training data, private labels, checkpoints, generated model outputs, ML benchmark data, or Jericho game ROM files. The copied tree keeps prompt templates and Jericho descriptions, but omits `ML_source/data_sources` and `jericho_game_sources/jericho-game-suite`. Preprocessing scripts are included as references and should be run with user-provided raw data paths.

## Smoke-Test Status

The current open-source tree has been smoke-tested with external resources from the original internal tree:

- General timer server: `/health`, `/register`, and `/call` work.
- Agentic ML server and `MLTimerTool`: a leaf-classification submission was generated, evaluated, and timed successfully when `ML_source/data_sources` was supplied externally.
- Jericho server and tools: `zork1.z5` loaded from an external ROM path; available actions, score, max score, `look`, and end-game calls returned successfully after installing the Jericho dependencies and spaCy model.
- One-step RL smoke with Qwen3-8B completed on a single H200 with actor/ref CPU offload. Without offload, the same 8B smoke can OOM during optimizer update on one GPU.

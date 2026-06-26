#!/usr/bin/env bash
set -euo pipefail

# Example launcher for Timely RL training. Fill these paths for your machine.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/path/to/base_model}"
TRAIN_FILE="${TRAIN_FILE:-/path/to/train.parquet}"
VAL_FILE="${VAL_FILE:-/path/to/val.parquet}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-internbootcamp/bootcamps/Basic_LLM_timer/configs/llm_timer_tool_config_with_server_urls.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-timely_rl_qwen3_8b_example}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT_NAME}}"

LLM_TIMER_DIR="${PROJECT_DIR}/internbootcamp/bootcamps/Basic_LLM_timer/LLM-Timer"
CONFIG_DIR="${PROJECT_DIR}/internbootcamp/bootcamps/Basic_LLM_timer/configs"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl:${LLM_TIMER_DIR}:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/timely_rl_flashinfer}"
# Useful for import/config smoke tests on machines where CUDA is installed but no GPU is visible.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${PROJECT_DIR}"

if [[ ! -f "${TOOL_CONFIG_PATH}" ]]; then
  echo "Tool config not found: ${TOOL_CONFIG_PATH}" >&2
  echo "Start the tool backend first, or set TOOL_CONFIG_PATH to a native tool config." >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  --config-path "${CONFIG_DIR}" \
  --config-name llm_timer_multiturn_w_tool_grpo.yaml \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
  trainer.project_name="timely_rl" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"

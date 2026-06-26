#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
TOOLS_YAML_PATH="${TOOLS_YAML_PATH:-internbootcamp/bootcamps/Basic_LLM_timer/configs/llm_timer_tool_config.yaml}"
PORT="${PORT:-18091}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LOG_DIR="${LOG_DIR:-./logs/tool_server}"
TIMEOUT_PER_QUERY="${TIMEOUT_PER_QUERY:-600}"

LLM_TIMER_DIR="${PROJECT_DIR}/internbootcamp/bootcamps/Basic_LLM_timer/LLM-Timer"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl:${LLM_TIMER_DIR}:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${PROJECT_DIR}"

python3 -m internbootcamp.utils.tool_server.cli \
  --mode unified \
  --tools_yaml_path "${TOOLS_YAML_PATH}" \
  --port "${PORT}" \
  --num_workers "${NUM_WORKERS}" \
  --keep_running \
  --log_dir "${LOG_DIR}" \
  --timeout_per_query "${TIMEOUT_PER_QUERY}"

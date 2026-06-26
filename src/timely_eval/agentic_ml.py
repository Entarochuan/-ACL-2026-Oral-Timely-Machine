"""Agentic machine-learning evaluation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from timely_eval.agents import AsyncChatAgent, ChatModelConfig
from timely_eval.io_utils import append_jsonl, read_jsonl
from timely_eval.ml_metrics import evaluate_submission
from timely_eval.ml_sandbox import ExecutionResult, run_python_code_in_isolation
from timely_eval.parsing import extract_answer_or_tool_call, extract_python_code
from timely_eval.prompts import AGENTIC_ML_SYSTEM, AGENTIC_ML_TOOL_PROMPT
from timely_eval.timer import Timer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgenticMLConfig:
    benchmark_name: str
    data_dir: Path
    private_test_path: Path
    prompt_template: Path
    output_dir: Path
    solver: ChatModelConfig
    id_column: str | None = "id"
    is_binary: bool = False
    binary_label_column: str | None = None
    batch_size: int = 4
    workers: int = 4
    max_turns: int = 3
    tested_nums: int = 1
    execution_timeout: int = 180
    time_limit_multipliers: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    preserve_workspaces: bool = True


class AgenticMLEvaluator:
    """Evaluate agents that write and iteratively improve ML code."""

    def __init__(self, config: AgenticMLConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_prompt = Path(config.prompt_template).read_text(encoding="utf-8")
        self.solver_agent = AsyncChatAgent(
            config.solver,
            system_prompt=AGENTIC_ML_SYSTEM,
            role="solver",
            description="Agentic ML solver",
        )
        self.sem = asyncio.Semaphore(config.workers)

    @property
    def speed_summary_path(self) -> Path:
        return self.output_dir / (
            f"time_test_summary_{self.config.benchmark_name}_bsz_{self.config.batch_size}"
            f"_workers_{self.config.workers}.jsonl"
        )

    async def run_speed_test(self) -> dict[str, Any]:
        results: list[tuple[str, float, dict[str, Any] | None, dict[str, Any]]] = []
        pbar = tqdm(
            total=self.config.batch_size * self.config.tested_nums,
            desc=f"Agentic ML speed {self.config.benchmark_name}",
        )

        for _ in range(self.config.tested_nums):
            batch = [self._speed_test_one() for _ in range(self.config.batch_size)]
            for result in await asyncio.gather(*batch):
                results.append(result)
                pbar.update(1)
        pbar.close()

        total_time = 0.0
        total_accuracy = 0.0
        valid_count = 0
        for response, duration, code_result, evaluation in results:
            is_valid = bool(evaluation.get("is_valid_submission")) and "accuracy" in evaluation
            if is_valid:
                valid_count += 1
                total_time += duration
                total_accuracy += float(evaluation.get("accuracy", 0.0))
            append_jsonl(
                self.output_dir
                / (
                    f"time_test_w_tool_batch_{self.config.benchmark_name}_bsz_{self.config.batch_size}"
                    f"_workers_{self.config.workers}.jsonl"
                ),
                {
                    "benchmark_name": self.config.benchmark_name,
                    "model_response": response,
                    "total_time": duration,
                    "result": code_result,
                    "evaluation_result": evaluation,
                    "is_valid_for_avg": is_valid,
                },
            )

        summary = {
            "benchmark_name": self.config.benchmark_name,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.workers,
            "total_cases": len(results),
            "valid_case_count": valid_count,
            "total_time": total_time,
            "average_time": total_time / valid_count if valid_count else 0.0,
            "average_accuracy": total_accuracy / valid_count if valid_count else 0.0,
        }
        append_jsonl(self.speed_summary_path, summary)
        return summary

    async def run_full_eval(self, average_time: float | None = None) -> list[dict[str, Any]]:
        if average_time is None:
            if not self.speed_summary_path.exists():
                raise FileNotFoundError(f"Missing speed summary: {self.speed_summary_path}")
            summary = read_jsonl(self.speed_summary_path)[-1]
            average_time = float(summary.get("average_time") or 0.0)
        if average_time <= 0:
            raise ValueError("average_time must be positive. Run speed_test first or pass --average-time.")

        all_summaries: list[dict[str, Any]] = []
        for multiplier in self.config.time_limit_multipliers:
            tasks = [
                self._time_limited_one(multiplier, average_time)
                for _ in range(self.config.batch_size)
            ]
            results = await asyncio.gather(*tasks)
            valid = [item for item in results if item["best_accuracy"] != "None"]
            avg_acc = (
                sum(float(item["best_accuracy"]) for item in valid) / len(results)
                if results
                else 0.0
            )
            summary = {
                "benchmark_name": self.config.benchmark_name,
                "batch_size": self.config.batch_size,
                "workers": self.config.workers,
                "time_limit_multiplier": multiplier,
                "average_accuracy": avg_acc,
                "valid_results": len(valid),
            }
            all_summaries.append(summary)
            append_jsonl(
                self.output_dir
                / (
                    f"full_eval_batch_{self.config.benchmark_name}_bsz_{self.config.batch_size}"
                    f"_workers_{self.config.workers}_time_limit_prob_{multiplier}.jsonl"
                ),
                summary,
            )
        return all_summaries

    async def _speed_test_one(
        self,
    ) -> tuple[str, float, dict[str, Any] | None, dict[str, Any]]:
        timer = Timer(mode="eval")
        timer.start()
        messages = [
            {"role": "system", "content": AGENTIC_ML_SYSTEM + "\n" + AGENTIC_ML_TOOL_PROMPT},
            {"role": "user", "content": self.task_prompt},
        ]
        current_work_dir: str | None = None
        response = ""
        code_result: dict[str, Any] | None = None
        evaluation: dict[str, Any] = {}

        for turn in range(2):
            response = await self.solver_agent.generate_response(messages, sem=self.sem)
            code = self._extract_code_from_response(response)
            duration = float(timer.call(return_format="value"))
            if code is None:
                evaluation = {
                    "is_valid_submission": False,
                    "reason": "No Python code found in response.",
                    "current_duration": duration,
                    "accuracy": 0.0,
                }
                if turn == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": response},
                            {
                                "role": "user",
                                "content": (
                                    "No valid Python code was found. Please wrap code in "
                                    "```python ... ``` and write ./submission.csv."
                                ),
                            },
                        ]
                    )
                    continue
                return response, duration, code_result, evaluation

            execution = await self._execute_and_evaluate(code, current_work_dir)
            if current_work_dir is None:
                current_work_dir = execution["execution"].work_dir
            duration = float(timer.call(return_format="value"))
            code_result = _execution_to_dict(execution["execution"])
            evaluation = execution["evaluation"]
            evaluation["current_duration"] = duration
            if evaluation.get("is_valid_submission"):
                return response, duration, code_result, evaluation

            if turn == 0:
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                f"Execution/evaluation failed: {evaluation.get('reason', 'unknown')}.\n"
                                f"You have spent {duration:.2f} seconds. Try again."
                            ),
                        },
                    ]
                )

        return response, float(timer.call(return_format="value")), code_result, evaluation

    async def _time_limited_one(self, multiplier: float, average_time: float) -> dict[str, Any]:
        timer = Timer(mode="eval")
        timer.start()
        time_limit = multiplier * average_time
        messages = [
            {"role": "system", "content": AGENTIC_ML_SYSTEM + "\n" + AGENTIC_ML_TOOL_PROMPT},
            {
                "role": "user",
                "content": f"{self.task_prompt}\nPlease finish the task within {time_limit:.2f} seconds.",
            },
        ]
        current_work_dir: str | None = None
        collected: list[dict[str, Any]] = []

        for _ in range(self.config.max_turns):
            response = await self.solver_agent.generate_response(messages, sem=self.sem)
            code = self._extract_code_from_response(response)
            current_duration = float(timer.call(return_format="value"))
            if code is None:
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                "No valid Python code found. "
                                f"You have spent {current_duration:.2f} seconds."
                            ),
                        },
                    ]
                )
                if current_duration >= 1.1 * time_limit:
                    break
                continue

            execution = await self._execute_and_evaluate(code, current_work_dir)
            if current_work_dir is None:
                current_work_dir = execution["execution"].work_dir
            current_duration = float(timer.call(return_format="value"))
            evaluation = execution["evaluation"]
            evaluation["current_duration"] = current_duration
            collected.append(evaluation)

            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": (
                            f"Evaluation result: {evaluation}\n"
                            f"You have spent {current_duration:.2f} seconds."
                        ),
                    },
                ]
            )
            if current_duration >= 1.5 * time_limit:
                break

        valid = [
            item
            for item in collected
            if item.get("is_valid_submission") and item.get("current_duration", 0.0) <= 1.5 * time_limit
        ]
        best = max(valid, key=lambda item: float(item.get("accuracy", 0.0)), default=None)
        result = {
            "benchmark_name": self.config.benchmark_name,
            "duration": float(timer.call(return_format="value")),
            "messages": messages,
            "collected_eval_results": collected,
            "best_eval_result": best if best is not None else "None",
            "best_accuracy": best.get("accuracy") if best is not None else "None",
            "time_limit": time_limit,
            "time_limit_multiplier": multiplier,
        }
        append_jsonl(
            self.output_dir
            / (
                f"iterative_tool_call_item_w_time_limit_{self.config.benchmark_name}"
                f"_bsz_{self.config.batch_size}_workers_{self.config.workers}"
                f"_time_limit_prob_{multiplier}.jsonl"
            ),
            result,
        )
        return result

    async def _execute_and_evaluate(
        self,
        code: str,
        current_work_dir: str | None,
    ) -> dict[str, Any]:
        execution = await run_python_code_in_isolation(
            code=code,
            base_dir=str(self.output_dir / "workspace"),
            timeout=self.config.execution_timeout,
            input_dir=str(self.config.data_dir),
            preserve_workspace=self.config.preserve_workspaces,
            extra_env={"KAGGLE_DATA_DIR": str(self.config.data_dir)},
            saved_work_dir=current_work_dir,
        )
        if execution.returncode != 0:
            return {
                "execution": execution,
                "evaluation": {
                    "is_valid_submission": False,
                    "reason": f"Code execution failed with return code {execution.returncode}.",
                    "accuracy": 0.0,
                },
            }
        if not execution.submission_path:
            return {
                "execution": execution,
                "evaluation": {
                    "is_valid_submission": False,
                    "reason": "No submission.csv was produced.",
                    "accuracy": 0.0,
                },
            }
        try:
            evaluation = evaluate_submission(
                execution.submission_path,
                str(self.config.private_test_path),
                id_column=self.config.id_column,
                is_binary=self.config.is_binary,
                binary_label_column=self.config.binary_label_column,
            )
            evaluation["is_valid_submission"] = True
        except Exception as exc:  # noqa: BLE001
            evaluation = {
                "is_valid_submission": False,
                "reason": f"Submission evaluation failed: {exc}",
                "accuracy": 0.0,
            }
        return {"execution": execution, "evaluation": evaluation}

    @staticmethod
    def _extract_code_from_response(response: str) -> str | None:
        code = extract_python_code(response)
        if code is not None:
            return code
        has_tool_call, _has_answer, tool_call, _answer = extract_answer_or_tool_call(response)
        if has_tool_call and tool_call:
            try:
                import json

                parsed = json.loads(tool_call)
                return parsed.get("arguments", {}).get("code")
            except Exception:  # noqa: BLE001
                return None
        return None


def _execution_to_dict(result: ExecutionResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout": _truncate(result.stdout, 5000),
        "stderr": _truncate(result.stderr, 3000),
        "timeout": result.timeout,
        "work_dir": result.work_dir,
        "execution_time": result.execution_time,
        "submission_path": result.submission_path,
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]

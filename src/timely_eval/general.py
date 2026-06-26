"""General reasoning evaluation with time-aware tool use."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from timely_eval.agents import AsyncChatAgent, ChatModelConfig
from timely_eval.io_utils import append_jsonl, completed_ids, read_jsonl, write_json
from timely_eval.parsing import extract_answer_or_tool_call, extract_judge_result
from timely_eval.prompts import JUDGE_SYSTEM_PROMPT, TIME_AWARE_GENERAL_SYSTEM, TIME_TOOL_PROMPT
from timely_eval.scoring import compute_score
from timely_eval.timer import Timer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneralEvalConfig:
    data_path: Path
    output_dir: Path
    benchmark_name: str
    solver: ChatModelConfig
    judge: ChatModelConfig | None = None
    batch_size: int = 16
    workers: int = 16
    judge_workers: int | None = None
    time_limit_multipliers: list[float] = field(default_factory=lambda: [0.75, 1.0, 2.0, 3.0])
    max_turns: int = 100
    sample_limit: int | None = None
    resume: bool = True


class GeneralReasoningEvaluator:
    """Two-stage evaluator for static reasoning benchmarks.

    Stage 1 measures per-item baseline duration. Stage 2 gives each item a time
    budget equal to ``baseline_duration * multiplier`` and records correctness
    under that budget.
    """

    def __init__(self, config: GeneralEvalConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.solver_agent = AsyncChatAgent(
            config.solver,
            system_prompt=TIME_AWARE_GENERAL_SYSTEM,
            role="solver",
            description="General reasoning solver",
        )
        self.judge_agent = (
            AsyncChatAgent(
                config.judge,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                role="judge",
                description="General answer judge",
            )
            if config.judge is not None
            else None
        )
        self.solver_sem = asyncio.Semaphore(config.workers)
        self.judge_sem = asyncio.Semaphore(config.judge_workers or config.workers)
        self.data = self._load_data()

    @property
    def speed_result_path(self) -> Path:
        return self.output_dir / (
            f"speed_test_{self.config.benchmark_name}_bsz_{self.config.batch_size}"
            f"_workers_{self.config.workers}.jsonl"
        )

    @property
    def timed_result_path(self) -> Path:
        probs = "_".join(str(prob) for prob in self.config.time_limit_multipliers)
        return self.output_dir / (
            f"time_test_w_tool_{self.config.benchmark_name}_probs_{probs}"
            f"_bsz_{self.config.batch_size}_workers_{self.config.workers}.jsonl"
        )

    def _load_data(self) -> list[dict[str, Any]]:
        items = read_jsonl(self.config.data_path)
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if "question" not in item or "answer" not in item:
                raise ValueError("Each JSONL item must contain 'question' and 'answer'.")
            normalized.append(
                {
                    **item,
                    "id": str(item.get("id", item.get("ID", index))),
                    "question": str(item["question"]),
                    "answer": str(item["answer"]),
                }
            )
        if self.config.sample_limit is not None:
            normalized = normalized[: self.config.sample_limit]
        return normalized

    async def run_speed_test(self) -> float:
        items = self._filter_resume(self.data, self.speed_result_path)
        if not items:
            logger.info("No remaining items for speed test.")
            return 0.0

        durations: list[float] = []
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        pbar = tqdm(total=len(items), desc=f"Speed test {self.config.benchmark_name}")

        async def worker() -> None:
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    result = await self._speed_test_item(item)
                    durations.append(float(result["total_time"]))
                    append_jsonl(self.speed_result_path, result)
                finally:
                    queue.task_done()
                    pbar.update(1)

        await asyncio.gather(*(worker() for _ in range(self.config.workers)))
        pbar.close()
        return sum(durations) / len(durations) if durations else 0.0

    async def run_time_limited_eval(self) -> dict[str, Any]:
        speed_results = self._load_speed_results()
        items = [item for item in self.data if item["id"] in speed_results]
        items = self._filter_resume(items, self.timed_result_path)
        if not items:
            logger.info("No remaining items for time-limited evaluation.")
            return await self.analyze_results()

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        pbar = tqdm(total=len(items), desc=f"Timed eval {self.config.benchmark_name}")

        async def worker() -> None:
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    result = await self._time_limited_item(item, speed_results[item["id"]])
                    append_jsonl(self.timed_result_path, result)
                finally:
                    queue.task_done()
                    pbar.update(1)

        await asyncio.gather(*(worker() for _ in range(self.config.workers)))
        pbar.close()
        return await self.analyze_results()

    async def analyze_results(self) -> dict[str, Any]:
        if not self.speed_result_path.exists():
            raise FileNotFoundError(f"Missing speed-test results: {self.speed_result_path}")
        if not self.timed_result_path.exists():
            raise FileNotFoundError(f"Missing time-limited results: {self.timed_result_path}")

        speed_by_id = {str(item["ID"]): item for item in read_jsonl(self.speed_result_path)}
        timed_by_id = {str(item["ID"]): item for item in read_jsonl(self.timed_result_path)}
        common_ids = [item_id for item_id in timed_by_id if item_id in speed_by_id]
        total = len(common_ids)
        if total == 0:
            raise ValueError("No overlapping items between speed and timed results.")

        original_correct = sum(bool(speed_by_id[item_id].get("is_correct")) for item_id in common_ids)
        multiplier_correct: dict[str, int] = {str(prob): 0 for prob in self.config.time_limit_multipliers}
        multiplier_within: dict[str, int] = {str(prob): 0 for prob in self.config.time_limit_multipliers}

        for item_id in common_ids:
            infos = timed_by_id[item_id].get("time_limit_eval_infos", {})
            for prob in self.config.time_limit_multipliers:
                prob_key = str(prob)
                info = infos.get(prob_key, {})
                if info.get("within_time_limit"):
                    multiplier_within[prob_key] += 1
                if info.get("is_correct") and info.get("within_time_limit"):
                    multiplier_correct[prob_key] += 1

        summary = {
            "benchmark_name": self.config.benchmark_name,
            "total_samples": total,
            "original_correct": original_correct,
            "original_accuracy": original_correct / total,
            "time_limit_results": {
                prob_key: {
                    "correct": multiplier_correct[prob_key],
                    "within_time_limit": multiplier_within[prob_key],
                    "accuracy": multiplier_correct[prob_key] / total,
                }
                for prob_key in multiplier_correct
            },
        }
        write_json(self.output_dir / f"summary_{self.config.benchmark_name}.json", summary)
        return summary

    async def _speed_test_item(self, item: dict[str, Any]) -> dict[str, Any]:
        timer = Timer(mode="eval")
        timer.start()
        messages = [
            {"role": "system", "content": TIME_AWARE_GENERAL_SYSTEM + "\n" + TIME_TOOL_PROMPT},
            {"role": "user", "content": item["question"]},
        ]

        response = ""
        final_answer: str | None = None
        for _ in range(self.config.max_turns):
            response = await self.solver_agent.generate_response(messages, sem=self.solver_sem)
            has_tool_call, has_answer, _tool_call, answer = extract_answer_or_tool_call(response)
            messages.append({"role": "assistant", "content": response})
            if has_answer:
                final_answer = answer
                break
            if has_tool_call:
                messages.append({"role": "user", "content": f"<tool_response>{timer.call()}</tool_response>"})
                continue
            break

        total_time = float(timer.call(return_format="value"))
        score = compute_score(final_answer, item["answer"])
        is_correct, judge_response = await self._judge_if_needed(item, final_answer, score)
        return {
            "question": item["question"],
            "messages": messages,
            "final_answer": final_answer,
            "total_time": total_time,
            "is_correct": is_correct,
            "answer_score": score,
            "judge_response": judge_response,
            "original_answer": item["answer"],
            "ID": item["id"],
        }

    async def _time_limited_item(
        self,
        item: dict[str, Any],
        speed_result: dict[str, Any],
    ) -> dict[str, Any]:
        save_result = {
            "question": item["question"],
            "answer": item["answer"],
            "ID": item["id"],
            "time_limit_eval_infos": {},
        }
        baseline_duration = float(speed_result["total_time"])
        for multiplier in self.config.time_limit_multipliers:
            info = await self._run_item_with_time_limit(item, baseline_duration * multiplier)
            save_result["time_limit_eval_infos"][str(multiplier)] = info
        return save_result

    async def _run_item_with_time_limit(
        self,
        item: dict[str, Any],
        time_limit: float,
    ) -> dict[str, Any]:
        timer = Timer(mode="eval")
        timer.start()
        messages = [
            {"role": "system", "content": TIME_AWARE_GENERAL_SYSTEM + "\n" + TIME_TOOL_PROMPT},
            {
                "role": "user",
                "content": f"The question is: {item['question']}\nPlease answer within {time_limit:.2f} seconds.",
            },
        ]

        final_answer: str | None = None
        saw_tool_call = False
        for _ in range(self.config.max_turns):
            response = await self.solver_agent.generate_response(messages, sem=self.solver_sem)
            has_tool_call, has_answer, _tool_call, answer = extract_answer_or_tool_call(response)
            messages.append({"role": "assistant", "content": response})
            if has_answer:
                final_answer = answer
                break
            if has_tool_call:
                saw_tool_call = True
                messages.append({"role": "user", "content": f"<tool_response>{timer.call()}</tool_response>"})
                if float(timer.call(return_format="value")) >= 1.1 * time_limit:
                    break
                continue
            break

        duration = float(timer.call(return_format="value"))
        within_time_limit = duration <= 1.1 * time_limit
        score = compute_score(final_answer, item["answer"])
        is_correct, judge_response = await self._judge_if_needed(item, final_answer, score)
        return {
            "duration": duration,
            "time_limit": time_limit,
            "is_correct": bool(is_correct),
            "answer_score": score,
            "judge_response": judge_response,
            "messages": messages,
            "within_time_limit": within_time_limit,
            "whole_trace_has_tool_call": saw_tool_call,
        }

    async def _judge_if_needed(
        self,
        item: dict[str, Any],
        final_answer: str | None,
        score: float,
    ) -> tuple[bool, str | None]:
        if score == 1.0:
            return True, "math_reward_exact_match"
        if self.judge_agent is None or final_answer is None:
            return False, None

        judge_messages = [
            {"role": "system", "content": self.judge_agent.system_prompt},
            {
                "role": "user",
                "content": (
                    f"The question is: {item['question']}\n"
                    f"The model generated answer is: {final_answer}\n"
                    f"The reference answer is: {item['answer']}\n"
                    "Please judge whether the model response is correct."
                ),
            },
        ]
        response = await self.judge_agent.generate_response(judge_messages, sem=self.judge_sem)
        return extract_judge_result(response) == "yes", response

    def _load_speed_results(self) -> dict[str, dict[str, Any]]:
        if not self.speed_result_path.exists():
            raise FileNotFoundError(
                f"Speed-test results not found: {self.speed_result_path}. Run mode=speed_test first."
            )
        return {str(item["ID"]): item for item in read_jsonl(self.speed_result_path)}

    def _filter_resume(
        self,
        items: list[dict[str, Any]],
        result_path: Path,
    ) -> list[dict[str, Any]]:
        if not self.config.resume:
            return items
        done = completed_ids(result_path)
        return [item for item in items if str(item["id"]) not in done]

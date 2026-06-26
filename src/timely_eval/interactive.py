"""Optional Jericho text-game evaluation."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from timely_eval.agents import AsyncChatAgent, ChatModelConfig
from timely_eval.io_utils import append_jsonl, write_json
from timely_eval.parsing import extract_tool_calls
from timely_eval.prompts import INTERACTIVE_SYSTEM, INTERACTIVE_TOOL_PROMPT
from timely_eval.timer import Timer

logger = logging.getLogger(__name__)

DEFAULT_TOOL_DURATIONS = {
    "step": 1.0,
    "get_available_actions": 0.2,
    "get_score": 0.1,
    "get_max_score": 0.1,
    "end_game": 0.1,
}


@dataclass(slots=True)
class InteractiveEvalConfig:
    game_path: Path
    output_dir: Path
    solver: ChatModelConfig
    batch_size: int = 8
    workers: int = 6
    max_test_steps: int = 32
    eval_max_steps: list[int] = field(default_factory=lambda: [10, 20, 30, 50, 100])
    tool_durations: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOOL_DURATIONS))


class JerichoToolEnvironment:
    """Thin tool wrapper around ``jericho.FrotzEnv``."""

    def __init__(self, game_path: str | Path) -> None:
        try:
            from jericho import FrotzEnv
        except ImportError as exc:
            raise ImportError(
                "Jericho is required for interactive evaluation. Install with `pip install -e '.[interactive]'`."
            ) from exc

        self.env = FrotzEnv(str(game_path))
        initial_state = self.env.get_state()
        self.state_history: list[Any] = [initial_state]
        self.current_index = 0

    def get_valid_actions(self) -> list[str]:
        return self.env.get_valid_actions()

    def get_current_state(self) -> Any:
        return self.state_history[self.current_index]

    def get_score(self) -> str:
        return f"Your current score is: {self.env.get_score()}."

    def get_max_score(self) -> str:
        return f"The max score of the game is {self.env.get_max_score()}."

    def step(self, action: str) -> str:
        text, reward, done, info = self.env.step(action)
        new_state = self.env.get_state()
        if self.current_index < len(self.state_history) - 1:
            self.state_history = self.state_history[: self.current_index + 1]
        self.state_history.append(
            {
                "textual_response": text,
                "immediate_reward": reward,
                "done": done,
                "info": info,
                "state": new_state,
            }
        )
        self.current_index += 1
        suffix = (
            f"The game is terminated. The final score is: {self.env.get_score()}."
            if done
            else "The game is not terminated."
        )
        return f"The response is: {text}\nThe step reward is: {reward}.\n{suffix}"

    def end_game(self) -> str:
        return self.get_score()

    def check_game_termination(self) -> str:
        if self.env.victory():
            return "The game is terminated. You win the game."
        if self.env.game_over():
            return "The game is terminated. You lose the game."
        return "The game is not terminated. You can continue to play the game."


class InteractiveEvaluator:
    """Evaluate agents in Jericho text games."""

    def __init__(self, config: InteractiveEvalConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.solver_agent = AsyncChatAgent(
            config.solver,
            system_prompt=INTERACTIVE_SYSTEM,
            role="solver",
            description="Interactive game solver",
        )
        self.sem = asyncio.Semaphore(config.workers)

    async def run_speed_eval(self) -> dict[str, Any]:
        trajectories = await asyncio.gather(
            *(self._run_speed_game() for _ in range(self.config.batch_size)),
            return_exceptions=True,
        )
        valid = [item for item in trajectories if isinstance(item, dict) and item.get("total_steps", 0) > 0]
        summary = self._summarize_trajectories(trajectories, valid)
        write_json(self.output_dir / "speed_summary.json", summary)
        return summary

    async def run_full_eval(self, average_duration_per_step: float | None = None) -> list[dict[str, Any]]:
        if average_duration_per_step is None:
            import json

            summary_path = self.output_dir / "speed_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing speed summary: {summary_path}")
            average_duration_per_step = float(json.loads(summary_path.read_text())["average_duration_per_step"])

        summaries: list[dict[str, Any]] = []
        for max_steps in self.config.eval_max_steps:
            trajectories = await asyncio.gather(
                *(
                    self._run_timed_game(max_steps, average_duration_per_step)
                    for _ in range(self.config.batch_size)
                ),
                return_exceptions=True,
            )
            valid = [item for item in trajectories if isinstance(item, dict) and item.get("total_steps", 0) > 0]
            summary = self._summarize_trajectories(trajectories, valid)
            summary["eval_max_steps"] = max_steps
            write_json(self.output_dir / f"max_steps_{max_steps}_summary.json", summary)
            summaries.append(summary)
        return summaries

    async def _run_speed_game(self) -> dict[str, Any]:
        return await self._run_game(max_steps=self.config.max_test_steps, time_limit=None)

    async def _run_timed_game(self, max_steps: int, average_duration_per_step: float) -> dict[str, Any]:
        return await self._run_game(max_steps=max_steps, time_limit=max_steps * average_duration_per_step)

    async def _run_game(self, *, max_steps: int, time_limit: float | None) -> dict[str, Any]:
        timer = Timer(mode="static", speed_factor=1.0)
        timer.start()
        env = JerichoToolEnvironment(self.config.game_path)
        current_tool_duration = 0.0
        game_ended = False
        initial_obs = env.get_current_state()[-1]
        user_prompt = f"Game started. Initial observation:\n{initial_obs}"
        if time_limit is not None:
            user_prompt += f"\nThe time limit for this task is {time_limit:.2f} seconds."
        messages = [
            {"role": "system", "content": INTERACTIVE_SYSTEM + "\n\n" + INTERACTIVE_TOOL_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        trajectory: dict[str, Any] = {
            "env_name": str(self.config.game_path),
            "max_score": env.get_max_score(),
            "max_step": max_steps,
            "time_limit": time_limit,
            "steps": [],
            "final_score": 0,
            "success": False,
            "all_responses": [],
        }

        for step_idx in range(max_steps):
            response = await self.solver_agent.generate_response(messages, sem=self.sem)
            trajectory["all_responses"].append(response)
            if _has_conclusion(response):
                trajectory["success"] = True
                break

            tool_calls = extract_tool_calls(response)
            step_record = {"step": step_idx, "agent_response": response, "tool_calls": tool_calls, "tool_results": []}
            if not tool_calls:
                elapsed = float(timer.call(return_format="value")) + current_tool_duration
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": f"Error: no valid tool call found. You have played for {elapsed:.1f} seconds.",
                        },
                    ]
                )
                trajectory["steps"].append(step_record)
                if time_limit is not None and elapsed >= 1.001 * time_limit:
                    break
                continue

            tool_call = self._normalize_tool_call(tool_calls[0])
            result = await self._execute_tool(tool_call, env)
            tool_name = tool_call["name"]
            current_tool_duration += self.config.tool_durations.get(tool_name, 0.0)
            elapsed = float(timer.call(return_format="value")) + current_tool_duration
            tool_result = {
                "tool": tool_name,
                "arguments": tool_call.get("arguments", {}),
                "result": f"<tool_response>{result}\nYou have played for {elapsed:.1f} seconds.</tool_response>",
                "duration": self.config.tool_durations.get(tool_name, 0.0),
                "accumulated_time_duration": elapsed,
            }
            step_record["tool_results"] = [tool_result]
            step_record["cumulative_actural_time_duration"] = elapsed
            trajectory["steps"].append(step_record)
            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": tool_result["result"]},
                ]
            )
            if tool_name == "end_game":
                game_ended = True
            termination_status = env.check_game_termination()
            if "The game is terminated" in termination_status or game_ended:
                trajectory["success"] = "win" in termination_status.lower()
                break
            if time_limit is not None and elapsed >= 1.001 * time_limit:
                break

        final_score = _first_int(env.get_score())
        total_actual = float(timer.call(return_format="value")) + current_tool_duration
        trajectory.update(
            {
                "final_score": final_score,
                "total_steps": len(trajectory["steps"]),
                "total_tool_duration": current_tool_duration,
                "total_actural_time_duration": total_actual,
                "average_tool_duration": current_tool_duration / len(trajectory["steps"])
                if trajectory["steps"]
                else 0.0,
                "average_actural_time_duration": total_actual / len(trajectory["steps"])
                if trajectory["steps"]
                else 0.0,
            }
        )
        append_jsonl(self.output_dir / f"trajectories_max_steps_{max_steps}.jsonl", trajectory)
        return trajectory

    async def _execute_tool(self, tool_call: dict[str, Any], env: JerichoToolEnvironment) -> str:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        if name == "step":
            action = arguments.get("action", "")
            return env.step(action) if action else "Error: 'action' is required."
        if name == "get_available_actions":
            return f"Available actions: {env.get_valid_actions()}"
        if name == "get_score":
            return env.get_score()
        if name == "get_max_score":
            return env.get_max_score()
        if name == "end_game":
            return env.end_game()
        return f"Error: Unknown tool '{name}'."

    @staticmethod
    def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        name = str(tool_call.get("name", ""))
        if name in DEFAULT_TOOL_DURATIONS:
            tool_call.setdefault("arguments", {})
            return tool_call
        return {"name": "step", "arguments": {"action": name}}

    @staticmethod
    def _summarize_trajectories(
        trajectories: list[dict[str, Any] | BaseException],
        valid: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dict_trajectories = [item for item in trajectories if isinstance(item, dict)]
        total = len(trajectories)
        total_steps = sum(int(item.get("total_steps", 0)) for item in dict_trajectories)
        total_actual = sum(float(item.get("total_actural_time_duration", 0.0)) for item in valid)
        return {
            "total_games": total,
            "failed_games": sum(not isinstance(item, dict) for item in trajectories),
            "successful_games": len(valid),
            "total_success": sum(bool(item.get("success")) for item in dict_trajectories),
            "total_steps": total_steps,
            "average_score": sum(float(item.get("final_score", 0)) for item in dict_trajectories) / total
            if total
            else 0.0,
            "average_duration_per_step": total_actual / total_steps if total_steps else 0.0,
            "average_actural_time_duration": total_actual / len(valid) if valid else 0.0,
        }


def _has_conclusion(text: str) -> bool:
    return bool(re.search(r"<conclusion>\s*(.*?)\s*</conclusion>", text or "", re.DOTALL))


def _first_int(text: str) -> int:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else 0

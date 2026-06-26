"""Command-line entry point for Timely Eval."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from timely_eval.agentic_ml import AgenticMLConfig, AgenticMLEvaluator
from timely_eval.agents import ChatModelConfig
from timely_eval.general import GeneralEvalConfig, GeneralReasoningEvaluator
from timely_eval.interactive import InteractiveEvalConfig, InteractiveEvaluator


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    if args.command == "general":
        result = asyncio.run(_run_general(args))
    elif args.command == "agentic-ml":
        result = asyncio.run(_run_agentic_ml(args))
    elif args.command == "interactive":
        result = asyncio.run(_run_interactive(args))
    else:
        parser.error("A subcommand is required.")
        return

    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timely-eval",
        description="Evaluation framework for time-aware test-time scaling agents.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command")

    general = subparsers.add_parser("general", help="Run general reasoning evaluation.")
    _add_model_args(general)
    _add_judge_args(general)
    general.add_argument("--mode", choices=["speed_test", "time_test_w_tool", "result_analysis"], required=True)
    general.add_argument("--data-path", type=Path, required=True)
    general.add_argument("--benchmark-name", default="custom")
    general.add_argument("--output-dir", type=Path, default=Path("outputs/general"))
    general.add_argument("--batch-size", type=int, default=16)
    general.add_argument("--workers", type=int, default=16)
    general.add_argument("--judge-workers", type=int, default=None)
    general.add_argument("--time-limit-probs", type=float, nargs="+", default=[0.75, 1.0, 2.0, 3.0])
    general.add_argument("--max-turns", type=int, default=100)
    general.add_argument("--sample-limit", type=int, default=None)
    general.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    agentic = subparsers.add_parser("agentic-ml", help="Run agentic ML evaluation.")
    _add_model_args(agentic)
    agentic.add_argument("--mode", choices=["speed_test", "full_eval"], required=True)
    agentic.add_argument("--benchmark-name", default="custom-ml")
    agentic.add_argument("--data-dir", type=Path, required=True)
    agentic.add_argument("--private-test-path", type=Path, required=True)
    agentic.add_argument("--prompt-template", type=Path, required=True)
    agentic.add_argument("--output-dir", type=Path, default=Path("outputs/agentic_ml"))
    agentic.add_argument("--id-column", default="id")
    agentic.add_argument("--id-column-none", action="store_true", help="Use row order instead of an id column.")
    agentic.add_argument("--is-binary", action="store_true")
    agentic.add_argument("--binary-label-column", default=None)
    agentic.add_argument("--batch-size", type=int, default=4)
    agentic.add_argument("--workers", type=int, default=4)
    agentic.add_argument("--max-turns", type=int, default=3)
    agentic.add_argument("--tested-nums", type=int, default=1)
    agentic.add_argument("--execution-timeout", type=int, default=180)
    agentic.add_argument("--time-limit-probs", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    agentic.add_argument("--average-time", type=float, default=None)
    agentic.add_argument("--preserve-workspaces", action=argparse.BooleanOptionalAction, default=True)

    interactive = subparsers.add_parser("interactive", help="Run optional Jericho text-game evaluation.")
    _add_model_args(interactive)
    interactive.add_argument("--mode", choices=["speed_eval", "full_eval"], required=True)
    interactive.add_argument("--game-path", type=Path, required=True)
    interactive.add_argument("--output-dir", type=Path, default=Path("outputs/interactive"))
    interactive.add_argument("--batch-size", type=int, default=8)
    interactive.add_argument("--workers", type=int, default=6)
    interactive.add_argument("--max-test-steps", type=int, default=32)
    interactive.add_argument("--max-steps", type=int, nargs="+", default=[10, 20, 30, 50, 100])
    interactive.add_argument("--average-duration-per-step", type=float, default=None)

    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Solver model name served by the API endpoint.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=None, help="API key value. Prefer --api-key-env for shared scripts.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", default=None)


def _add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-model", default=None, help="Optional judge model for non-exact matches.")
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=0.1)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)


def _model_config_from_args(args: argparse.Namespace) -> ChatModelConfig:
    return ChatModelConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )


def _judge_config_from_args(args: argparse.Namespace) -> ChatModelConfig | None:
    if not args.judge_model:
        return None
    return ChatModelConfig(
        model=args.judge_model,
        api_base=args.judge_api_base or args.api_base,
        api_key=args.judge_api_key,
        api_key_env=args.judge_api_key_env,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )


async def _run_general(args: argparse.Namespace) -> dict:
    evaluator = GeneralReasoningEvaluator(
        GeneralEvalConfig(
            data_path=args.data_path,
            output_dir=args.output_dir,
            benchmark_name=args.benchmark_name,
            solver=_model_config_from_args(args),
            judge=_judge_config_from_args(args),
            batch_size=args.batch_size,
            workers=args.workers,
            judge_workers=args.judge_workers,
            time_limit_multipliers=args.time_limit_probs,
            max_turns=args.max_turns,
            sample_limit=args.sample_limit,
            resume=args.resume,
        )
    )
    if args.mode == "speed_test":
        return {"average_time": await evaluator.run_speed_test()}
    if args.mode == "time_test_w_tool":
        return await evaluator.run_time_limited_eval()
    return await evaluator.analyze_results()


async def _run_agentic_ml(args: argparse.Namespace) -> dict | list[dict]:
    evaluator = AgenticMLEvaluator(
        AgenticMLConfig(
            benchmark_name=args.benchmark_name,
            data_dir=args.data_dir,
            private_test_path=args.private_test_path,
            prompt_template=args.prompt_template,
            output_dir=args.output_dir,
            solver=_model_config_from_args(args),
            id_column=None if args.id_column_none else args.id_column,
            is_binary=args.is_binary,
            binary_label_column=args.binary_label_column,
            batch_size=args.batch_size,
            workers=args.workers,
            max_turns=args.max_turns,
            tested_nums=args.tested_nums,
            execution_timeout=args.execution_timeout,
            time_limit_multipliers=args.time_limit_probs,
            preserve_workspaces=args.preserve_workspaces,
        )
    )
    if args.mode == "speed_test":
        return await evaluator.run_speed_test()
    return await evaluator.run_full_eval(average_time=args.average_time)


async def _run_interactive(args: argparse.Namespace) -> dict | list[dict]:
    evaluator = InteractiveEvaluator(
        InteractiveEvalConfig(
            game_path=args.game_path,
            output_dir=args.output_dir,
            solver=_model_config_from_args(args),
            batch_size=args.batch_size,
            workers=args.workers,
            max_test_steps=args.max_test_steps,
            eval_max_steps=args.max_steps,
        )
    )
    if args.mode == "speed_eval":
        return await evaluator.run_speed_eval()
    return await evaluator.run_full_eval(average_duration_per_step=args.average_duration_per_step)


if __name__ == "__main__":
    main()

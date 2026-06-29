#!/usr/bin/env python3
"""Build a tiny Timely RL general-timer dataset for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = """Act as a Time-Aware Strategic Reasoner. Your objective is to solve the task within the given time window.

Output Requirements:
- Wrap your condensed final insights in <summary>...</summary> tags.
- Record your actual elapsed duration in <conclusion>total duration: {time} seconds</conclusion> tags.
- Wrap your final answer in <answer>...</answer> tags."""


QUESTIONS = [
    ("What is 17 + 25?", "42", 8.0, 1.0),
    ("A box has 3 red balls and 5 blue balls. How many balls are there in total?", "8", 8.0, 1.0),
    ("If a train travels 60 miles in 2 hours, what is its average speed in miles per hour?", "30", 10.0, 1.0),
    ("Compute 9 * 7.", "63", 8.0, 1.0),
    ("What is half of 144?", "72", 8.0, 1.0),
    ("A rectangle has length 6 and width 4. What is its area?", "24", 10.0, 1.0),
]


def make_row(index: int, question: str, answer: str, required_time: float, timer_speed_factor: float) -> dict:
    user_prompt = f"{question}\nPlease finish the task within {required_time:.2f} seconds."
    return {
        "data_source": "bootcamp/LlmTimer",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "ability": "timer",
        "reward_model": {
            "ground_truth": {
                "required_time": required_time,
                "timer_mode": "static",
                "timer_speed_factor": timer_speed_factor,
                "answer": answer,
                "question": question,
                "data_type": "general_reasoning",
                "with_time_limit": True,
            },
            "style": "rule",
        },
        "extra_info": {
            "index": index,
            "tools_kwargs": {
                "get_duration": {
                    "create_kwargs": {
                        "identity": {
                            "timer_mode": "static",
                            "timer_speed_factor": timer_speed_factor,
                        }
                    }
                }
            },
            "need_tools_kwargs": True,
        },
    }


def write_split(rows: list[dict], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(path / "train.parquet", index=False)
    dataframe.iloc[:2].to_parquet(path / "val.parquet", index=False)
    with (path / "train.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (path / "val.jsonl").open("w", encoding="utf-8") as file:
        for row in rows[:2]:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="examples/timer_toy_data", help="Directory for toy parquet/jsonl files.")
    args = parser.parse_args()

    rows = [make_row(index, *item) for index, item in enumerate(QUESTIONS)]
    write_split(rows, Path(args.output_dir))
    print(f"Wrote {len(rows)} train rows and 2 validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()

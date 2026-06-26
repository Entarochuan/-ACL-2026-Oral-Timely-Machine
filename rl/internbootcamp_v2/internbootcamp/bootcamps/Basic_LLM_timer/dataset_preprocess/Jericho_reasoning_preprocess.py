# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os
import json
import random
from tqdm import tqdm
import datasets
from datasets import Dataset
from datetime import datetime

from verl.utils.hdfs_io import copy, makedirs

training_split = [
    # ("game_905", "905.z5"),
    # ("acorncourt", "acorncourt.z5"), 
    ("advent", "advent.z5"),
    # ("adventureland", "adventureland.z5"),
    # ("afflicted", "afflicted.z8"),
    # ("anchor", "anchor.z8"),
    # ("awaken", "awaken.z5"),
    # ("balances", "balances.z5"),
    # ("ballyhoo", "ballyhoo.z3"),
    # ("curses", "curses.z5"),
    # ("cutthroat", "cutthroat.z3"),
    # ("deephome", "deephome.z5"),
    ("detective", "detective.z5"),
    # ("dragon", "dragon.z5"),
    ("enchanter", "enchanter.z3"), # ok
    # ("enter", "enter.z5"),
    # ("gold", "gold.z5"),
    # ("hhgg", "hhgg.z3"),
    # ("hollywood", "hollywood.z3"),
    # ("huntdark", "huntdark.z5"), # ok
    # ("infidel", "infidel.z3"),
    # ("inhumane", "inhumane.z5"), # ok
    # ("jewel", "jewel.z5"),
    # ("karn", "karn.z5"), # ok
    # ("lgop", "lgop.z3"),
    # ("library", "library.z5"), # ok
    # ("loose", "loose.z5"),
    # ("lostpig", "lostpig.z8"),
    # ("ludicorp", "ludicorp.z5"),
    # ("lurking", "lurking.z3"),
    # ("moonlit", "moonlit.z5"),
    # ("murdac", "murdac.z5"),
    # ("night", "night.z5"),
    # ("omniquest", "omniquest.z5"),
    # ("partyfoul", "partyfoul.z8"),
    # ("pentari", "pentari.z5"),
    # ("planetfall", "planetfall.z3"),
    # ("plundered", "plundered.z3"),
    # ("reverb", "reverb.z5"),
    # ("seastalker", "seastalker.z3"),
    # ("sherlock", "sherlock.z5"),
    # ("snacktime", "snacktime.z8"),
    # ("sorcerer", "sorcerer.z3"), # ok
    # # ("spellbrkr", "spellbrkr.z3"),
    # ("spirit", "spirit.z5"), # ok
    # ("temple", "temple.z5"),
    
    # ("theatre", "theatre.z5"),
    # ("trinity", "trinity.z4"),
    # ("tryst205", "tryst205.z5"),
    # ("weapon", "weapon.z5"),
    # ("wishbringer", "wishbringer.z3"),
    # ("yomomma", "yomomma.z8"),
    # ("zenon", "zenon.z5"),
    ("zork1", "zork1.z5"),
    # ("zork2", "zork2.z5"),
    # ("zork3", "zork3.z5"),
    # ("ztuu", "ztuu.z5"),
]

DESCRIPTION_DIR = "./jericho_game_sources/jericho_descriptions"

def load_game_description(game_file_name: str):
    """
    加载指定游戏的描述文件。

    参数:
        game_file_name: 原始游戏文件名，例如 'acorncourt.z5'

    返回:
        description: 初始观测文本（不含 max_score 行）
        max_score: int 类型的最大得分
    """
    path = os.path.join(DESCRIPTION_DIR, f"{game_file_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Description file not found for {game_file_name}: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty description file for {game_file_name}")

    # 第一行形如: "max_score: 350"
    first_line = lines[0].strip()
    if not first_line.startswith("max_score:"):
        raise ValueError(f"Invalid header line in {path}: {first_line}")

    try:
        max_score = int(first_line.split(":", 1)[1].strip())
    except Exception as e:
        raise ValueError(f"Failed to parse max_score in {path}: {e}")

    # 其余内容视为描述文本
    description = "".join(lines[1:]).lstrip("\n")
    return description, max_score

def load_from_jsonl(local_load_path, num_samples=None):
    loaded_dataset = []
    count = 0
    with open(local_load_path, "r", encoding="utf-8") as f:
        pbar = tqdm(f, desc="Loading dataset from JSONL file", total=num_samples)
        for line in pbar:
            if num_samples is not None and count >= num_samples:
                break
            try:
                example = json.loads(line)
                loaded_dataset.append(example)
                count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line at index {count}")
                continue
    return loaded_dataset


def make_map_fn(split, orig_prob:float=0.0):

    def process_fn(example, idx):

        game_name = example["game_name"]
        game_file_name = example["game_file_name"]
        game_description, max_score = load_game_description(game_name)
        sampled_duration = example["sampled_duration"]
        random_sample = random.random()

        if random.random() < 0.1:
            sampled_speed_factor = random.uniform(0.5, 1.0)
        if random_sample < 0.4:
            sampled_speed_factor = example["sampled_speed_factor"]
        elif random_sample < 0.6:
            changed_speed_factor = random.uniform(0.05, 0.5)
            sampled_speed_factor = example["sampled_speed_factor"] * changed_speed_factor
        else :
            if random.random() < 0.8:
                sampled_speed_factor = random.uniform(0.05, 0.1)
                sampled_duration = min(sampled_duration, random.uniform(50, 200))
            else : # 几乎忽视time
                sampled_speed_factor = random.uniform(0.005, 0.01)
                sampled_duration = min(sampled_duration, random.uniform(100, 400))
        
        if random.random() >= orig_prob:
            processed_example = dict(
                data_source="bootcamp/Jericho", # 这个是用来算reward的
                prompt=[
                    {
                        "role": "system",
                        "content": """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer complex challenges within a strictly enforced time window. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a reasoning path proves too time-consuming, decisively pivot to a more efficient heuristic or alternative method. \n4. **Convergence**: You must guarantee a complete conclusion before the deadline. A partial perfect derivation is a failure; a complete, logically sound conclusion is success. Do not try unavailable or dangerous actions. \n\nOnly select tools from the following list: step, get_available_actions, get_score, get_max_score, end_game.\n\nOutput Requirements:\n- Record your actual elapsed duration in the following format: \n<conclusion>total duration: {time(elapsed duration given by last tool call)} seconds</conclusion> \n- Summarize your final score in the following format: \n<score>{your_score(an integer)}</score>"""
                    }, 
                    {
                        "role": "user",
                        "content": game_description + f"\nPlease finish the task within {sampled_duration} seconds."
                    },
                ], 
                ability="timer", 
                reward_model={
                    "ground_truth": {
                        "required_time": sampled_duration, 
                        "timer_mode": "static",
                        "timer_speed_factor": sampled_speed_factor,
                        "max_score": max_score,
                        "data_type": "jericho",
                        "game_name": game_name,
                        "with_time_limit": True,
                    },
                    "style": "rule"
                },
                extra_info={
                    "max_assistant_turns": 300,
                    "tools_kwargs": {
                        "step": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "get_available_actions": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "get_score": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "get_max_score": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "end_game": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                    },
                    "need_tools_kwargs": True,
                }
            )
            return processed_example
        else : 
            processed_example = dict(
                data_source="bootcamp/Jericho", # 这个是用来算reward的
                prompt=[
                    {
                        "role": "system",
                        "content": """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer complex challenges. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a reasoning path proves too time-consuming, decisively pivot to a more efficient heuristic or alternative method. \n4. **Convergence**: You must guarantee a complete conclusion before the deadline. A partial perfect derivation is a failure; a complete, logically sound conclusion is success. Do not try unavailable or dangerous actions. Do not try unavailable or dangerous actions. \n\nOnly select tools from the following list: step, get_available_actions, get_score, get_max_score, end_game.\n\nOutput Requirements:\n- Record your actual elapsed duration in the following format: \n<conclusion>total duration: {time(elapsed duration given by last tool call)} seconds</conclusion> \n- Summarize your final score in the following format: \n<score>{your_score(an integer)}</score>"""
                    }, 
                    {
                        "role": "user",
                        "content": game_description
                    },
                ], 
                ability="timer", 
                reward_model={
                    "ground_truth": {
                        "required_time": sampled_duration, 
                        "timer_mode": "static",
                        "timer_speed_factor": sampled_speed_factor,
                        "max_score": max_score,
                        "data_type": "jericho",
                        "game_name": game_name,
                        "with_time_limit": False,
                    },
                    "style": "rule"
                },
                extra_info={
                    "max_assistant_turns": 300,
                    "tools_kwargs": {
                        "step": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "get_available_actions": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "get_score": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "get_max_score": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                        "end_game": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                    "env_name": game_file_name,
                                }
                            },
                        },
                    },
                    "need_tools_kwargs": True,
                }
            )
            return processed_example

    return process_fn

def sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor):

    random_sample = random.random()
    if random_sample < 0.1:
        actural_duration = random.uniform(args.min_duration, args.max_duration)
        sampled_speed_factor = random.uniform(args.min_timer_speed_factor, args.mid_timer_speed_factor)
        sampled_duration = int(actural_duration / sampled_speed_factor)
    elif random_sample < 0.95:
        sampled_duration = random.triangular(min_duration, max_duration, mode=mid_duration)
        if random.random() < 0.85:
            sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
        else :
            sampled_speed_factor = random.uniform(mid_timer_speed_factor, max_timer_speed_factor)
    else :
        sampled_duration = random.triangular(min_duration, mid_duration, mode=max_duration)
        sampled_speed_factor = random.uniform(0.01, 1.0)
        sampled_duration = int(sampled_duration / sampled_speed_factor)

    return sampled_duration, sampled_speed_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/mix_train/jericho_V6_1229_timelyRL", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)

    parser.add_argument("--min_duration", type=float, default=60.0, help="The minimum duration for the task.")
    parser.add_argument("--mid_duration", type=float, default=300.0, help="The middle duration for the task.")
    parser.add_argument("--max_duration", type=float, default=900.0, help="The maximum duration for the task.")
    parser.add_argument("--min_timer_speed_factor", type=float, default=0.75, help="The timer speed factor.")
    parser.add_argument("--mid_timer_speed_factor", type=float, default=1.0, help="The middle timer speed factor.")
    parser.add_argument("--max_timer_speed_factor", type=float, default=1.25, help="The maximum timer speed factor.")

    parser.add_argument("--orig_prob", type=float, default=0.1, help="The probability of the original task.(Without time limit.)")

    args = parser.parse_args()

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)

    # print(f"Loading data from {args.local_load_path}...")
    # raw_data_list = load_from_jsonl(args.local_load_path, args.num_samples)
    # print(f"Loaded {len(raw_data_list)} samples.")

    dataset_list = []
    print(f"Sampling {len(training_split)} games...")
    for game_name, game_file_name in training_split:
        for _ in range(1000):
            
            sampled_duration, sampled_speed_factor = sample_args_selection(args.min_duration, args.mid_duration, args.max_duration, args.min_timer_speed_factor, args.mid_timer_speed_factor, args.max_timer_speed_factor)
            dataset_list.append(dict(
                game_name=game_name,
                game_file_name=game_file_name,
                sampled_duration=int(sampled_duration),
                sampled_speed_factor=sampled_speed_factor,
            ))

    print(f"Shuffling {len(dataset_list)} samples...")
    random.shuffle(dataset_list)
    hf_dataset = Dataset.from_list(dataset_list)

    print("Processing dataset format...")
    remove_columns = hf_dataset.column_names
    
    # 在 map 时指定 remove_columns，这样就会删掉旧的 messages 等字段，只保留新生成的字段
    processed_dataset = hf_dataset.map(
        function=make_map_fn(split="train", orig_prob=args.orig_prob), 
        with_indices=True,
        remove_columns=remove_columns
    )

    # print(f"Splitting dataset with test_size={args.test_size}...")
    
    test_size_param = 0.1
    split_dataset = processed_dataset.train_test_split(test_size=test_size_param, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # === 保存部分 ===
    
    # 1. 保存为 Parquet
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join(local_save_dir, f"{time_stamp}")
    os.makedirs(save_dir, exist_ok=True)

    print("Saving to Parquet...")
    train_parquet_path = os.path.join(save_dir, f"train.parquet")
    test_parquet_path = os.path.join(save_dir, f"test.parquet")
    
    train_dataset.to_parquet(train_parquet_path)
    test_dataset.to_parquet(test_parquet_path)
    print(f"Saved Parquet files to {local_save_dir}")

    # 2. 保存为 JSONL (新增功能)
    print("Saving to JSONL...")
    train_jsonl_path = os.path.join(save_dir, f"train.jsonl")
    test_jsonl_path = os.path.join(save_dir, f"test.jsonl")
    
    # Dataset.to_json 默认就是 json lines 格式，force_ascii=False 保证中文正常显示
    train_dataset.to_json(train_jsonl_path, force_ascii=False)
    test_dataset.to_json(test_jsonl_path, force_ascii=False)
    print(f"Saved JSONL files to {local_save_dir}")

    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
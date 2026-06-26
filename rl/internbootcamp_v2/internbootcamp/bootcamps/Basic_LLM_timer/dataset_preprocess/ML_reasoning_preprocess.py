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


ML_Task_configs = dict(
    leaf_classification=dict(
        data_path="./ML_source/data_sources/leaf-classification/prepared",
        private_test_path="./ML_source/data_sources/leaf-classification/prepared/private/test.csv",
        benchmark_name="leaf-classification",
        work_dir="./work_dir/ml_test/leaf-classification",
        prompt_template="./ML_source/prompt_templates/leaf_classification.txt",
        id_column="id",
        is_binary=False,
        binary_label_column=None,
    ),
    random_acts_of_pizza=dict(
        data_path="./ML_source/data_sources/random-acts-of-pizza/prepared",
        private_test_path="./ML_source/data_sources/random-acts-of-pizza/prepared/private/test.csv",
        benchmark_name="random-acts-of-pizza",
        work_dir="./work_dir/ml_test/random-acts-of-pizza", 
        prompt_template="./ML_source/prompt_templates/random-acts-of-pizza.txt",
        id_column="request_id",
        is_binary=True,
        binary_label_column="requester_received_pizza",
    ),
    detecting_insults_in_social_commentary=dict(
        data_path="./ML_source/data_sources/detecting-insults-in-social-commentary/prepared",
        private_test_path="./ML_source/data_sources/detecting-insults-in-social-commentary/prepared/private/test.csv",
        benchmark_name="detecting-insults-in-social-commentary",
        work_dir="./work_dir/ml_test/detecting-insults-in-social-commentary",
        prompt_template="./ML_source/prompt_templates/detecting-insults-in-social-commentary.txt",
        id_column=None,
        is_binary=True,
        binary_label_column="Insult",
    ),
    spaceship_titanic=dict(
        data_path="./ML_source/data_sources/spaceship-titanic/prepared",
        private_test_path="./ML_source/data_sources/spaceship-titanic/prepared/private/test.csv",
        benchmark_name="spaceship-titanic",
        work_dir="./work_dir/ml_test/spaceship-titanic",
        prompt_template="./ML_source/prompt_templates/spaceship-titanic.txt",
        id_column="PassengerId",
        is_binary=True,
        binary_label_column="Transported",
    ),
)

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

def sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor):

    random_sample = random.random()
    if random_sample < 0.1:
        actural_duration = random.uniform(args.min_duration, args.max_duration)
        sampled_speed_factor = random.uniform(args.min_timer_speed_factor, args.mid_timer_speed_factor)
        sampled_duration = int(actural_duration / sampled_speed_factor)
    elif random_sample < 0.95:
        sampled_duration = random.triangular(min_duration, max_duration, mode=mid_duration)
        if random.random() < 0.25:
            sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
        else :
            sampled_speed_factor = random.uniform(mid_timer_speed_factor, max_timer_speed_factor)
    else :
        sampled_duration = random.triangular(min_duration, max_duration, mode=max_duration)
        sampled_speed_factor = random.uniform(0.5, 1.5)
        sampled_duration = int(sampled_duration / sampled_speed_factor)
    
    return sampled_duration, sampled_speed_factor

def make_map_fn(split, orig_prob:float=0.0):

    def process_fn(example, idx):
        
        # TASK congfigs
        task_configs = example["task_configs"]
        task_id = example["task_id"]
        data_path = task_configs.get("data_path")
        private_test_path = task_configs.get("private_test_path")
        benchmark_name = task_configs.get("benchmark_name")
        original_work_dir = task_configs.get("work_dir")
        work_dir = f"{original_work_dir}/{task_id}"
        prompt_template = task_configs.get("prompt_template")
        id_column = task_configs.get("id_column")
        is_binary = task_configs.get("is_binary")
        binary_label_column = task_configs.get("binary_label_column")
        with open(prompt_template, "r", encoding="utf-8") as f:
            task_description = f.read()

        # Time limit related configs ... 

        time_limit = example["sampled_duration"]
        timer_speed_factor = example["sampled_speed_factor"]

        random_choice = random.random() 
        if random_choice <= 0.1: # with time limit
            processed_example = dict(
                data_source="bootcamp/MachineLearning", # 这个是用来算reward的
                prompt=[
                    {
                        "role": "system",
                        "content": """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer machine learning reasoning tasks within a strictly enforced time window. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a solution proves too time-consuming, decisively pivot to a more efficient solution or alternative method. \n4. **Convergence**: You are encouraged to guarantee a complete conclusion before the deadline. A complete, logically sound conclusion is success.\n\nPlease fisrt wrap your code in ```python ```, and then call the tool to get execution results. \nOutput Requirements:\n- Record your actual elapsed duration in the following format: \n<conclusion>total duration: {time(elapsed duration given by last tool call)} seconds</conclusion> \n- Summarize the best accuracy you reached in the following format: \n<accuracy>{your_accuracy(a float between 0 and 1)}</accuracy>"""
                    }, 
                    {
                        "role": "user",
                        "content": task_description + f"\nPlease finish the task within {time_limit} seconds."
                    },
                ], 
                ability="timer", 
                reward_model={
                    "ground_truth": {
                        "required_time": time_limit, 
                        "timer_mode": "static",
                        "timer_speed_factor": timer_speed_factor,
                        "data_type": "MachineLearning",
                        "task_name": benchmark_name,
                        "with_time_limit": True,
                    },
                    "style": "rule"
                },
                extra_info={
                    "max_assistant_turns": 5,
                    "tools_kwargs": {
                        "execute_code_and_get_duration": {
                            "create_kwargs": {
                                "identity": {
                                    "data_path": data_path,
                                    "private_test_path": private_test_path,
                                    "benchmark_name": benchmark_name,
                                    "work_dir": work_dir,
                                    "prompt_template": prompt_template,
                                    "id_column": id_column,
                                    "is_binary": is_binary,
                                    "binary_label_column": binary_label_column,
                                    "task_id": task_id,
                                }
                            },
                        },
                    },
                    "need_tools_kwargs": True,
                }
            )
            return processed_example
        elif random.random() <= 1.0: # Single Round
            processed_example = dict(
                data_source="bootcamp/MachineLearning", # 这个是用来算reward的
                prompt=[
                    {
                        "role": "system",
                        "content": """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer machine learning reasoning tasks within a strictly enforced time window. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a solution proves too time-consuming, decisively pivot to a more efficient solution or alternative method. \n4. **Convergence**: You are encouraged to guarantee a complete conclusion before the deadline. A complete, logically sound conclusion is success.\n\nPlease fisrt wrap your code in ```python ```, and then call the tool to get execution results. \nOutput Requirements:\n- Record your actual elapsed duration in the following format: \n<conclusion>total duration: {time(elapsed duration given by last tool call)} seconds</conclusion> \n- Summarize the best accuracy you reached in the following format: \n<accuracy>{your_accuracy(a float between 0 and 1)}</accuracy>"""
                    }, 
                    {
                        "role": "user",
                        "content": task_description
                    },
                ], 
                ability="timer", 
                reward_model={
                    "ground_truth": {
                        "required_time": time_limit, 
                        "timer_mode": "static",
                        "timer_speed_factor": random.uniform(0.75, 1.25),
                        "data_type": "MachineLearning",
                        "task_name": benchmark_name,
                        "with_time_limit": True,
                        "single_round": True,
                    },
                    "style": "rule"
                },
                extra_info={
                    "max_assistant_turns": 5,
                    "tools_kwargs": {
                        "execute_code_and_get_duration": {
                            "create_kwargs": {
                                "identity": {
                                    "data_path": data_path,
                                    "private_test_path": private_test_path,
                                    "benchmark_name": benchmark_name,
                                    "work_dir": work_dir,
                                    "prompt_template": prompt_template,
                                    "id_column": id_column,
                                    "is_binary": is_binary,
                                    "binary_label_column": binary_label_column,
                                    "task_id": task_id,
                                }
                            },
                        },
                    },
                    "need_tools_kwargs": True,
                }
            )
            return processed_example


    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/mix_train/ml_reasoning_1229_origRL", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)

    # parser.add_argument("--time_limit", type=float, default=200.0, help="The time limit for the task.")
    parser.add_argument("--min_duration", type=float, default=50.0, help="The minimum duration for the task.")
    parser.add_argument("--mid_duration", type=float, default=100.0, help="The middle duration for the task.")
    parser.add_argument("--max_duration", type=float, default=200.0, help="The maximum duration for the task.")
    parser.add_argument("--min_timer_speed_factor", type=float, default=0.1, help="The minimum timer speed factor for the task.")
    parser.add_argument("--mid_timer_speed_factor", type=float, default=0.5, help="The middle timer speed factor for the task.")
    parser.add_argument("--max_timer_speed_factor", type=float, default=1.5, help="The maximum timer speed factor for the task.")

    args = parser.parse_args()

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)

    # print(f"Loading data from {args.local_load_path}...")
    # raw_data_list = load_from_jsonl(args.local_load_path, args.num_samples)
    # print(f"Loaded {len(raw_data_list)} samples.")

    dataset_list = []
    
    for task_name, task_configs in ML_Task_configs.items():
        task_id_counter = 0
        for _ in range(1000):
            sampled_duration, sampled_speed_factor = sample_args_selection(args.min_duration, args.mid_duration, args.max_duration, args.min_timer_speed_factor, args.mid_timer_speed_factor, args.max_timer_speed_factor)
            task_id = f"{task_name}_{task_id_counter}"
            dataset_list.append(dict(
                task_name=task_name,
                task_configs=task_configs,
                task_id=task_id,
                sampled_duration=sampled_duration,
                sampled_speed_factor=sampled_speed_factor,
            ))
            
            task_id_counter += 1

    random.shuffle(dataset_list)
    hf_dataset = Dataset.from_list(dataset_list)

    print("Processing dataset format...")
    remove_columns = hf_dataset.column_names
    
    # 在 map 时指定 remove_columns，这样就会删掉旧的 messages 等字段，只保留新生成的字段
    processed_dataset = hf_dataset.map(
        function=make_map_fn(split="train"), 
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
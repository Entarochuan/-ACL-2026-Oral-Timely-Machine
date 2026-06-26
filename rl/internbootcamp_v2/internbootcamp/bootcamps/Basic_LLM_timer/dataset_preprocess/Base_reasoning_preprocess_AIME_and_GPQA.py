# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# # Copyright 2023-2024 SGLang Team
# # Copyright 2025 ModelBest Inc. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Preprocess the DAPO-Math-17k dataset to multiturn format
# """

# import argparse
# import os
# import json
# import random
# from tqdm import tqdm
# import datasets
# from datasets import Dataset
# from datetime import datetime

# from verl.utils.hdfs_io import copy, makedirs

# def load_from_jsonl(local_load_path, num_samples=None):
#     loaded_dataset = []
#     count = 0
#     with open(local_load_path, "r", encoding="utf-8") as f:
#         pbar = tqdm(f, desc="Loading dataset from JSONL file", total=num_samples)
#         for line in pbar:
#             if num_samples is not None and count >= num_samples:
#                 break
#             try:
#                 example = json.loads(line)
#                 loaded_dataset.append(example)
#                 count += 1
#             except json.JSONDecodeError:
#                 print(f"Warning: Skipping invalid JSON line at index {count}")
#                 continue
#     return loaded_dataset

# def sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor):

#     # 一个采样更多元参数的函数，采样比例比较启发式设计。
    
#     random_prob = random.random()
#     if random_prob < 0.25 : # 25% 概率采样在 min 和 mid 之间
#         sampled_duration = random.triangular(min_duration, mid_duration, mid_duration)
#         sampled_duration = round(sampled_duration, 2) # 保留一位小数
#     elif random_prob < 0.75 : # 50% 概率采样在 mid 和 max 之间
#         sampled_duration = random.uniform(mid_duration, max_duration)
#         sampled_duration = round(sampled_duration, 2) # 保留一位小数
#     elif random_prob < 0.8 : # 15% 概率采样在 mid 和 max 之间
#         sampled_duration = random.triangular(mid_duration, max_duration, max_duration)
#         sampled_duration = round(sampled_duration, 2) 
#     elif random_prob < 0.95 : # 采样在 mid 和 max 之间(更大的time limit)
#         sampled_duration = random.triangular(min_duration, max_duration, max_duration)
#         sampled_duration = round(sampled_duration, 2) #
#     else :
#         sampled_duration = random.uniform(min_duration, max_duration)
#         sampled_duration = round(sampled_duration, 2) 

#     # 2. 采样 timer_speed_factor
#     # 速度因子通常使用均匀分布即可，如果也需要偏向性可同样用 triangular

#     if random.random() < 0.6 : # min 和 mid之间的速度
#         if random.random() < 0.8 :
#             if sampled_duration < 2.0 : # 太小的？鼓励少说！
#                 if random.random() < 0.15 :
#                     sampled_speed_factor = random.uniform(min_timer_speed_factor, 1.0)
#                 else :
#                     sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
#             elif sampled_duration >= mid_duration : # 比较长的，鼓励多说。
#                 if random.random() < 0.75 :
#                     sampled_speed_factor = random.uniform(min_timer_speed_factor, 1.0) # 鼓励多说(时间流逝速度更慢)
#                 else :
#                     sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
#             else :
#                 sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
#                 sampled_speed_factor = round(sampled_speed_factor, 2)
#         else :
#             sampled_speed_factor = 1.0 # 以一定概率还是维持原有速度
#     else : # 保留一定概率采样更大的速度因子
#         random_prob = random.random()
#         if random_prob < 0.2 :
#             sampled_speed_factor = random.triangular(mid_timer_speed_factor, max_timer_speed_factor, max_timer_speed_factor)
#             sampled_speed_factor = round(sampled_speed_factor, 2)
#         elif random_prob < 0.8 :
#             sampled_speed_factor = random.triangular(min_timer_speed_factor, max_timer_speed_factor, max_timer_speed_factor)
#             sampled_speed_factor = round(sampled_speed_factor, 2)
#         else :
#             sampled_speed_factor = random.uniform(min_timer_speed_factor, max_timer_speed_factor)
#             sampled_speed_factor = round(sampled_speed_factor, 2)

#     return sampled_duration, sampled_speed_factor

# def sample_args_selection_multiply(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor):

#     # 一个根据真实时间反推参数的采样函数
#     required_time = random.uniform(min_duration, mid_duration)
#     sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
#     sampled_duration = required_time * sampled_speed_factor

#     return sampled_duration, sampled_speed_factor


# def make_map_fn(split, min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor, orig_prob):

#     def process_fn(example, idx):
#         # 安全获取 user content
#         user_content = ""
#         if "messages" in example and len(example["messages"]) > 0:
#             user_content = example["messages"][0].get("content", "")
        
#         # 安全获取 reference_answer
#         ref_answer = ""
#         if "messages" in example and len(example["messages"]) > 0:
#             if "info" in example["messages"][0]:
#                 ref_answer = example["messages"][0]["info"].get("reference_answer", "")
#         if ref_answer == "" :
#             return None
        
#         # === 采样逻辑开始 ===
#         # 1. 随机取两个config
#         if random.random() >= orig_prob:
#             # 有时间限制
#             if random.random() < 0.7 :
#                 sampled_duration, sampled_speed_factor = sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
#             else :
#                 sampled_duration, sampled_speed_factor = sample_args_selection_multiply(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
            
#             processed_example = dict(
#                 data_source="bootcamp/LlmTimer", 
#                 prompt=[
#                     {
#                         "role": "system",
#                         "content": """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer complex challenges within a strictly enforced time window. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a reasoning path proves too time-consuming, decisively pivot to a more efficient heuristic or alternative method. \n4. **Convergence**: You must guarantee a complete conclusion before the deadline. A partial perfect derivation is a failure; a complete, logically sound conclusion is success.\n\nOutput Requirements:\n- Wrap your condensed final insights in <summary>...</summary> tags.\n- Record your actual elapsed duration in <conclusion>total duration: {time} seconds</conclusion> tags. Your final answer should be wrapped in <answer> and </answer> tags, for example, <answer>$\\boxed{100}$</answer>, <answer>\\boxed{B}</answer>, etc."""
#                     }, 
#                     {
#                         "role": "user",
#                         "content": user_content + f"\nPlease finish the task within {sampled_duration:.2f} seconds."
#                     },
#                 ], 
#                 ability="timer", 
#                 reward_model={
#                     "ground_truth": {
#                         "required_time": sampled_duration, 
#                         "timer_mode": "static",
#                         "timer_speed_factor": sampled_speed_factor,
#                         "answer": ref_answer, 
#                         "data_type": "general_reasoning", 
#                         "with_time_limit": True,
#                     },
#                     "style": "rule"
#                 },
#                 extra_info={
#                     "tools_kwargs": {
#                         "get_duration": {
#                             "create_kwargs": {
#                                 "identity": {
#                                     "timer_mode": "static",
#                                     "timer_speed_factor": sampled_speed_factor,
#                                 }
#                             },
#                         },
#                     },
#                     "need_tools_kwargs": True,
#                 }
#             )
#             if processed_example is not None:
#                 return processed_example

#         else :
#             if random.random() < 0.7 :
#                 sampled_duration, sampled_speed_factor = sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
#             else :
#                 sampled_duration, sampled_speed_factor = sample_args_selection_multiply(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
            
#             processed_example = dict(
#                 data_source="bootcamp/LlmTimer", 
#                 prompt=[
#                     {
#                         "role": "system",
#                         "content": """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer complex challenges within a strictly enforced time window. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a reasoning path proves too time-consuming, decisively pivot to a more efficient heuristic or alternative method. \n4. **Convergence**: You must guarantee a complete conclusion before the deadline. A partial perfect derivation is a failure; a complete, logically sound conclusion is success.\n\nOutput Requirements:\n- Wrap your condensed final insights in <summary>...</summary> tags.\n- Record your actual elapsed duration in <conclusion>total duration: {time} seconds</conclusion> tags. Your final answer should be wrapped in <answer> and </answer> tags, for example, <answer>$\\boxed{100}$</answer>, <answer>\\boxed{B}</answer>, etc."""
#                     }, 
#                     {
#                         "role": "user",
#                         "content": user_content
#                     },
#                 ], 
#                 ability="timer", 
#                 reward_model={
#                     "ground_truth": {
#                         "required_time": 1000, 
#                         "timer_mode": "static",
#                         "timer_speed_factor": sampled_speed_factor,
#                         "answer": ref_answer, 
#                         "data_type": "general_reasoning",
#                         "with_time_limit": False,
#                     },
#                     "style": "rule"
#                 },
#                 extra_info={
#                     "tools_kwargs": {
#                         "get_duration": {
#                             "create_kwargs": {
#                                 "identity": {
#                                     "timer_mode": "static",
#                                     "timer_speed_factor": sampled_speed_factor,
#                                 }
#                             },
#                         },
#                     },
#                     "need_tools_kwargs": True,
#                 }
#             )
#             if processed_example is not None:
#                 return processed_example            

#     return process_fn

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_dir", default="./data/mix_train/general_reasoning", help="The save directory for the preprocessed dataset.")
#     parser.add_argument("--hdfs_dir", default=None)
#     parser.add_argument("--local_load_path", default="/path/to/am_0.5M.jsonl", help="The local path to the raw dataset.")
#     parser.add_argument("--num_samples", type=int, default=30000, help="Number of samples to load. If None, load all.")
#     parser.add_argument("--test_size", type=int, default=100, help="Number of samples for test set.")

#     parser.add_argument("--min_duration", type=float, default=1.0, help="The minimum duration for the task.")
#     parser.add_argument("--mid_duration", type=float, default=10.0, help="The middle duration for the task.")
#     parser.add_argument("--max_duration", type=float, default=30.0, help="The maximum duration for the task.")
#     parser.add_argument("--min_timer_speed_factor", type=float, default=0.5, help="The timer speed factor.")
#     parser.add_argument("--mid_timer_speed_factor", type=float, default=2.0, help="The middle timer speed factor.")
#     parser.add_argument("--max_timer_speed_factor", type=float, default=10.0, help="The maximum timer speed factor.")

#     parser.add_argument("--orig_prob", type=float, default=0.1, help="The probability of the original task.(Without time limit.)")

#     args = parser.parse_args()

#     local_save_dir = args.local_dir
#     os.makedirs(local_save_dir, exist_ok=True)

#     print(f"Loading data from {args.local_load_path}...")
#     raw_data_list = load_from_jsonl(args.local_load_path, args.num_samples)
#     print(f"Loaded {len(raw_data_list)} samples.")

#     hf_dataset = Dataset.from_list(raw_data_list)

#     print("Processing dataset format...")
#     remove_columns = hf_dataset.column_names
    
#     # 在 map 时指定 remove_columns，这样就会删掉旧的 messages 等字段，只保留新生成的字段
#     processed_dataset = hf_dataset.map(
#         function=make_map_fn("train", args.min_duration, args.mid_duration, args.max_duration, args.min_timer_speed_factor, args.mid_timer_speed_factor, args.max_timer_speed_factor, args.orig_prob), 
#         with_indices=True,
#         remove_columns=remove_columns
#     )

#     print(f"Splitting dataset with test_size={args.test_size}...")
    
#     test_size_param = args.test_size
#     if test_size_param > 1 and isinstance(test_size_param, int):
#         if test_size_param >= len(processed_dataset):
#             print(f"Warning: test_size ({test_size_param}) is larger than total dataset ({len(processed_dataset)}). Using 10% for test.")
#             test_size_param = 0.1
    
#     split_dataset = processed_dataset.train_test_split(test_size=test_size_param, seed=42)
#     train_dataset = split_dataset['train']
#     test_dataset = split_dataset['test']

#     print(f"Train set size: {len(train_dataset)}")
#     print(f"Test set size: {len(test_dataset)}")

#     # === 保存部分 ===
    
#     # 1. 保存为 Parquet
#     time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     save_dir = os.path.join(local_save_dir, f"{time_stamp}")
#     os.makedirs(save_dir, exist_ok=True)

#     print("Saving to Parquet...")
#     train_parquet_path = os.path.join(save_dir, f"train.parquet")
#     test_parquet_path = os.path.join(save_dir, f"test.parquet")
    
#     train_dataset.to_parquet(train_parquet_path)
#     test_dataset.to_parquet(test_parquet_path)
#     print(f"Saved Parquet files to {local_save_dir}")

#     # 2. 保存为 JSONL (新增功能)
#     print("Saving to JSONL...")
#     train_jsonl_path = os.path.join(save_dir, f"train.jsonl")
#     test_jsonl_path = os.path.join(save_dir, f"test.jsonl")
    
#     # Dataset.to_json 默认就是 json lines 格式，force_ascii=False 保证中文正常显示
#     train_dataset.to_json(train_jsonl_path, force_ascii=False)
#     test_dataset.to_json(test_jsonl_path, force_ascii=False)
#     print(f"Saved JSONL files to {local_save_dir}")

#     if args.hdfs_dir is not None:
#         print(f"Copying to HDFS: {args.hdfs_dir}")
#         makedirs(args.hdfs_dir)
#         copy(src=local_save_dir, dst=args.hdfs_dir)

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
Preprocess the DAPO-Math-17k dataset to multiturn format (Fixed for filtering)
"""

import argparse
import os
import json
import random
from tqdm import tqdm
import datasets
from datasets import Dataset
from datetime import datetime

# 尝试导入 verl，如果环境没有配置可能需要注释掉或自行处理
try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    print("Warning: verl module not found. HDFS operations will be skipped.")
    def copy(src, dst): pass
    def makedirs(path): pass

def load_from_jsonl(local_load_path, num_samples=None):
    loaded_dataset = []
    count = 0
    if not os.path.exists(local_load_path):
        raise FileNotFoundError(f"File not found: {local_load_path}")
        
    with open(local_load_path, "r", encoding="utf-8") as f:
        # 先计算总行数用于进度条（可选，如果文件巨大可跳过）
        total_lines = sum(1 for _ in open(local_load_path, "r", encoding="utf-8")) if num_samples is None else num_samples
        f.seek(0)
        
        pbar = tqdm(f, desc="Loading dataset from JSONL file", total=total_lines)
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
    # 保持原有的启发式采样逻辑
    random_prob = random.random()
    if random_prob < 0.15 : 
        sampled_duration = random.triangular(min_duration, mid_duration, mid_duration)
    elif random_prob < 0.75 : 
        sampled_duration = random.uniform(mid_duration, max_duration)
    elif random_prob < 0.8 : 
        sampled_duration = random.triangular(mid_duration, max_duration, max_duration)
    elif random_prob < 0.95 : 
        sampled_duration = random.triangular(min_duration, max_duration, max_duration)
    else :
        sampled_duration = random.uniform(min_duration, max_duration)
    
    sampled_duration = round(sampled_duration, 2)

    # 2. 采样 timer_speed_factor
    if random.random() < 0.6 : 
        if random.random() < 0.8 :
            if sampled_duration < 2.0 : 
                if random.random() < 0.05 :
                    sampled_speed_factor = random.uniform(min_timer_speed_factor, 1.0)
                else :
                    sampled_speed_factor = random.uniform(min_timer_speed_factor, mid_timer_speed_factor)
            elif sampled_duration >= mid_duration : 
                if random.random() < 0.75 :
                    sampled_speed_factor = random.uniform(min_timer_speed_factor, max_timer_speed_factor) 
                else :
                    sampled_speed_factor = random.uniform(min_timer_speed_factor, max_timer_speed_factor)
            else :
                sampled_speed_factor = random.uniform(min_timer_speed_factor, max_timer_speed_factor)
        else :
            sampled_speed_factor = mid_timer_speed_factor 
    else : 
        random_prob = random.random()
        if random_prob < 0.2 :
            sampled_speed_factor = random.triangular(mid_timer_speed_factor, max_timer_speed_factor, max_timer_speed_factor)
        elif random_prob < 0.8 :
            sampled_speed_factor = random.triangular(min_timer_speed_factor, max_timer_speed_factor, max_timer_speed_factor)
        else :
            sampled_speed_factor = random.uniform(min_timer_speed_factor, max_timer_speed_factor)
    
    sampled_speed_factor = round(sampled_speed_factor, 2)
    return sampled_duration, sampled_speed_factor

def sample_args_selection_multiply(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor):
    required_time = random.uniform(min_duration, max_duration)
    sampled_speed_factor = random.uniform(min_timer_speed_factor, max_timer_speed_factor)
    sampled_duration = required_time * sampled_speed_factor
    return round(sampled_duration, 2), round(sampled_speed_factor, 2)

def make_map_fn(split, min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor, orig_prob):

    # 定义系统提示词
    SYSTEM_PROMPT = """Act as a Time-Aware Strategic Reasoner. Your objective is to conquer complex challenges within a strictly enforced time window. \n\nYou must treat Time as your most critical resource. \n1. **Initial Assessment**: Immediately gauge the problem's complexity against the remaining time. \n2. **Cognitive Budgeting**: Allocate specific time slots for analyzing, utilizing tools, and synthesizing the final answer. Do not over-invest in low-value details.\n3. **Dynamic Adjustment**: If a reasoning path proves too time-consuming, decisively pivot to a more efficient heuristic or alternative method. \n4. **Convergence**: You must guarantee a complete conclusion before the deadline. A partial perfect derivation is a failure; a complete, logically sound conclusion is success.\n\nOutput Requirements:\n- Wrap your condensed final insights in <summary>...</summary> tags.\n- Record your actual elapsed duration in <conclusion>total duration: {time} seconds</conclusion> tags. Your final answer should be wrapped in <answer> and </answer> tags, for example, <answer>$\\boxed{100}$</answer>, <answer>\\boxed{B}</answer>, etc."""

    def _process_single_item(example):
        """处理单个样本的内部逻辑"""
        # 安全获取 user content
        user_content = example['question']
        target_phrase = '\n\nRemember to put your answer on its own line after "Answer:".'

        # 执行替换
        user_content = user_content.replace(target_phrase, "")
                
        # 安全获取 reference_answer
        ref_answer = example['answer']
        # ref_answer = ""
        # if "messages" in example and len(example["messages"]) > 0:
        #     if "info" in example["messages"][0]:
        #         ref_answer = example["messages"][0]["info"].get("reference_answer", "")
        
        # 核心过滤：如果没有答案，返回 None
        if not ref_answer:
            return None
        
        # === 采样逻辑 ===
        if random.random() >= orig_prob:
            # Case 1: 有时间限制的任务
            if random.random() < 0.7 :
                sampled_duration, sampled_speed_factor = sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
            else :
                sampled_duration, sampled_speed_factor = sample_args_selection_multiply(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
            
            return dict(
                data_source="bootcamp/LlmTimer", 
                prompt=[
                    {"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": user_content + f"\nPlease finish the task within {sampled_duration:.2f} seconds."}
                ], 
                ability="timer", 
                reward_model={
                    "ground_truth": {
                        "required_time": sampled_duration, 
                        "timer_mode": "static",
                        "timer_speed_factor": sampled_speed_factor,
                        "answer": ref_answer, 
                        "question": example['question'],
                        "data_type": "general_reasoning", 
                        "with_time_limit": True,
                    },
                    "style": "rule"
                },
                extra_info={
                    "tools_kwargs": {
                        "get_duration": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                }
                            },
                        },
                    },
                    "need_tools_kwargs": True,
                }
            )

        else:
            # Case 2: 原始概率下的任务 (Mock time limit)
            if random.random() < 0.7 :
                sampled_duration, sampled_speed_factor = sample_args_selection(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
            else :
                sampled_duration, sampled_speed_factor = sample_args_selection_multiply(min_duration, mid_duration, max_duration, min_timer_speed_factor, mid_timer_speed_factor, max_timer_speed_factor)
            
            return dict(
                data_source="bootcamp/LlmTimer", 
                prompt=[
                    {"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": user_content}
                ], 
                ability="timer", 
                reward_model={
                    "ground_truth": {
                        "required_time": 1000, # 虚拟的大时间
                        "timer_mode": "static",
                        "timer_speed_factor": sampled_speed_factor,
                        "answer": ref_answer, 
                        "question": example['question'],
                        "data_type": "general_reasoning",
                        "with_time_limit": False,
                    },
                    "style": "rule"
                },
                extra_info={
                    "tools_kwargs": {
                        "get_duration": {
                            "create_kwargs": {
                                "identity": {
                                    "timer_mode": "static",
                                    "timer_speed_factor": sampled_speed_factor,
                                }
                            },
                        },
                    },
                    "need_tools_kwargs": True,
                }
            )

    def process_batch(batch):
        """
        Batch 处理函数。
        在这里我们可以安全地丢弃 _process_single_item 返回 None 的数据。
        """
        # 初始化输出字典
        output = {
            "data_source": [], 
            "prompt": [], 
            "ability": [], 
            "reward_model": [], 
            "extra_info": []
        }
        
        # 将列式 batch 转换为行式遍历
        # batch 是 {'col1': [val1, val2], 'col2': [val1, val2]}
        keys = batch.keys()
        # 获取 batch 大小
        batch_size = len(next(iter(batch.values())))
        
        for i in range(batch_size):
            # 重组单个 example
            example = {k: batch[k][i] for k in keys}
            
            # 处理单条数据
            result = _process_single_item(example)
            
            # 只有处理成功才加入输出
            if result is not None:
                for k, v in result.items():
                    output[k].append(v)
        
        return output

    return process_batch

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_dir", default="./processed_data", help="The save directory for the preprocessed dataset.")
#     parser.add_argument("--hdfs_dir", default=None)
#     parser.add_argument("--local_load_path", default="dapo_math_17k.jsonl", help="The local path to the raw dataset.")
#     parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to load. If None, load all.")
#     parser.add_argument("--test_size", type=int, default=100, help="Number of samples for test set.")

#     parser.add_argument("--min_duration", type=float, default=1.0)
#     parser.add_argument("--mid_duration", type=float, default=10.0)
#     parser.add_argument("--max_duration", type=float, default=30.0)
#     parser.add_argument("--min_timer_speed_factor", type=float, default=0.5)
#     parser.add_argument("--mid_timer_speed_factor", type=float, default=2.0)
#     parser.add_argument("--max_timer_speed_factor", type=float, default=10.0)

#     parser.add_argument("--orig_prob", type=float, default=0.1, help="Probability of tasks without explicit time limit.")

#     args = parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/mix_train/general_reasoning_TSET_withQ_1230_fasterlimit", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_load_path", default="/path/to/dapo_math_17k.jsonl", help="The local path to the raw dataset.")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples to load. If None, load all.")
    parser.add_argument("--test_size", type=int, default=100, help="Number of samples for test set.")

    parser.add_argument("--min_duration", type=float, default=5.0, help="The minimum duration for the task.")
    parser.add_argument("--mid_duration", type=float, default=10.0, help="The middle duration for the task.")
    parser.add_argument("--max_duration", type=float, default=20.0, help="The maximum duration for the task.")
    parser.add_argument("--min_timer_speed_factor", type=float, default=0.9, help="The timer speed factor.")
    parser.add_argument("--mid_timer_speed_factor", type=float, default=1.5, help="The middle timer speed factor.")
    parser.add_argument("--max_timer_speed_factor", type=float, default=3.0, help="The maximum timer speed factor.")

    parser.add_argument("--orig_prob", type=float, default=0.1, help="The probability of the original task.(Without time limit.)")

    args = parser.parse_args()

    # 1. 设置保存路径
    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)

    # 2. 加载数据
    print(f"Loading data from {args.local_load_path}...")

    local_data_paths = [
        "/path/to/processed_AIME.jsonl", 
        "/path/to/processed_GPQA_Diamond.jsonl"
    ]

    raw_data_list = []
    for local_data_path in local_data_paths:
        raw_data_list.extend(load_from_jsonl(local_data_path, args.num_samples))

    print(f"Loaded {len(raw_data_list)} samples.")

    hf_dataset = Dataset.from_list(raw_data_list)
    
    # 记录原始列名，用于 remove_columns
    remove_columns = hf_dataset.column_names

    print("Processing dataset (Batch mode)...")
    
    # 3. 执行 Map (开启 batched=True)
    # 这样可以在 process_batch 内部自动过滤掉返回 None 的数据
    processed_dataset = hf_dataset.map(
        function=make_map_fn("train", args.min_duration, args.mid_duration, args.max_duration, 
                             args.min_timer_speed_factor, args.mid_timer_speed_factor, args.max_timer_speed_factor, 
                             args.orig_prob), 
        batched=True,           # <--- 关键修改：开启 Batch 模式
        batch_size=1000,        # 每次处理1000条，提高效率
        remove_columns=remove_columns,
        desc="Mapping and Filtering"
    )

    print(f"Dataset size after filtering: {len(processed_dataset)}")

    # 4. 数据分割
    print(f"Splitting dataset with test_size={args.test_size}...")
    test_size_param = args.test_size
    
    # 防止 test_size 比总数据量大
    if len(processed_dataset) <= test_size_param:
        print(f"Warning: Dataset size ({len(processed_dataset)}) <= test_size ({test_size_param}). Using 10% for test.")
        test_size_param = 0.05
        
    split_dataset = processed_dataset.train_test_split(test_size=test_size_param, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # === 5. 保存数据 ===
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join(local_save_dir, f"{time_stamp}")
    os.makedirs(save_dir, exist_ok=True)

    print("Saving to Parquet...")
    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))

    print("Saving to JSONL...")
    train_dataset.to_json(os.path.join(save_dir, "train.jsonl"), force_ascii=False, orient="records", lines=True)
    test_dataset.to_json(os.path.join(save_dir, "test.jsonl"), force_ascii=False, orient="records", lines=True)
    
    print(f"All files saved to {save_dir}")

    # 6. 上传 HDFS (如果需要)
    if args.hdfs_dir is not None:
        try:
            print(f"Copying to HDFS: {args.hdfs_dir}")
            makedirs(args.hdfs_dir)
            copy(src=save_dir, dst=args.hdfs_dir)
            print("HDFS upload complete.")
        except Exception as e:
            print(f"HDFS upload failed: {e}")
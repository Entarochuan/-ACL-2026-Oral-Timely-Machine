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
from jericho import FrotzEnv

# ==== 基本路径配置 ====
# 所有 jericho 游戏文件所在目录（包含 .z3/.z4/.z5/.z8）
GAME_DIR = "./jericho_game_sources/jericho-game-suite"
# 描述文件输出目录（绝对路径，按用户指定）
DESCRIPTION_DIR = "./jericho_game_sources/jericho_descriptions"


def generate_initial_descriptions():
    """
    对目录下所有游戏文件：
    - 载入环境，获取初始观测和 max_score
    - 将结果保存为 txt，方便后续简单读取
    """
    os.makedirs(DESCRIPTION_DIR, exist_ok=True)

    # 收集所有游戏文件（按扩展名过滤）
    valid_exts = (".z3", ".z4", ".z5", ".z8")
    all_files = sorted(
        f for f in os.listdir(GAME_DIR)
        if f.lower().endswith(valid_exts)
    )

    print(f"Found {len(all_files)} game files.")

    for fname in all_files:
        game_path = os.path.join(GAME_DIR, fname)
        save_path = os.path.join(DESCRIPTION_DIR, f"{fname[:-3]}.txt")

        # 已存在则跳过，避免重复计算（可以按需修改）
        if os.path.exists(save_path):
            print(f"[Skip] {fname} already exists.")
            continue

        print(f"[Process] {fname}")
        env = None
        try:
            env = FrotzEnv(game_path)
            # 获取初始观测
            state = env.get_state()
            initial_obs = state[-1]

            # 获取 max_score（数字）
            max_score = env.get_max_score()

            # 写入 txt 文件
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"max_score: {max_score}\n")
                f.write("Game started. Initial observation:\n")
                f.write(str(initial_obs))

        except Exception as e:
            print(f"[Error] Failed processing {fname}: {e}")
        finally:
            if env is not None:
                env.close()


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


if __name__ == "__main__":
    # 跑一遍目录下所有游戏，生成对应的描述 txt
    # generate_initial_descriptions()
    env_name = "acorncourt"
    description, max_score = load_game_description(env_name)
    print(description)
    print(max_score)


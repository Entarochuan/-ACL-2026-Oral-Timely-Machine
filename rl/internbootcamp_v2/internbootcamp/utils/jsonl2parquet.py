import pandas as pd
import json

def safe_json_dumps(x):
    """安全地将对象转为 JSON 字符串，失败时转为普通字符串"""
    try:
        return json.dumps(x, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(x)

def is_empty_value(val):
    """判断是否是'空值'，用于决定是否删除字段"""
    if val is None:
        return True
    if isinstance(val, str) and val == "":
        return True
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    # 可选：扩展其他空值类型，如 set(), 0, False 等（目前不删）
    return False

def recursive_convert(obj, keys_to_convert=('ground_truth', 'identity'), remove_empty=True):
    """
    递归遍历对象（dict/list）：
      1. 如果是字典，且键在 keys_to_convert 中 → 转为 JSON 字符串
      2. 如果 remove_empty=True，且值是“空值” → 删除该键
      3. 递归处理子结构
    """
    if isinstance(obj, dict):
        # 注意：遍历时不能直接 del，先收集要删的键
        keys_to_delete = []
        for key in list(obj.keys()):
            val = obj[key]

            # 1. 如果是目标键，且非空 → 转 JSON 字符串
            if key in keys_to_convert and not is_empty_value(val):
                obj[key] = safe_json_dumps(val)

            # 2. 递归处理子对象（无论是否目标键）
            recursive_convert(val, keys_to_convert, remove_empty)

            # 3. 如果开启 remove_empty 且当前值为空 → 标记删除
            if remove_empty and is_empty_value(val):
                keys_to_delete.append(key)

        # 统一删除空字段（避免运行时改变 dict 大小）
        for key in keys_to_delete:
            del obj[key]

    elif isinstance(obj, list):
        # 递归处理列表中每个元素
        for i in range(len(obj)):
            recursive_convert(obj[i], keys_to_convert, remove_empty)

    return obj

def jsonl_to_parquet(jsonl_path, parquet_path, to_str=False, convert_keys=('ground_truth', 'identity'), remove_empty=True):
    """
    将 JSONL 转为 Parquet。
    - to_str=True: 递归转换 convert_keys 中的字段为 JSON 字符串
    - remove_empty=True: 递归删除所有“空值”字段（None/""/[]/{}）
    """
    df = pd.read_json(jsonl_path, lines=True)

    if to_str or remove_empty:  # 只要开启任一功能，就递归处理
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: recursive_convert(x, convert_keys, remove_empty) if pd.notnull(x) else x
            )

    df.to_parquet(parquet_path, index=False)
    print(f"✅ 成功将 {jsonl_path} 转换为 {parquet_path}")

if __name__ == "__main__":
    jsonl_path = "/path/to/v1_bootcamps_train.jsonl"
    parquet_path = jsonl_path.replace(".jsonl", ".parquet")
    jsonl_to_parquet(jsonl_path, parquet_path, to_str=True)
    # print("parquet_path: ", parquet_path)
    # jsonl_to_parquet("./verl/data/verl_oeis_test.jsonl", "./verl/data/verl_oeis_test.parquet")

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def binary_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    二分类 log-loss 计算函数。

    Args:
        y_true: 形状为 (N,) 的整数标签数组，取值为 0 或 1
        y_pred: 形状为 (N,) 的预测概率数组，表示正类（标签为1）的概率
        eps:   为了避免 log(0) 做的数值截断
    """
    if y_true.ndim != 1:
        raise ValueError(f"y_true 必须是一维数组，当前形状: {y_true.shape}")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred 必须是一维数组，当前形状: {y_pred.shape}")
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true 和 y_pred 的样本数不一致: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    # 数值稳定处理
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    
    # 二分类 log-loss: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(np.mean(loss))


def multiclass_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    多分类 log-loss 计算函数。

    Args:
        y_true: 形状为 (N,) 的整数标签数组，取值范围为 [0, K-1]
        y_pred: 形状为 (N, K) 的预测概率数组，每一行对应一个样本
        eps:   为了避免 log(0) 做的数值截断
    """
    if y_true.ndim != 1:
        raise ValueError(f"y_true 必须是一维数组，当前形状: {y_true.shape}")
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred 必须是二维数组，当前形状: {y_pred.shape}")
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true 和 y_pred 的样本数不一致: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    # 数值稳定处理
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    # 行归一化，确保每行概率之和为 1
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred = y_pred / row_sums

    # 取出每个样本在真实标签上的预测概率
    idx = (np.arange(y_true.shape[0]), y_true)
    p_true = y_pred[idx]

    return float(-np.mean(np.log(p_true)))


def accuracy_from_proba(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_binary: bool = False,
) -> float:
    """
    根据预测概率计算 top-1 准确率。
    
    Args:
        y_true: 真实标签（一维数组）
        y_pred: 预测概率（对于多分类是二维数组，对于二分类是一维数组）
        is_binary: 是否为二分类问题
    """
    if y_true.ndim != 1:
        raise ValueError("y_true 必须是一维数组。")
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true 和 y_pred 的样本数不一致: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )
    
    if is_binary:
        # 二分类：将概率转换为预测标签（>=0.5 为 1，否则为 0）
        if y_pred.ndim != 1:
            raise ValueError("二分类时 y_pred 必须是一维数组。")
        pred_labels = (y_pred >= 0.5).astype(int)
    else:
        # 多分类：取概率最大的类别
        if y_pred.ndim != 2:
            raise ValueError("多分类时 y_pred 必须是二维数组。")
        pred_labels = np.argmax(y_pred, axis=1)
    
    return float(np.mean(pred_labels == y_true))


def _identify_numeric_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    """
    识别 DataFrame 中的数值列（排除指定列）。
    
    Args:
        df: 输入的 DataFrame
        exclude_cols: 要排除的列名列表
    
    Returns:
        数值列名列表
    """
    numeric_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        # 尝试转换为数值类型，如果成功则认为是数值列
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # 如果转换失败，跳过该列
            continue
    return numeric_cols


def _load_submission(
    submission_path: str, 
    id_column: Optional[str] = "id",
    is_binary: bool = False,
    binary_label_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], bool]:
    """
    加载 submission.csv，返回 DataFrame、类别列名列表和是否使用行号对齐。
    
    Args:
        submission_path: submission.csv 文件路径
        id_column: id 列名，如果为 None 则按行号对齐
        is_binary: 是否为二分类问题
        binary_label_column: 二分类场景下的标签列名（如果指定，则只使用该列）
    
    Returns:
        (DataFrame, 类别列名列表, 是否使用行号对齐)
    """
    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        raise ValueError(f"加载 submission.csv 失败: {e}")
    has_id_column = id_column is not None and id_column in sub.columns
    
    if not has_id_column:
        # 如果没有 id 列，创建基于行号的 id
        if id_column is None:
            id_column = "__row_index__"
        sub[id_column] = np.arange(len(sub))
    
    # 确定类别列
    if is_binary and binary_label_column is not None:
        # 二分类且指定了标签列名
        if binary_label_column not in sub.columns:
            raise KeyError(f"submission.csv 中未找到指定的标签列 '{binary_label_column}'。")
        class_cols = [binary_label_column]
    else:
        # 自动识别数值列作为类别列
        exclude_cols = [id_column] if id_column else []
        class_cols = _identify_numeric_columns(sub, exclude_cols)
        
        if not class_cols:
            raise ValueError("submission.csv 中未找到任何数值类别列。")
    
    # 检查是否有重复 id（仅在存在原始 id 列时检查）
    if has_id_column and sub[id_column].duplicated().any():
        dup_count = int(sub[id_column].duplicated().sum())
        raise ValueError(f"submission.csv 中存在重复 id，数量: {dup_count}")

    return sub, class_cols, not has_id_column


def _load_private_test(
    private_test_path: str, 
    class_cols: List[str], 
    id_column: Optional[str] = "id",
    is_binary: bool = False,
    use_row_index: bool = False,
    binary_label_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载 private/test.csv。
    
    Args:
        private_test_path: private test 文件路径
        class_cols: 类别列名列表（从 submission 中获取）
        id_column: id 列名
        is_binary: 是否为二分类问题
        use_row_index: 是否使用行号对齐
        binary_label_column: 二分类场景下的标签列名
    
    Returns:
        包含 id 和类别列的 DataFrame
    """
    df = pd.read_csv(private_test_path)
    
    # 处理 id 列
    if use_row_index:
        if id_column is None:
            id_column = "__row_index__"
        if id_column not in df.columns:
            df[id_column] = np.arange(len(df))
    else:
        if id_column is None or id_column not in df.columns:
            raise KeyError(f"private test 文件中必须包含 '{id_column}' 列。")

    # 确定实际使用的类别列
    if is_binary and binary_label_column is not None:
        # 二分类且指定了标签列名
        if binary_label_column not in df.columns:
            raise KeyError(f"private test 文件中未找到指定的标签列 '{binary_label_column}'。")
        actual_class_cols = [binary_label_column]
    else:
        # 使用从 submission 中获取的类别列
        actual_class_cols = class_cols
        missing_cols = [c for c in actual_class_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"private test 文件中缺失以下类别列: {missing_cols}"
            )

    # 返回包含 id 和类别列的 DataFrame
    cols = [id_column] + actual_class_cols
    return pd.DataFrame(df[cols]).copy()


def _align_data(
    submission_df: pd.DataFrame,
    private_df: pd.DataFrame,
    class_cols: List[str],
    id_column: str,
    use_row_index: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    对齐 submission 和 private test 数据。
    
    Returns:
        (对齐后的 private test DataFrame, 对齐后的 submission DataFrame, 样本数)
    """
    if use_row_index:
        # 按行号一一对应
        n_private = len(private_df)
        n_submission = len(submission_df)
        
        if n_submission < n_private:
            raise ValueError(
                f"submission.csv 的行数 ({n_submission}) 少于 private test 的行数 ({n_private})，无法一一对应。"
            )
        
        priv_aligned = private_df[class_cols]
        sub_aligned = submission_df.iloc[:n_private][class_cols]
        n_samples = n_private
    else:
        # 按 id 对齐
        sub_indexed = submission_df.set_index(id_column)
        priv_indexed = private_df.set_index(id_column)

        n_private = priv_indexed.shape[0]
        common_ids = priv_indexed.index.intersection(sub_indexed.index)
        
        if common_ids.empty:
            raise ValueError("private test 与 submission.csv 的 id 没有交集，无法评测。")
        if len(common_ids) < n_private:
            missing = n_private - len(common_ids)
            raise ValueError(
                f"submission.csv 中缺失部分 private test 样本的 id，缺失数量: {missing}"
            )

        priv_aligned = priv_indexed.loc[common_ids][class_cols]
        sub_aligned = sub_indexed.loc[common_ids][class_cols]
        n_samples = len(common_ids)
    
    return priv_aligned, sub_aligned, n_samples


def evaluate_submission(
    submission_path: str,
    private_test_path: str,
    id_column: Optional[str] = "id",
    is_binary: bool = False,
    binary_label_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对 submission.csv 进行评测，支持多分类和二分类问题。

    对于多分类问题:
        - private/test.csv 中每一行是 one-hot 标签（id + 所有类别的 0/1 列）
        - submission.csv 中每一行是对应的概率分布（id + 所有类别的概率列）
    
    对于二分类问题:
        - private/test.csv 中每一行是标签（id + 一个标签列，值为 0 或 1）
        - submission.csv 中每一行是概率（id + 一个概率列，值为 0-1 之间的浮点数）
        - 如果指定了 binary_label_column，则使用该列作为标签列

    评测内容:
        - 对齐 id 后，计算 log-loss（二分类或多分类）
        - 计算 top-1 准确率
        - 做一些基本合法性检查（id 对齐、类别列覆盖、概率归一等）
    
    注意:
        - 如果 id_column 为 None 或两个文件都没有 id_column，则按行号一一对应
        - 类别列会自动识别为数值列，避免包含字符串列

    返回:
        一个包含若干指标和统计信息的字典。
    """
    # 加载数据
    try: 
        submission_df, class_cols, use_row_index = _load_submission(
            submission_path, id_column, is_binary, binary_label_column
        )
    except Exception as e:
        raise ValueError(f"加载 submission.csv 失败: {e}")
    
    try:
        private_df = _load_private_test(
            private_test_path, class_cols, id_column, is_binary, use_row_index, binary_label_column
        )
    except Exception as e:
        raise ValueError(f"加载 private/test.csv 失败: {e}")

    # 对齐数据
    try:
        priv_aligned, sub_aligned, n_samples = _align_data(
            submission_df, private_df, class_cols, id_column, use_row_index
        )
    except Exception as e:
        raise ValueError(f"对齐数据失败: {e}")

    # 检查预测概率是否合法（非负、有限）
    try:
        y_pred_raw = sub_aligned.to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"submission.csv 中的类别列包含非数值数据，无法转换为浮点数。"
            f"请确保类别列只包含数值。错误详情: {e}"
        )
    
    # if not np.isfinite(y_pred_raw).all():
    #     raise ValueError("submission.csv 中存在非有限值 (NaN / inf)。")
    # if (y_pred_raw < 0).any():
    #     raise ValueError("submission.csv 中存在小于 0 的概率值。")

    # 根据分类类型计算指标
    if is_binary:
        # 二分类处理
        class_col = class_cols[0]
        y_true = priv_aligned[class_col].to_numpy(dtype=float)
        y_pred = sub_aligned[class_col].to_numpy(dtype=float)
        
        # 检查真实标签是否为 0 或 1
        if not np.isin(y_true, [0.0, 1.0]).all():
            raise ValueError(f"private test 中的标签列 '{class_col}' 包含非 0/1 的值。")
        
        # 检查预测概率是否在合理范围内
        if (y_pred > 1.0).any():
            raise ValueError("submission.csv 中存在大于 1 的概率值。")
        
        # 计算二分类指标
        logloss = binary_log_loss(y_true, y_pred)
        acc = accuracy_from_proba(y_true, y_pred, is_binary=True)
        
        return {
            "n_samples": n_samples,
            "n_classes": 2,
            "is_binary": True,
            "log_loss": logloss,
            "accuracy": acc,
            "class_columns": class_cols,
            "use_row_index": use_row_index,
        }
    else:
        # 多分类处理
        y_true_one_hot = priv_aligned.to_numpy(dtype=float)
        y_pred = y_pred_raw

        # 检查真实标签是否是 one-hot（每行和为 1，且元素为 0/1）
        row_sums = y_true_one_hot.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("private test 中的标签行不是严格的 one-hot（行和不为 1）。")
        if not np.isin(y_true_one_hot, [0.0, 1.0]).all():
            raise ValueError("private test 中的标签包含非 0/1 的值。")

        # 从 one-hot 还原为整数标签索引
        y_true_idx = np.argmax(y_true_one_hot, axis=1)

        # 计算多分类指标
        logloss = multiclass_log_loss(y_true_idx, y_pred)
        acc = accuracy_from_proba(y_true_idx, y_pred, is_binary=False)

        return {
            "n_samples": n_samples,
            "n_classes": int(len(class_cols)),
            "is_binary": False,
            "log_loss": logloss,
            "accuracy": acc,
            "class_columns": class_cols,
            "use_row_index": use_row_index,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate submission using private test labels (supports both binary and multiclass)."
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission.csv generated by the model.",
    )
    parser.add_argument(
        "--private-test",
        type=str,
        required=True,
        help="Path to private/test.csv with ground-truth labels.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Name of the ID column (default: 'id'). Use 'request_id' for random-acts-of-pizza. Use 'None' for row-based alignment.",
    )
    parser.add_argument(
        "--is-binary",
        action="store_true",
        help="Whether this is a binary classification task.",
    )
    parser.add_argument(
        "--binary-label-column",
        type=str,
        default=None,
        help="Name of the binary label column (for binary classification tasks).",
    )

    args = parser.parse_args()

    # 处理 id_column 参数
    id_col = None if args.id_column.lower() == "none" else args.id_column

    metrics = evaluate_submission(
        submission_path=args.submission,
        private_test_path=args.private_test,
        id_column=id_col,
        is_binary=args.is_binary,
        binary_label_column=args.binary_label_column,
    )

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

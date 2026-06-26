"""Submission metrics for Agentic ML toy/Kaggle-style tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def binary_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, got {y_true.shape}")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be one-dimensional, got {y_pred.shape}")
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Sample count mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))))


def multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, got {y_true.shape}")
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be two-dimensional, got {y_pred.shape}")
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Sample count mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(y_pred[(np.arange(y_true.shape[0]), y_true)])))


def accuracy_from_proba(y_true: np.ndarray, y_pred: np.ndarray, *, is_binary: bool = False) -> float:
    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Sample count mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    pred_labels = (y_pred >= 0.5).astype(int) if is_binary else np.argmax(y_pred, axis=1)
    return float(np.mean(pred_labels == y_true))


def evaluate_submission(
    submission_path: str,
    private_test_path: str,
    id_column: str | None = "id",
    is_binary: bool = False,
    binary_label_column: str | None = None,
) -> dict[str, Any]:
    submission_df, class_cols, use_row_index = _load_submission(
        submission_path,
        id_column,
        is_binary,
        binary_label_column,
    )
    private_df = _load_private_test(
        private_test_path,
        class_cols,
        id_column,
        is_binary,
        use_row_index,
        binary_label_column,
    )
    priv_aligned, sub_aligned, n_samples = _align_data(
        submission_df,
        private_df,
        class_cols,
        "__row_index__" if id_column is None else id_column,
        use_row_index,
    )

    y_pred_raw = sub_aligned.to_numpy(dtype=float)
    if is_binary:
        class_col = class_cols[0]
        y_true = priv_aligned[class_col].to_numpy(dtype=float)
        y_pred = sub_aligned[class_col].to_numpy(dtype=float)
        if not np.isin(y_true, [0.0, 1.0]).all():
            raise ValueError(f"Private labels in '{class_col}' must be 0 or 1.")
        if (y_pred > 1.0).any() or (y_pred < 0.0).any():
            raise ValueError("Binary prediction probabilities must be in [0, 1].")
        return {
            "n_samples": n_samples,
            "n_classes": 2,
            "is_binary": True,
            "log_loss": binary_log_loss(y_true, y_pred),
            "accuracy": accuracy_from_proba(y_true, y_pred, is_binary=True),
            "class_columns": class_cols,
            "use_row_index": use_row_index,
        }

    y_true_one_hot = priv_aligned.to_numpy(dtype=float)
    if not np.allclose(y_true_one_hot.sum(axis=1), 1.0):
        raise ValueError("Private multiclass labels must be one-hot rows.")
    if not np.isin(y_true_one_hot, [0.0, 1.0]).all():
        raise ValueError("Private multiclass labels must contain only 0/1 values.")
    y_true_idx = np.argmax(y_true_one_hot, axis=1)
    return {
        "n_samples": n_samples,
        "n_classes": int(len(class_cols)),
        "is_binary": False,
        "log_loss": multiclass_log_loss(y_true_idx, y_pred_raw),
        "accuracy": accuracy_from_proba(y_true_idx, y_pred_raw, is_binary=False),
        "class_columns": class_cols,
        "use_row_index": use_row_index,
    }


def _load_submission(
    submission_path: str,
    id_column: str | None,
    is_binary: bool,
    binary_label_column: str | None,
) -> tuple[pd.DataFrame, list[str], bool]:
    sub = pd.read_csv(submission_path)
    has_id_column = id_column is not None and id_column in sub.columns
    if not has_id_column:
        id_column = "__row_index__"
        sub[id_column] = np.arange(len(sub))

    if is_binary and binary_label_column is not None:
        if binary_label_column not in sub.columns:
            raise KeyError(f"Missing binary label column '{binary_label_column}' in submission.")
        class_cols = [binary_label_column]
    else:
        class_cols = _identify_numeric_columns(sub, [id_column])
        if not class_cols:
            raise ValueError("No numeric prediction columns found in submission.")

    if has_id_column and sub[id_column].duplicated().any():
        raise ValueError("Duplicate ids found in submission.")

    return sub, class_cols, not has_id_column


def _load_private_test(
    private_test_path: str,
    class_cols: list[str],
    id_column: str | None,
    is_binary: bool,
    use_row_index: bool,
    binary_label_column: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(private_test_path)
    effective_id = "__row_index__" if id_column is None else id_column
    if use_row_index:
        if effective_id not in df.columns:
            df[effective_id] = np.arange(len(df))
    elif effective_id not in df.columns:
        raise KeyError(f"Private test file must contain id column '{effective_id}'.")

    actual_class_cols = [binary_label_column] if is_binary and binary_label_column else class_cols
    missing_cols = [col for col in actual_class_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Private test file is missing label columns: {missing_cols}")
    return pd.DataFrame(df[[effective_id] + actual_class_cols]).copy()


def _align_data(
    submission_df: pd.DataFrame,
    private_df: pd.DataFrame,
    class_cols: list[str],
    id_column: str,
    use_row_index: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if use_row_index:
        n_private = len(private_df)
        if len(submission_df) < n_private:
            raise ValueError("submission.csv has fewer rows than private test.")
        return private_df[class_cols], submission_df.iloc[:n_private][class_cols], n_private

    sub_indexed = submission_df.set_index(id_column)
    priv_indexed = private_df.set_index(id_column)
    common_ids = priv_indexed.index.intersection(sub_indexed.index)
    if common_ids.empty:
        raise ValueError("No overlapping ids between private test and submission.")
    if len(common_ids) < len(priv_indexed):
        raise ValueError("submission.csv is missing private test ids.")
    return priv_indexed.loc[common_ids][class_cols], sub_indexed.loc[common_ids][class_cols], len(common_ids)


def _identify_numeric_columns(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    numeric_cols: list[str] = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    return numeric_cols

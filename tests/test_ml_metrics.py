from pathlib import Path

from timely_eval.ml_metrics import evaluate_submission


def test_binary_submission_metric(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    private = tmp_path / "private.csv"
    submission.write_text("id,label\n1,0.1\n2,0.9\n", encoding="utf-8")
    private.write_text("id,label\n1,0\n2,1\n", encoding="utf-8")

    result = evaluate_submission(
        str(submission),
        str(private),
        id_column="id",
        is_binary=True,
        binary_label_column="label",
    )
    assert result["accuracy"] == 1.0
    assert result["n_samples"] == 2


def test_multiclass_submission_metric(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    private = tmp_path / "private.csv"
    submission.write_text("id,A,B\n1,0.9,0.1\n2,0.2,0.8\n", encoding="utf-8")
    private.write_text("id,A,B\n1,1,0\n2,0,1\n", encoding="utf-8")

    result = evaluate_submission(str(submission), str(private), id_column="id")
    assert result["accuracy"] == 1.0
    assert result["n_classes"] == 2

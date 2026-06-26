from timely_eval.scoring import compute_score


def test_compute_score_boxed() -> None:
    assert compute_score("The answer is \\boxed{4}.", "4") == 1.0


def test_compute_score_plain() -> None:
    assert compute_score("0.5", "\\frac{1}{2}") == 1.0


def test_compute_score_none() -> None:
    assert compute_score(None, "4") == 0.0

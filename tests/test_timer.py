from timely_eval.timer import Timer


def test_timer_starts_and_returns_elapsed_value() -> None:
    timer = Timer(mode="eval")
    timer.start()

    elapsed = timer.call(return_format="value")

    assert isinstance(elapsed, float)
    assert elapsed >= 0.0

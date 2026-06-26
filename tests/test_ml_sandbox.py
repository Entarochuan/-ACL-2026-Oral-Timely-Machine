import asyncio
import os
from pathlib import Path

import pytest

from timely_eval.ml_sandbox import (
    _clean_environment,
    _write_user_script,
    create_isolated_workspace,
    run_python_code_in_isolation,
)


def test_create_isolated_workspace(tmp_path: Path) -> None:
    work_dir = create_isolated_workspace(str(tmp_path))
    assert Path(work_dir).is_dir()
    assert Path(work_dir).name.startswith("ml_sim_")


def test_clean_environment_removes_secrets() -> None:
    env = {
        "OPENAI_API_KEY": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "PATH": "/usr/bin",
    }
    cleaned = _clean_environment(env)
    assert "OPENAI_API_KEY" not in cleaned
    assert "AWS_SECRET_ACCESS_KEY" not in cleaned
    assert cleaned["PATH"] == "/usr/bin"


def test_write_user_script_creates_missing_workspace(tmp_path: Path) -> None:
    work_dir = tmp_path / "missing" / "workspace"

    script_path = _write_user_script("print('ok')", str(work_dir))

    assert script_path.is_file()
    assert script_path.parent == work_dir


@pytest.mark.skipif(
    os.getenv("TIMELY_EVAL_RUN_SUBPROCESS_TESTS") != "1",
    reason="Subprocess spawning is environment-dependent; set TIMELY_EVAL_RUN_SUBPROCESS_TESTS=1 to run.",
)
def test_run_python_code_in_isolation(tmp_path: Path) -> None:
    async def _run():
        return await run_python_code_in_isolation(
            """
            import csv
            with open("submission.csv", "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "label"])
                writer.writerow([1, 0.5])
            """,
            base_dir=str(tmp_path),
            timeout=5,
            preserve_workspace=True,
        )

    result = asyncio.run(_run())
    assert result.returncode == 0
    assert result.submission_path is not None
    assert Path(result.submission_path).exists()

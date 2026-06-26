"""Run model-generated Python code in an isolated working directory.

This is process isolation, not a security sandbox. For untrusted models, run
Timely Eval inside a container or VM with restricted filesystem and network.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timeout: bool
    work_dir: str
    execution_time: float
    submission_path: str | None


def create_isolated_workspace(base_dir: str | None = None) -> str:
    root = Path(base_dir).expanduser().absolute() if base_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / f"ml_sim_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def _write_user_script(code: str, work_dir: str, filename: str = "solution.py") -> Path:
    work_path = Path(work_dir).absolute()
    work_path.mkdir(parents=True, exist_ok=True)
    script_path = work_path / filename
    script_path.write_text(textwrap.dedent(code).strip() + "\n", encoding="utf-8")
    return script_path


async def run_python_code_in_isolation(
    code: str,
    *,
    base_dir: str | None = None,
    timeout: int = 180,
    input_dir: str | None = None,
    preserve_workspace: bool = False,
    extra_env: dict[str, str] | None = None,
    saved_work_dir: str | None = None,
) -> ExecutionResult:
    """Execute Python code and return stdout/stderr plus submission path."""

    start_time = time.time()
    work_dir = saved_work_dir or create_isolated_workspace(base_dir)
    result: ExecutionResult | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = -1

    try:
        script_path = _write_user_script(code, work_dir)
        if input_dir is not None:
            src = Path(input_dir).expanduser().absolute()
            dst = Path(work_dir) / "data"
            if not dst.exists() and not dst.is_symlink():
                try:
                    os.symlink(src, dst, target_is_directory=True)
                except OSError:
                    if src.is_dir():
                        shutil.copytree(src, dst)

        env = _clean_environment(os.environ.copy())
        if extra_env:
            env.update(extra_env)

        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    sys.executable or "python",
                    str(script_path),
                    cwd=work_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                returncode = process.returncode or 0
                stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            except asyncio.TimeoutError:
                timed_out = True
                returncode = -1
                process.kill()
                await process.wait()
                stderr = f"[timely_eval] Execution timed out after {timeout} seconds."
        except asyncio.TimeoutError:
            result = ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"[timely_eval] Failed to start user code within {timeout} seconds.",
                timeout=True,
                work_dir=work_dir,
                submission_path=None,
                execution_time=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            result = ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"[timely_eval] Failed to execute user code: {exc!r}",
                timeout=False,
                work_dir=work_dir,
                submission_path=None,
                execution_time=time.time() - start_time,
            )

        if result is None:
            submission_file = Path(work_dir) / "submission.csv"
            result = ExecutionResult(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                timeout=timed_out,
                work_dir=work_dir,
                submission_path=str(submission_file.absolute()) if submission_file.is_file() else None,
                execution_time=time.time() - start_time,
            )
    finally:
        if not preserve_workspace and saved_work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


def _clean_environment(env: dict[str, str]) -> dict[str, str]:
    blocked_fragments = (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "aws_",
        "gcp_",
        "google_",
        "azure_",
        "api_key",
        "secret",
        "token",
    )
    for key in list(env):
        lower = key.lower()
        if any(fragment in lower for fragment in blocked_fragments):
            env.pop(key, None)
    return env

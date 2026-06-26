import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import time

benchmark_configs = dict(
    leaf_classification=dict(
        data_path="./ML_source/leaf-classification/prepared",
        private_test_path="./ML_source/leaf-classification/prepared/private/test.csv",
        benchmark_name="leaf-classification",
        work_dir="workspace/leaf-classification",
        prompt_template="./prompt_templates/leaf_classification.txt",
        id_column="id",
        is_binary=False,
        binary_label_column=None,
    ),
    random_acts_of_pizza=dict(
        data_path="./ML_source/random-acts-of-pizza/prepared",
        private_test_path="./ML_source/random-acts-of-pizza/prepared/private/test.csv",
        benchmark_name="random-acts-of-pizza",
        work_dir="workspace/random-acts-of-pizza",                                                                                                                                                                                                                                                                                                                                                                                                                
        prompt_template="./prompt_templates/random-acts-of-pizza.txt",
        id_column="request_id",
        is_binary=True,
        binary_label_column="requester_received_pizza",
    ),
    detecting_insults_in_social_commentary=dict(
        data_path="./ML_source/detecting-insults-in-social-commentary/prepared",
        private_test_path="./ML_source/detecting-insults-in-social-commentary/prepared/private/test.csv",
        benchmark_name="detecting-insults-in-social-commentary",
        work_dir="workspace/detecting-insults-in-social-commentary",
        prompt_template="./prompt_templates/detecting-insults-in-social-commentary.txt",
        id_column=None,
        is_binary=True,
        binary_label_column="Insult",
    ),
    spaceship_titanic=dict(
        data_path="./ML_source/spaceship-titanic/prepared",
        private_test_path="./ML_source/spaceship-titanic/prepared/private/test.csv",
        benchmark_name="spaceship-titanic",
        work_dir="workspace/spaceship-titanic",
        prompt_template="./prompt_templates/spaceship-titanic.txt",
        id_column="PassengerId",
        is_binary=True,
        binary_label_column="Transported",
    ),
)

import asyncio
import os
import shutil
import sys
import tempfile
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import time

@dataclass
class ExecutionResult:
    """代码执行结果包装类。"""

    returncode: int
    stdout: str
    stderr: str
    timeout: bool
    work_dir: str
    execution_time: float
    submission_path: Optional[str]


def _safe_mkdir(path: Path) -> None:
    """
    安全创建目录。

    - 若目录已存在则直接返回；
    - 若父目录不存在则一并创建。
    """
    path.mkdir(parents=True, exist_ok=True)


def create_isolated_workspace(base_dir: Optional[str] = None) -> str:
    """
    创建一个隔离的工作目录，用于执行模型生成的代码。

    参数:
        base_dir: 若指定，则在该目录下创建子目录；否则使用系统临时目录。

    返回:
        新创建的工作目录绝对路径。
    """
    if base_dir is None:
        root = Path(tempfile.gettempdir())
    else:
        root = Path(base_dir).expanduser().absolute()

    _safe_mkdir(root)
    work_dir = root / f"ml_sim_{uuid.uuid4().hex}"
    _safe_mkdir(work_dir)
    return str(work_dir)


def _write_user_script(code: str, work_dir: str, filename: str = "solution.py") -> Path:
    """
    将用户/模型生成的代码写入隔离目录中的脚本文件。

    注意:
        - 会对代码做一次 `dedent`，避免因为缩进混乱导致语法错误；
        - 不会对代码做任何重写或“自动修复”，保持行为可控。
    """
    work_path = Path(work_dir).absolute()
    _safe_mkdir(work_path)

    script_path = work_path / filename
    normalized_code = textwrap.dedent(code).strip() + "\n"
    script_path.write_text(normalized_code, encoding="utf-8")
    return script_path


def _default_python_executable() -> str:
    """
    获取当前环境下的 Python 可执行文件路径。

    优先使用当前进程的解释器，确保依赖环境一致。
    """
    return sys.executable or "python"


async def run_python_code_in_isolation(
    code: str,
    *,
    base_dir: Optional[str] = None,
    timeout: int = 1800,
    input_dir: Optional[str] = None,
    preserve_workspace: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
    saved_work_dir: Optional[str] = None,
) -> ExecutionResult:
    """
    在隔离的工作目录中执行一段 Python 代码。

    典型使用场景:
        - LLM 根据 Kaggle 任务生成训练 / 推理脚本；
        - 该脚本需要在指定的数据目录上训练模型，并在工作目录下生成 `submission.csv`。

    参数:
        code:
            待执行的 Python 源代码字符串。通常为 LLM 生成的完整脚本。
        base_dir:
            隔离工作目录的根目录；若为 None，则使用系统临时目录。
        timeout:
            子进程执行的最大时间（秒）。超时将被终止，结果中 `timeout=True`。
        input_dir:
            只读数据目录（例如 Kaggle 提供的原始数据路径）。
            若提供，将在隔离目录中创建 `data` 软链接指向该目录：
                work_dir / "data" -> input_dir
        preserve_workspace:
            若为 True，则执行结束后保留工作目录（便于调试 / 手动检查）；
            若为 False，则在成功执行后自动删除工作目录。
        extra_env:
            传递给子进程的额外环境变量字典。

    返回:
        ExecutionResult:
            - returncode: 子进程退出码；
            - stdout / stderr: 子进程标准输出与标准错误；
            - timeout: 是否因超时被终止；
            - work_dir: 实际使用的工作目录路径；
            - submission_path: 若在工作目录下检测到 `submission.csv`，则返回其绝对路径，否则为 None。
    """

    start_time = time.time()

    if saved_work_dir is not None:
        work_dir = saved_work_dir
    else:
        work_dir = create_isolated_workspace(base_dir)
    # 统一在 finally 中做清理，确保异常路径也能回收临时目录
    result: Optional[ExecutionResult] = None
    timed_out = False

    # 预先占位，避免类型检查警告
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1

    try:
        # 为用户代码准备脚本文件
        script_path = _write_user_script(code, work_dir)

        # 若提供了只读数据目录，在工作目录中创建软链接 data -> input_dir
        if input_dir is not None:
            src = Path(input_dir).expanduser().absolute()
            dst = Path(work_dir) / "data"
            if dst.exists() or dst.is_symlink():
                # 避免覆盖已有内容
                pass
            else:
                try:
                    os.symlink(src, dst, target_is_directory=True)
                except OSError:
                    # 某些环境可能不支持软链接，退化为复制目录（可能较慢）
                    if src.is_dir():
                        shutil.copytree(src, dst)

        env = os.environ.copy()

        # 限制子进程中可能影响安全的环境变量（如代理、凭据等）
        for key in list(env.keys()):
            lower = key.lower()
            if any(p in lower for p in ["http_proxy", "https_proxy", "all_proxy", "no_proxy", "aws_", "gcp_", "azure_"]):
                env.pop(key, None)

        if extra_env:
            env.update(extra_env)

        python_exe = _default_python_executable()

        try:
            # 使用 asyncio 异步执行子进程，避免阻塞事件循环
            process = await asyncio.create_subprocess_exec(
                python_exe, str(script_path),
                cwd=work_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                # 等待进程完成，带超时
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                returncode = process.returncode or 0
                stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            except asyncio.TimeoutError:
                # 超时，终止进程
                timed_out = True
                returncode = -1
                process.kill()
                await process.wait()  # 确保进程完全终止
                stdout = ""
                stderr = f"[ML_simulator] Execution timed out after {timeout} seconds."
                
        except Exception as e:
            # 子进程启动或执行过程中的非超时异常，封装为 ExecutionResult
            execution_time = time.time() - start_time
            result = ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"[ML_simulator] Failed to execute user code: {e}",
                timeout=False,
                work_dir=work_dir,
                submission_path=None,
                execution_time=execution_time,
            )

        # 正常执行或超时情况下，检查 submission.csv 并构造结果
        if result is None:
            submission_file = Path(work_dir) / "submission.csv"
            if submission_file.exists() and submission_file.is_file():
                submission_path = str(submission_file.absolute())
            else:
                submission_path = None

            execution_time = time.time() - start_time
            result = ExecutionResult(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                timeout=timed_out,
                work_dir=work_dir,
                submission_path=submission_path,
                execution_time=execution_time,
            )

    finally:
        # 若无需保留工作目录，则无论是否出错都尝试清理
        if not preserve_workspace:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                # 清理失败不应影响主流程
                pass

    return result


__all__ = [
    "ExecutionResult",
    "create_isolated_workspace",
    "run_python_code_in_isolation",
]
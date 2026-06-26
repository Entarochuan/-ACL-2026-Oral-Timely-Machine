# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4
import re

import requests

from internbootcamp.src.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

# 导入代码执行和评测模块
from internbootcamp.bootcamps.Basic_LLM_timer.ML_simulator import run_python_code_in_isolation
from internbootcamp.bootcamps.Basic_LLM_timer.metrics import evaluate_submission

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def extract_python_code(assistant_message: str) -> str:
    """
    从 assistant message 中提取 python 代码。
    """
    python_code_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    match = python_code_pattern.search(assistant_message)
    if match:
        return match.group(1).strip()
    else:
        return None

class MLTimerTool(BaseTool):
    """
    机器学习任务的计时工具。
    
    核心逻辑：
    1. create: 注册 timer server
    2. execute:
       - 记录工具调用开始时间
       - 执行代码 + 评测
       - 调用 /call 获取净时间（真实时间 - 之前所有工具调用时间）
       - 对净时间按 speed_factor 缩放
       - 调用 /add_tool_time 记录本次工具调用时间
       - 返回结果
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "ml_timer_tool",
                "description": "Execute machine learning code and get evaluation results with time tracking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute for the ML task",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.local_server_url = os.getenv("ML_TIMER_SERVER_URL", "http://localhost:8002")

    @rollout_trace_op
    async def create(self, instance_id: Optional[str] = None, identity: Optional[dict] = None, **kwargs) -> str:
        """
        创建工具实例并注册 timer。
        
        identity 参数应包含：
        - unique_tool_id: 唯一标识符
        - speed_factor: 时间缩放因子 (默认 1.0)
        - work_dir: 工作目录
        - data_path: 数据路径
        - private_test_path: 私有测试数据路径
        - id_column: ID 列名 (默认 "id")
        - is_binary: 是否为二分类 (默认 False)
        - binary_label_column: 二分类标签列名 (可选)
        - timeout: 代码执行超时时间 (默认 180)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        identity = identity or {}
        unique_tool_id = identity.get('unique_tool_id')
        latest_assistant_message = identity.get('latest_assistant_message')
        # print("[Debug MLTimerTool] latest_assistant_message:", latest_assistant_message)

        # 注册 Timer（使用新的 ml_task server，固定 static + 1.0）
        try:
            resp = requests.post(
                f"{self.local_server_url}/register",
                json={"id": unique_tool_id},
                timeout=30
            )
            resp.raise_for_status()
            logger.info(f"[MLTimerTool] Registered timer for {unique_tool_id}: {resp.json()}")
        except Exception as e:
            logger.error(f"[MLTimerTool] Failed to register timer: {e}")
            raise

        # 存储实例配置
        self._instance_dict[instance_id] = {
            "unique_tool_id": unique_tool_id,
            "task_id": identity.get('task_id'),
            "speed_factor": identity.get('speed_factor', 1.0),
            "work_dir": identity.get('work_dir', None),
            "data_path": identity.get('data_path', None),
            "private_test_path": identity.get('private_test_path', None),
            "id_column": identity.get('id_column', 'id'),
            "is_binary": identity.get('is_binary', False),
            "binary_label_column": identity.get('binary_label_column', None),
            "timeout": identity.get('timeout', 180),
            "current_work_dir": None,  # 保存当前工作目录以便连续执行
            "latest_assistant_message": identity.get('latest_assistant_message'),
        }

        # os.makedirs(identity.get('work_dir'), exist_ok=True)

        return instance_id

    async def _run_code_sync(self, instance_id: str, code: str) -> Tuple[Any, Dict[str, Any]]:
        """
        执行代码和评测。
        
        返回: (execution_result, evaluation_result)
        """
        instance = self._instance_dict[instance_id]
        
        # 执行代码
        print("Starting to run the python code...")
        result = await run_python_code_in_isolation(
            code=code,
            base_dir=instance['work_dir'],
            timeout=instance['timeout'],
            input_dir=instance['data_path'],
            preserve_workspace=True,
            extra_env={"KAGGLE_DATA_DIR": instance['data_path']} if instance['data_path'] else None,
        )
        print("Python code execution completed.")
        # print("[Debug MLTimerTool] result:", result)
        # 更新当前工作目录
        if instance['current_work_dir'] is None:
            instance['current_work_dir'] = result.work_dir
        
        # print("[Debug MLTimerTool] result.work_dir:", result.work_dir)
        
        # 评测结果
        evaluation_result = {}
        if result.returncode == 0 and result.submission_path:
            try:
                evaluation_result = evaluate_submission(
                    submission_path=result.submission_path,
                    private_test_path=instance['private_test_path'],
                    id_column=instance['id_column'],
                    is_binary=instance['is_binary'],
                    binary_label_column=instance['binary_label_column'],
                )
                evaluation_result['is_valid_submission'] = True
            except Exception as e:
                logger.warning(f"[MLTimerTool] Evaluation failed: {e}")
                evaluation_result = {
                    'is_valid_submission': False,
                    'reason': str(e),
                    'accuracy': 0.0,
                }
        elif result.returncode != 0:
            evaluation_result = {
                'is_valid_submission': False,
                'reason': f"Code execution failed with return code {result.returncode}",
                'accuracy': 0.0,
                'stderr': result.stderr,
            }
        else:
            evaluation_result = {
                'is_valid_submission': False,
                'reason': "No submission.csv generated",
                'accuracy': 0.0,
            }
        
        return result, evaluation_result

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """
        执行代码并返回计时结果。
        
        参数:
            instance_id: 实例 ID
            parameters: 工具参数字典，包含 'code' 字段
        
        返回:
            (result_string, reward, info_dict)
        """
        
        # 从 assistant message 中提取 Python 代码
        code = extract_python_code(self._instance_dict[instance_id]['latest_assistant_message'])

        if code == None: 
            return "Error: No executable python code provided. Please wrap your python code in ```python xxx `` and then call the tool.`", 0.0, {"error": "No executable python code provided"}
        
        instance = self._instance_dict.get(instance_id)
        if not instance:
            return "Error: Instance not found.", 0.0, {"error": "Instance not found"}
        
        unique_tool_id = instance['unique_tool_id']
        speed_factor = instance['speed_factor']
        
        # 1. 记录工具调用开始时间
        tool_start_time = time.time()
        
        # 2. 执行代码和评测
        try:
            exec_result, eval_result = await self._run_code_sync(instance_id, code)
        except Exception as e:
            logger.error(f"[MLTimerTool] Code execution error: {e}")
            exec_result = None
            eval_result = {
                'is_valid_submission': False,
                'reason': f"Execution error: {e}",
                'accuracy': 0.0,
            }
        
        # 3. 记录工具调用结束时间
        tool_end_time = time.time()
        tool_call_duration = tool_end_time - tool_start_time
        
        # 4. 调用 /call 获取净时间（真实时间 - 之前所有工具调用时间）
        try:
            call_resp = requests.post(
                f"{self.local_server_url}/call",
                json={"id": unique_tool_id, "return_format": "number"},
                timeout=10
            )
            call_resp.raise_for_status()
            call_data = call_resp.json()
            net_time = call_data.get('result', 0.0)
            real_time = call_data.get('real_time', 0.0)
            total_tool_time_before = call_data.get('total_tool_time', 0.0)
        except Exception as e:
            logger.error(f"[MLTimerTool] Failed to call timer: {e}")
            net_time = 0.0
            real_time = 0.0
            total_tool_time_before = 0.0
        
        print(f"[Debug MLTimerTool] net_time: {net_time}, real_time: {real_time}, total_tool_time_before: {total_tool_time_before}, existing_tool_call_times: {call_data.get('existing_tool_call_times', [])}")
        
        # 5. 对净时间按 speed_factor 缩放
        # 生成净时间应当是net_time再减去本次tool_call_duration(因为本次时间并没有被timer server记录)
        scaled_time = (net_time - tool_call_duration) * speed_factor
        # 返回的时间是经过缩放的真实时间加上真实的工具调用时间
        return_time = scaled_time + tool_call_duration + total_tool_time_before
        
        # 6. 调用 /add_tool_time 记录本次工具调用时间
        try:
            add_resp = requests.post(
                f"{self.local_server_url}/add_tool_time",
                json={"id": unique_tool_id, "tool_time": tool_call_duration},
                timeout=20
            )
            add_resp.raise_for_status()
            logger.info(f"[MLTimerTool] Added tool time {tool_call_duration:.2f}s for {unique_tool_id}")
        except Exception as e:
            logger.error(f"[MLTimerTool] Failed to add tool time: {e}")
        
        # 7. 构建返回结果
        # 获取 stdout 和 stderr，做空值保护
        stdout_str = exec_result.stdout if exec_result else ""
        stderr_str = eval_result.get('stderr', exec_result.stderr if exec_result else "")
        
        # 截断过长的输出（防止 Context Window 爆炸），可选
        if len(stdout_str) > 5000:
            stdout_str = stdout_str[:2000] + "\n...[Output Truncated]...\n" + stdout_str[-2000:]

        if eval_result.get('is_valid_submission', False):
            # === 情况 A: 成功生成提交并通过评测 ===
            accuracy = eval_result.get('accuracy', 0.0)
            result_str = (
                f"Code execution succeeded.\n"
                f"Stdout:\n{stdout_str}\n"  # 即使成功了，模型也可能打印了一些训练日志
                f"Evaluation accuracy: {accuracy:.4f}\n"
                f"You have spent {return_time:.2f} seconds (return time)."
            )
        else:
            # === 情况 B: 代码运行了，但没有生成有效提交 (可能是 EDA 阶段，也可能是报错) ===
            reason = eval_result.get('reason', 'Unknown error')
            
            # 判断是代码彻底崩了，还是仅仅没生成 csv
            if exec_result and exec_result.returncode != 0:
                status_msg = f"Code execution failed (Return Code {exec_result.returncode})."
            else:
                status_msg = "Code execution finished, but no valid 'submission.csv' was found (or evaluation failed)."

            result_str = (
                f"{status_msg}\n"
                f"Stdout:\n{stdout_str}\n"   # 【关键修改】必须展示 Stdout，否则 EDA 无法进行
                f"Stderr:\n{stderr_str}\n"
                f"Evaluation Failure Reason: {reason}\n"
                f"You have spent {return_time:.2f} seconds (return time)."
            )
            
        logger.info(f"[MLTimerTool] Execute result for {unique_tool_id}: scaled_time={scaled_time:.2f}s")
        
        return result_str, 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """释放实例资源"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# ==================== 测试代码 ====================
def test_ml_timer_tool():
    """测试 MLTimerTool"""
    import asyncio
    import time
    
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "ml_timer_tool",
            "description": "Execute machine learning code and get evaluation results with time tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute for the ML task",
                    },
                },
                "required": ["code"],
            },
        },
    })
    
    unique_id = "TEST_ID_1"
    print(f"[Test] Unique ID: {unique_id}")
    
    tool = MLTimerTool(config={}, tool_schema=tool_schema)
    
    # 创建实例
    create_kwargs = {
        "identity": {
            "unique_tool_id": unique_id,
            "speed_factor": 1.0,
            "work_dir": "./work_dir/ml_test/leaf-classification",
            "data_path": "./ML_source/data_sources/leaf-classification/prepared",
            "private_test_path": "./ML_source/data_sources/leaf-classification/prepared/private/test.csv",
            "timeout": 180,
            "id_column": "id",
            "is_binary": False,
            "binary_label_column": None,
        }
    }
    
    # 执行简单代码
    test_code = """
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('./data/public/train.csv')
test_data = pd.read_csv('./data/public/test.csv')

# Separate features and target
X = train_data.drop(['id', 'species'], axis=1)
y = train_data['species']

# Encode the species labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Prepare test features
X_test = test_data.drop(['id'], axis=1)

# Use cross-validation to find optimal parameters
# 保持 n_jobs=-1 以测试多核并行处理能力
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# --- 修改点 1：大幅缩减参数网格 ---
# 原来有 100+ 种组合，现在强制只用一种最简单的组合
param_grid = {
    'n_estimators': [10],      # 原来是 [100, 200, 300]，改为 10 棵树，速度极快
    'max_depth': [5],          # 原来是 [10, 15, 20, None]，改为 5 层，防止过深
    'min_samples_split': [2],  # 只留一个默认值
    'min_samples_leaf': [1]    # 只留一个默认值
}

# --- 修改点 2：减少交叉验证折数 ---
# n_splits 从 5 改为 2 (最小合法值)，大幅减少训练次数
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_model, param_grid, cv=cv, 
    scoring='neg_log_loss',
    n_jobs=-1, verbose=0
)

# Fit the grid search
print("Starting training (Quick Mode)...")
grid_search.fit(X, y_encoded)

# Get the best model
best_rf = grid_search.best_estimator_

# Generate predictions (probabilities)
predictions = best_rf.predict_proba(X_test)

# Create submission dataframe
submission = pd.DataFrame(predictions, columns=label_encoder.classes_)
submission.insert(0, 'id', test_data['id'])

# Ensure probabilities are in valid range
epsilon = 1e-15
submission.iloc[:, 1:] = np.clip(submission.iloc[:, 1:], epsilon, 1 - epsilon)

# Save submission file
submission.to_csv('./submission.csv', index=False)

# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")
print("Test finished successfully.")
```
"""
    create_kwargs['identity']['latest_assistant_message'] = test_code

    instance_id = asyncio.run(tool.create(instance_id=None, **create_kwargs))
    print(f"[Test] Instance created: {instance_id}")
    
    # 等待一秒模拟思考时间
    time.sleep(1)
    
    print("Starting to execute the code...")
    response, reward, info = asyncio.run(tool.execute(instance_id=instance_id, parameters={}))
    print("Code execution completed.")
    print(f"[Test] Response: {response}")
    # print(f"[Test] Info: {info}")
    
    # 释放资源
    asyncio.run(tool.release(instance_id=instance_id))
    print("[Test] Instance released")


if __name__ == "__main__":
    test_ml_timer_tool()
    # python -m internbootcamp.bootcamps.Basic_LLM_timer.MachineLearning_timer_tool

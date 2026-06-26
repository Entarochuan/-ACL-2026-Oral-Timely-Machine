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

import json
import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from internbootcamp.src.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op
from llm_timer import Timer
import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Tool 1: TimeCallForGeneralTasksTool
class TimeCallForGeneralTasksTool(BaseTool):
    """A tool for calling the general tasks with time limit."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "time_call_for_general_tasks",
                "description": "A tool for getting time duration when reasoning for general tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    @rollout_trace_op
    async def create(self, instance_id: Optional[str] = None, identity: dict = None, **kwargs) -> str:
        """创建工具实例"""
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = identity

        local_server_url = "http://localhost:8000"
        unique_tool_id = identity.get('unique_tool_id', None)

        # 注册 Timer
        requests.post(f"{local_server_url}/register", json={
            "id": unique_tool_id,
            "mode": identity['timer_mode'],
            "speed_factor": identity['timer_speed_factor']
        })

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": identity,
            "reward": 0.0,
            "unique_tool_id": unique_tool_id,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, **kwargs) -> Tuple[str, float, dict]:
        
        local_server_url = "http://localhost:8000"
        tool_response = requests.post(f"{local_server_url}/call", json={
            "id": self._instance_dict[instance_id]['unique_tool_id'],
            "return_format": "text" # 或者 value
        })
        
        print(f"[Debug LLM Timer Tools] Tool Response: {tool_response.json()}")
        return tool_response.json()['result'], 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

# Tool 2: TimeCallForGeneralTasksTool

def test_tools(tool_name: str):

    import asyncio
    import time
    import uuid

    # Test TimerCall tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "time_call_for_general_tasks",
            "description": "A tool for getting time duration when reasoning for general tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "The mode of the tool.",
                        "enum": ["get_duration", "terminate_timer"],
                    },
                },
                "required": ["mode"],
            },
        },
    })

    unique_id = uuid4().hex
    print(f"[Debug LLM Timer Tools] Unique ID: {unique_id}")
    tool = TimeCallForGeneralTasksTool(config=None, tool_schema=tool_schema)
    create_kwargs = {"identity": {"timer_mode": "static", "timer_speed_factor": 1.0}, "unique_tool_id": unique_id}
    instance_id = asyncio.run(tool.create(instance_id=None, **create_kwargs))
    time.sleep(1)

    create_kwargs = {"identity": {"timer_mode": "static", "timer_speed_factor": 1.0}, "unique_tool_id": unique_id}
    instance_id = asyncio.run(tool.create(instance_id=None, **create_kwargs))

    time.sleep(1)

    response, reward, info = asyncio.run(tool.execute(instance_id=instance_id))
    print(response, reward, info)
    asyncio.run(tool.release(instance_id=instance_id))

if __name__ == "__main__":

    # test_tools("timecall")
    test_tools("jericho")
    # python -m internbootcamp.bootcamps.Basic_LLM_timer.LLM_timer_tools
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

# --- 基类 (处理通用的注册和Server通信逻辑) ---
class JerichoBaseTool(BaseTool):
    """
    Jericho 游戏的基类工具。
    负责处理 Session 注册和与 HTTP Server 的底层通信。
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # 假设 Server 运行在本地 8001 端口
        self.server_url = "http://localhost:8001"

    @rollout_trace_op
    async def create(self, instance_id: Optional[str] = None, identity: dict = None, **kwargs) -> str:
        """
        所有 Jericho 子工具在初始化时都会调用此方法。
        它向 Server 注册 Session，确保后端环境已准备好。
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # 获取全局唯一的 Tool/Session ID
        unique_tool_id = identity.get('unique_tool_id')
        env_name = identity.get('env_name')
        speed_factor = identity.get('timer_speed_factor', 1.0)
        mode = identity.get('mode', 'static')

        try:
            # 向 Server 注册/保活 Session (幂等操作)
            requests.post(f"{self.server_url}/register", json={
                "id": unique_tool_id,
                "env_name": env_name,
                "timer_mode": mode,
                "speed_factor": speed_factor
            }, timeout=30)
        except Exception as e:
            logger.error(f"[JerichoTool] Failed to register session: {e}")

        self._instance_dict[instance_id] = {
            "unique_tool_id": unique_tool_id
        }
        return instance_id

    def _call_server(self, instance_id: str, mode: str, action: str = None) -> Tuple[str, float, dict]:
        """
        统一的 Server 调用逻辑
        """
        session_data = self._instance_dict.get(instance_id)
        if not session_data:
            return "Error: Session instance not found.", 0.0, {}

        unique_tool_id = session_data['unique_tool_id']
        
        payload = {"id": unique_tool_id, "mode": mode}
        if action:
            payload["action"] = action

        try:
            response = requests.post(f"{self.server_url}/execute", json=payload, timeout=30)
            
            if response.status_code == 200:
                resp_json = response.json()
                result = resp_json.get('result', '')
                
                # 简单的日志
                # print(f"[Debug] ID: {unique_tool_id} | Mode: {mode} | Action: {action}")
                
                # 格式化返回值
                if isinstance(result, list):
                    result_str = json.dumps(result)
                else:
                    result_str = str(result)
                return result_str, 0.0, {}
            else:
                err = f"Server Error ({response.status_code}): {response.text}"
                logger.error(err)
                return err, 0.0, {}
        except Exception as e:
            err = f"Execution Exception: {str(e)}"
            logger.error(err)
            return err, 0.0, {}
            
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# --- 具体功能的 Tool 类 ---

class JerichoStepTool(JerichoBaseTool):
    """
    Tool to perform an action in the game.
    """
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        
        print(f"[Debug Jericho Step Tool] Parameters: {parameters}")
        parameters.update(kwargs)
        print(f"[Debug Jericho Step Tool] Combined Parameters: {parameters}")
        
        # 3. 获取 action
        action = parameters.get('action')
        
        if not action:
            return f"An Error Occurred: 'action' parameter is required. Current parameters: {parameters}, kwargs: {kwargs}", 0.0, {}
            
        return self._call_server(instance_id, mode="step", action=action)


class JerichoGetAvailableActionsTool(JerichoBaseTool):
    """
    Tool to get a list of valid actions.
    """
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        return self._call_server(instance_id, mode="get_available_actions")


class JerichoGetScoreTool(JerichoBaseTool):
    """
    Tool to get the current score.
    """
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        return self._call_server(instance_id, mode="get_score")


class JerichoGetMaxScoreTool(JerichoBaseTool):
    """
    Tool to get the maximum possible score.
    """
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        return self._call_server(instance_id, mode="get_max_score")

class JerichoStepBackTool(JerichoBaseTool):
    """
    Tool to revert the last action (undo).
    """
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        return self._call_server(instance_id, mode="step_back")

class JerichoEndGameTool(JerichoBaseTool):
    """
    Tool to end the game.
    """
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        return self._call_server(instance_id, mode="end_game")


def test_tools(tool_name: str, env_name: str, instance_id: str):

    import asyncio
    import time
    tool = None
    if tool_name == "jericho_step":
        tool = JerichoStepTool(config={}, tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "JerichoStepTool",
                "description": "Tool to perform an action in the game.",
            }
        }))

    elif tool_name == "jericho_get_available_actions":
        tool = JerichoGetAvailableActionsTool(config={}, tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "JerichoGetAvailableActionsTool",
                "description": "Tool to get a list of valid actions.",
            }
        }))
    elif tool_name == "jericho_get_score":
        tool = JerichoGetScoreTool(config={}, tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "JerichoGetScoreTool",
                "description": "Tool to get the current score.",
            }
        }))
    elif tool_name == "jericho_get_max_score":
        tool = JerichoGetMaxScoreTool(config={}, tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "JerichoGetMaxScoreTool",
                "description": "Tool to get the maximum possible score.",
            }
        }))
    elif tool_name == "jericho_step_back":
        tool = JerichoStepBackTool(config={}, tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "JerichoStepBackTool",
                "description": "Tool to revert the last action (undo).",
            }
        }))
    elif tool_name == "jericho_end_game":
        tool = JerichoEndGameTool(config={}, tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "JerichoEndGameTool",
                "description": "Tool to end the game.",
            }
        }))
    else:
        raise ValueError(f"Invalid tool name: {tool_name}")
    
    unique_id = uuid4().hex
    print(f"[Debug Jericho Timer Tools] Unique ID: {unique_id}")
    tool = JerichoStepTool(config=None, tool_schema=OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "JerichoStepTool",
            "description": "Tool to perform an action in the game.",
        }
    }))
    create_kwargs = {"identity": {"unique_tool_id": unique_id, "env_name": env_name}}
    instance_id = asyncio.run(tool.create(instance_id=None, **create_kwargs))
    time.sleep(1)

    create_kwargs = {"identity": {"unique_tool_id": unique_id, "env_name": env_name}}
    instance_id = asyncio.run(tool.create(instance_id=None, **create_kwargs))
    response, reward, info = asyncio.run(tool.execute(instance_id=instance_id, parameters={"action": "go north"}))
    print(response, reward, info)
    asyncio.run(tool.release(instance_id=instance_id))

    time.sleep(1)
    
if __name__ == "__main__":
    instance_id = uuid4().hex
    env_name = "acorncourt.z5"
    tool_name = "jericho_step"
    test_tools(tool_name, env_name, instance_id)

    # python -m internbootcamp.bootcamps.Basic_LLM_timer.Jericho_timer_tool
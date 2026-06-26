import asyncio
import time
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# 假设 llm_timer 已经安装
from llm_timer import Timer

# --- 日志配置 ---
# 设置日志格式：时间 - 日志级别 - 消息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LLMTimerServer")

# --- 配置 ---
TTL_SECONDS = 1800       # 30分钟没有操作就回收
CLEANUP_INTERVAL = 120   # 每120秒执行一次清理扫描

# --- 数据结构封装 ---
@dataclass
class TimerEntry:
    instance: Timer
    last_accessed: float
    tool_call_times: List[float] = field(default_factory=list)  # 存储每次 tool call 的时间

# 全局存储: { "session_id": TimerEntry }
TIMER_STORE: Dict[str, TimerEntry] = {}

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    cleanup_task = asyncio.create_task(background_cleanup())
    yield
    logger.info("Server shutting down...")
    cleanup_task.cancel()

app = FastAPI(title="LLM Timer Server", lifespan=lifespan)

# --- 后台清理逻辑 ---
async def background_cleanup():
    logger.info(f"Background cleanup task started. TTL={TTL_SECONDS}s, Interval={CLEANUP_INTERVAL}s")
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = time.time()
            keys_to_remove = [
                tid for tid, entry in TIMER_STORE.items() 
                if now - entry.last_accessed > TTL_SECONDS
            ]
            for k in keys_to_remove:
                del TIMER_STORE[k]
            
            if keys_to_remove:
                logger.info(f"[Cleanup] Removed {len(keys_to_remove)} expired timers. Current active: {len(TIMER_STORE)}")
            else:
                # 只有调试时才开启这行，防止日志刷屏
                # logger.debug(f"[Cleanup] No expired timers found. Current active: {len(TIMER_STORE)}")
                pass
                
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled.")
            break
        except Exception as e:
            # 打印后台任务的未知错误，防止任务悄悄挂掉
            logger.error(f"[Cleanup Error] Unexpected error in cleanup loop: {e}", exc_info=True)

# --- 请求模型 ---
class RegisterRequest(BaseModel):
    id: str = Field(..., description="Unique identifier")

class CallRequest(BaseModel):
    id: str = Field(..., description="Unique identifier")
    return_format: str = Field("text")

class AddToolTimeRequest(BaseModel):
    id: str = Field(..., description="Unique identifier")
    tool_time: float = Field(..., description="Tool call duration to add")

# --- 接口实现 ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "active_timers": len(TIMER_STORE)}

@app.post("/register")
async def register_timer(req: RegisterRequest):
    """
    注册接口: 固定使用 mode="static", speed_factor=1.0
    """
    # 1. 检查是否存在
    if req.id in TIMER_STORE:
        entry = TIMER_STORE[req.id]
        entry.last_accessed = time.time()
        
        logger.info(f"[Register] ID '{req.id}' exists. Refreshed timestamp. Skipped creation.")
        return {
            "status": "exists", 
            "id": req.id, 
            "message": "Timer already active. Skipped creation to preserve state."
        }

    # 2. 不存在，执行创建逻辑 (固定配置: static + 1.0)
    try:
        timer = Timer(mode="static", speed_factor=1.0)
        timer.start()
        
        TIMER_STORE[req.id] = TimerEntry(
            instance=timer,
            last_accessed=time.time(),
            tool_call_times=[]
        )
        
        logger.info(f"[Register] Successfully created timer for ID '{req.id}' with mode='static', speed_factor=1.0")
        return {"status": "created", "id": req.id}
        
    except Exception as e:
        # 【关键】捕获创建失败的异常，并打印堆栈
        logger.error(f"[Register Error] Failed to create timer for ID '{req.id}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize timer: {str(e)}")

@app.post("/add_tool_time")
async def add_tool_time(req: AddToolTimeRequest):
    """
    添加工具调用时间到指定 id 的列表中
    """
    entry = TIMER_STORE.get(req.id)
    
    if not entry:
        logger.warning(f"[AddToolTime Error] Timer ID '{req.id}' not found or expired.")
        raise HTTPException(status_code=404, detail=f"Timer '{req.id}' not found or expired.")
    
    # 续命
    entry.last_accessed = time.time()
    # 添加 tool call 时间
    entry.tool_call_times.append(req.tool_time)
    
    logger.info(f"[AddToolTime] ID '{req.id}' added tool_time={req.tool_time}. Total tool calls: {len(entry.tool_call_times)}")
    return {
        "id": req.id, 
        "status": "added", 
        "tool_time": req.tool_time,
        "total_tool_time": sum(entry.tool_call_times)
    }


@app.post("/call")
async def call_timer(req: CallRequest):
    """
    返回: 真实经过时间 - tool_call_times 之和
    """
    entry = TIMER_STORE.get(req.id)
    
    if not entry:
        # 这是一个常规错误（比如过期的ID），用 warning 级别即可
        logger.warning(f"[Call Error] Timer ID '{req.id}' not found or expired.")
        raise HTTPException(status_code=404, detail=f"Timer '{req.id}' not found or expired.")
    
    # 续命
    entry.last_accessed = time.time()
    
    try:
        # 获取 timer 的真实经过时间 (返回数值格式便于计算)
        real_time = entry.instance.call(return_format="value")
        # 计算 tool call 时间总和
        total_tool_time = sum(entry.tool_call_times)
        # 净时间 = 真实时间 - 工具调用时间
        net_time = real_time - total_tool_time
        
        # 根据请求格式返回
        if req.return_format == "number":
            result = net_time
        else:
            # 默认 text 格式
            result = f"{net_time:.2f}"
        
        return {
            "id": req.id, 
            "result": result,
            "real_time": real_time,
            "total_tool_time": total_tool_time, 
            "existing_tool_call_times": entry.tool_call_times
        }
    except Exception as e:
        # 【关键】捕获 Timer 内部运行时的异常
        logger.error(f"[Call Error] Exception during timer execution for ID '{req.id}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing timer: {str(e)}")

if __name__ == "__main__":
    # log_config=None 让 uvicorn 使用我们上面定义的 logging 配置，或者两者共存
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)
import asyncio
import time
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
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
    mode: str = Field("eval")
    speed_factor: Optional[float] = None
    speed_factor_range: Optional[Tuple[float, float]] = None
    noise_range: Optional[Tuple[float, float]] = None

class CallRequest(BaseModel):
    id: str = Field(..., description="Unique identifier")
    return_format: str = Field("text")

# --- 接口实现 ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "active_timers": len(TIMER_STORE)}

@app.post("/register")
async def register_timer(req: RegisterRequest):
    """
    注册接口
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

    # 2. 不存在，执行创建逻辑
    try:
        init_kwargs = {"mode": req.mode}
        if req.speed_factor is not None:
            init_kwargs["speed_factor"] = req.speed_factor
        if req.speed_factor_range is not None:
            init_kwargs["speed_factor_range"] = req.speed_factor_range
        if req.noise_range is not None:
            init_kwargs["noise_range"] = req.noise_range
        
        timer = Timer(**init_kwargs)
        timer.start()
        
        TIMER_STORE[req.id] = TimerEntry(
            instance=timer,
            last_accessed=time.time()
        )
        
        logger.info(f"[Register] Successfully created timer for ID '{req.id}' with mode '{req.mode}'")
        return {"status": "created", "id": req.id}
        
    except Exception as e:
        # 【关键】捕获创建失败的异常，并打印堆栈
        logger.error(f"[Register Error] Failed to create timer for ID '{req.id}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize timer: {str(e)}")

@app.post("/call")
async def call_timer(req: CallRequest):
    entry = TIMER_STORE.get(req.id)
    
    if not entry:
        # 这是一个常规错误（比如过期的ID），用 warning 级别即可
        logger.warning(f"[Call Error] Timer ID '{req.id}' not found or expired.")
        raise HTTPException(status_code=404, detail=f"Timer '{req.id}' not found or expired.")
    
    # 续命
    entry.last_accessed = time.time()
    
    try:
        result = entry.instance.call(return_format=req.return_format)
        # 成功时不打印日志，因为 call 可能非常高频，打印会影响性能
        return {"id": req.id, "result": result}
    except Exception as e:
        # 【关键】捕获 Timer 内部运行时的异常
        logger.error(f"[Call Error] Exception during timer execution for ID '{req.id}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing timer: {str(e)}")

if __name__ == "__main__":
    # log_config=None 让 uvicorn 使用我们上面定义的 logging 配置，或者两者共存
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
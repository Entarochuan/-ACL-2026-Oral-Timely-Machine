import asyncio
import time
import uvicorn
import logging
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

# --- 尝试导入依赖 ---
try:
    from jericho import FrotzEnv
except ImportError:
    logging.warning("jericho module not found. Using Mock environment.")
    class FrotzEnv:
        def __init__(self, name): self.score = 0; self.moves = 0
        def get_state(self): return b"state"
        def get_valid_actions(self): return ["look", "wait"]
        def get_score(self): return self.score
        def step(self, action): self.moves+=1; return f"Executed {action}", 0, False, {}
        def set_state(self, s): pass
        def get_max_score(self): return 100
        def victory(self): return False
        def game_over(self): return False

try:
    from func_timeout import func_timeout, FunctionTimedOut
except ImportError:
    logging.warning("func_timeout module not found. Running Jericho steps without timeout protection.")
    class FunctionTimedOut(Exception):
        pass

    def func_timeout(timeout, func, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))

try:
    from llm_timer import Timer
except ImportError:
    logging.warning("llm_timer module not found. Using Mock Timer.")
    class Timer:
        def __init__(self, mode="eval", speed_factor=1.0, **kwargs):
            self.start_time = time.time()
            self.speed_factor = speed_factor
        def start(self): self.start_time = time.time()
        def call(self, return_format="value"):
            # 模拟：返回流逝时间 * 系数
            elapsed = (time.time() - self.start_time) * self.speed_factor
            if return_format == "text": return f"{elapsed:.2f}s"
            return elapsed

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("JerichoServer")

# --- 全局配置 ---
TTL_SECONDS = 1800       # 30分钟无操作自动清理
CLEANUP_INTERVAL = 60    # 每60秒检查一次

# 每个 Tool Mode 对应的模拟耗时 (秒)
# 这些时间不乘系数，直接累加
TOOL_DURATION_SETTINGS = dict(
    step=1.0,
    get_available_actions=0.2,
    get_score=0.1,
    get_max_score=0.1,
    check_game_termination=0.1,
    step_back=0.5,
    end_game=0.1,
)
DEFAULT_DURATION = 0.1

# --- 游戏环境封装 ---
class JerichoToolEnvironment:
    """Jericho Tool for Agentic LLMs (Stateful Wrapper)"""
    def __init__(self, env_name: str):
        self.env_name = env_name
        try:
            # 根据你的环境修改这里的路径前缀
            local_sources_dir = "./jericho_game_sources/jericho-game-suite"
            if not os.path.isabs(env_name) and not env_name.startswith(local_sources_dir):
                full_path = os.path.join(local_sources_dir, env_name)
            else:
                full_path = env_name
            
            # 检查文件是否存在，不存在尝试直接加载(可能是Mock或绝对路径)
            # if os.path.exists(full_path):
            #     self.env = FrotzEnv(full_path)
            # else:
            #     self.env = FrotzEnv(env_name)

            if os.path.exists(full_path):
                self.env = SafeFrotzEnv(full_path)
            else:
                self.env = SafeFrotzEnv(env_name)
                
        except Exception as e:
            raise ValueError(f"Failed to load game rom '{env_name}': {e}")
            
        initial_state = self.env.get_state()
        self.state_history = [initial_state]
        self.current_index = 0
        
    def get_valid_actions(self):
        return self.env.get_valid_actions()
    
    def get_score(self):
        return f"Your current score is: {self.env.get_score()}."

    def step(self, action: str):
        textual_response, immediate_reward, done, info = self.env.step(action)
        new_state = self.env.get_state()
        if self.current_index < len(self.state_history) - 1:
            self.state_history = self.state_history[:self.current_index + 1]
        self.state_history.append(dict(
            textual_response=textual_response, immediate_reward=immediate_reward,
            done=done, info=info, state=new_state
        ))
        self.current_index += 1
        
        response = f"The response is: {textual_response}\nThe step reward is: {immediate_reward}."
        if done:
            response += f"\nThe game is terminated. Final score: {self.env.get_score()}."
        return response

    def step_back(self):
        if self.current_index <= 0: return "Cannot step back. Index is 0."
        self.current_index -= 1
        new_state = self.state_history[self.current_index]
        state_val = new_state['state'] if isinstance(new_state, dict) else new_state
        self.env.set_state(state_val)
        resp = new_state.get('textual_response', 'Start') if isinstance(new_state, dict) else "Start"
        return f"Stepped back 1 step. Response: {resp}. Score: {self.env.get_score()}."

    def get_max_score(self): return f"Max score is {self.env.get_max_score()}."
    
    def check_game_termination(self):
        if self.env.victory(): return "Terminated. You win."
        if self.env.game_over(): return "Terminated. You lose."
        return "Not terminated."
    
    def end_game(self): return f"Game ended. Final score: {self.env.get_score()}."

# --- 数据结构封装 ---
@dataclass
class SessionEntry:
    game_instance: JerichoToolEnvironment
    timer_instance: Timer
    simulated_accumulated_time: float # 累积的模拟时间（不乘系数）
    last_accessed: float

# 全局存储: { "session_id": SessionEntry }
SESSION_STORE: Dict[str, SessionEntry] = {}

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    cleanup_task = asyncio.create_task(background_cleanup())
    yield
    logger.info("Server shutting down...")
    cleanup_task.cancel()

app = FastAPI(title="Jericho & Timer Server", lifespan=lifespan)

# --- 后台清理逻辑 ---
async def background_cleanup():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = time.time()
            keys_to_remove = [
                sid for sid, entry in SESSION_STORE.items() 
                if now - entry.last_accessed > TTL_SECONDS
            ]
            for k in keys_to_remove:
                del SESSION_STORE[k]
            if keys_to_remove:
                logger.info(f"[Cleanup] Removed {len(keys_to_remove)} sessions.")
        except asyncio.CancelledError: break
        except Exception as e: logger.error(f"[Cleanup] Error: {e}")

# --- API Models ---
class RegisterRequest(BaseModel):
    id: str = Field(..., description="Unique session identifier")
    env_name: str = Field("zork1.z5", description="Game ROM path")
    # Timer Params
    timer_mode: str = Field("eval", description="Timer mode")
    speed_factor: float = Field(1.0, description="Real time speed multiplier")
    speed_factor_range: Optional[Tuple[float, float]] = None
    noise_range: Optional[Tuple[float, float]] = None

class CommandRequest(BaseModel):
    id: str
    mode: str
    action: Optional[str] = None

# --- API 实现 ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(SESSION_STORE)}

@app.post("/register")
async def register(req: RegisterRequest):
    """同时初始化游戏环境和 Timer"""
    if req.id in SESSION_STORE:
        SESSION_STORE[req.id].last_accessed = time.time()
        return {"status": "exists", "id": req.id, "message": "Session refreshed."}

    try:
        # 1. 初始化游戏
        logger.info(f"[Register] Init Game: {req.id} -> {req.env_name}")
        game = JerichoToolEnvironment(req.env_name)
        
        # 2. 初始化 Timer
        logger.info(f"[Register] Init Timer: {req.id} (Factor: {req.speed_factor})")
        timer_kwargs = {"mode": req.timer_mode, "speed_factor": req.speed_factor}
        if req.speed_factor_range: timer_kwargs["speed_factor_range"] = req.speed_factor_range
        if req.noise_range: timer_kwargs["noise_range"] = req.noise_range
        
        timer = Timer(mode=req.timer_mode, speed_factor=req.speed_factor)
        timer.start()
        
        # 3. 存储
        SESSION_STORE[req.id] = SessionEntry(
            game_instance=game,
            timer_instance=timer,
            simulated_accumulated_time=0.0,
            last_accessed=time.time()
        )
        return {"status": "created", "id": req.id}
        
    except Exception as e:
        logger.error(f"[Register Error] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute(req: CommandRequest):
    """执行游戏指令并附带时间"""
    entry = SESSION_STORE.get(req.id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    # 续命
    entry.last_accessed = time.time()
    
    try:
        # 1. 执行游戏逻辑
        result_text = None
        if req.mode == "step":
            if not req.action: raise ValueError("Action required for step.")
            result_text = entry.game_instance.step(req.action)
        elif req.mode == "get_available_actions":
            result_text = entry.game_instance.get_valid_actions() # returns list
        elif req.mode == "get_score":
            result_text = entry.game_instance.get_score()
        elif req.mode == "get_max_score":
            result_text = entry.game_instance.get_max_score()
        elif req.mode == "step_back":
            result_text = entry.game_instance.step_back()
        elif req.mode == "end_game":
            result_text = entry.game_instance.end_game()
        else:
            raise ValueError(f"Unknown mode: {req.mode}")
            
        # 2. 计算并拼接时间
        # 逻辑：Total = (Real_Elapsed * Factor) + Accumulated_Simulated
        
        # 2.1 获取当前 Tool 的固定模拟时长
        sim_duration = TOOL_DURATION_SETTINGS.get(req.mode, DEFAULT_DURATION)
        
        # 2.2 更新累积时间
        entry.simulated_accumulated_time += sim_duration
        
        # 2.3 获取 Timer 的计算结果 (真实流逝 * 系数)
        # 注意：这里我们只取 value，不需要它自带的格式化
        timer_real_scaled = entry.timer_instance.call(return_format="value")
        
        # 2.4 计算最终展示时间
        final_total_seconds = timer_real_scaled + entry.simulated_accumulated_time
        
        # 2.5 格式化并追加
        # time_str = f"\n[Duration: {final_total_seconds:.2f}s]"
        time_str = f"\nYou have played for {final_total_seconds:.2f} seconds."
        
        if isinstance(result_text, list):
            # 如果是列表，先转字符串再拼接，或者可以包装成一个 dict 返回
            # 这里按照通常 Agent 的做法，直接转 string
            final_result = json.dumps(result_text) + time_str
        else:
            final_result = str(result_text) + time_str
            
        return {
            "id": req.id, 
            "mode": req.mode, 
            "result": final_result,
            # 返回一些元数据供调试
            "debug_info": {
                "real_scaled": timer_real_scaled,
                "simulated_added": entry.simulated_accumulated_time,
                "this_tool_cost": sim_duration
            }
        }

    except Exception as e:
        logger.error(f"[Execute Error] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# from jericho import FrotzEnv
# from func_timeout import func_timeout, FunctionTimedOut
# import logging

class SafeFrotzEnv:
    """
    一个 FrotzEnv 的安全包装器。
    用法完全兼容 FrotzEnv，但在底层崩溃时会自动回档并重启，
    确保 Server 永远不会因为 'Emulator halted' 而卡死。
    """
    def __init__(self, rom_path, seed=None):
        self.rom_path = rom_path
        self.seed = seed
        self.env = None
        self._start_env()
        # 初始化时保存一个快照，以防第一步就崩
        self.snapshot = self.env.get_state()

    def _start_env(self):
        """内部方法：启动或重启底层环境"""
        if self.env:
            try:
                self.env.close()
            except:
                pass
        self.env = FrotzEnv(self.rom_path, seed=self.seed)

    def step(self, action):
        """
        覆盖原有的 step 方法，增加超时保护和自动回滚。
        """
        # 1. [备份] 在执行动作前，先保存当前状态 (存档)
        try:
            current_state = self.env.get_state()
            self.snapshot = current_state
        except Exception:
            # 如果连获取状态都报错，说明已经是坏的了，尝试用旧存档恢复
            pass

        try:
            # 2. [执行] 尝试执行动作，限制 2 秒超时
            # Jericho 动作通常只需 0.0x 秒，2秒足够判断是否卡死
            obs, reward, done, info = func_timeout(15, self.env.step, args=(action,))
            return obs, reward, done, info

        except (FunctionTimedOut, Exception) as e:
            # 3. [救援] 如果超时或报错 (Emulator halted)
            print(f"⚠️ [SafeEnv] 检测到环境崩溃/卡死 (Action: '{action}'). 正在回滚...")
            
            # A. 销毁并重建环境
            self._start_env()
            
            # B. 恢复到崩溃前的状态 (读档)
            if self.snapshot:
                self.env.set_state(self.snapshot)
            
            # C. 伪造一个返回结果，骗过调用者，让程序继续运行
            # 告诉模型：这个动作没发生任何事（因为被回滚了）
            fake_obs = "Nothing happened, or the action was physically impossible. Please try another action." 
            fake_reward = 0
            fake_done = False
            fake_info = {"error": "emulator_crash_recovered", "msg": str(e)}
            
            return fake_obs, fake_reward, fake_done, fake_info

    def __getattr__(self, name):
        """
        魔法方法：将所有其他调用 (get_score, get_valid_world_actions 等)
        直接转发给底层的 self.env，确保兼容性 100%
        """
        return getattr(self.env, name)

if __name__ == "__main__":
    # 使用 8001 端口
    uvicorn.run(app, host="0.0.0.0", port=8001)

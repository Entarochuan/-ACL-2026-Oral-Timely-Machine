import time
import random


class Timer:
    """计时器，支持eval、static和dynamic三种模式"""
    
    def __init__(self, mode='eval', 
        speed_factor=1.0,
        speed_factor_range=(0.5, 2.0),
        noise_range=(0.99, 1.01)
    ):
        
        """
        Args:
            mode: 'eval' 或 'static' 或 'dynamic'
            speed_factor: 加速系数（仅在static模式下有效）
            speed_factor_range: 加速系数范围（仅在dynamic模式下有效）
            noise_range: 随机扰动范围（仅在训练模式下有效）
        """ 
        
        assert mode in ['eval', 'static', 'dynamic'], "mode must be 'eval' or 'static' or 'dynamic'"
        assert isinstance(speed_factor, float) and speed_factor > 0, "speed_factor must be a positive float"
        assert isinstance(noise_range, tuple) and len(noise_range) == 2 and noise_range[0] < noise_range[1], "noise_range must be a tuple of two positive floats with the first less than the second"
        
        self.mode = mode
        self.speed_factor = speed_factor
        self.speed_factor_range = speed_factor_range
        self.start_time = None
        self.noise_range = noise_range
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        
    def elapsed(self, return_format='text'):
        """
        获取已过时间
        
        Args:
            format: 'text' 返回文字描述，'value' 返回数值
            
        Returns:
            文字描述或浮点数
        """
        
        if self.start_time is None:
            raise ValueError("Timer not started")
            
        real_elapsed = time.time() - self.start_time
        
        if self.mode == 'eval':
            elapsed = real_elapsed
            
        elif self.mode == 'static': # 固定时间加速比
            real_elapsed = real_elapsed * self.speed_factor
            noise = random.uniform(self.noise_range[0], self.noise_range[1])
            elapsed = real_elapsed * noise
            
        elif self.mode == 'dynamic': # 动态时间加速比
            speed_factor = random.uniform(self.speed_factor_range[0], self.speed_factor_range[1])
            noise = random.uniform(self.noise_range[0], self.noise_range[1])
            elapsed = real_elapsed * speed_factor * noise
            
        if return_format == 'text':
            return f"{elapsed:.2f} seconds."
        elif return_format == 'value':
            return elapsed
        else:
            raise ValueError("return_format must be 'text' or 'value'")
            
    def call(self, return_format='text'):
        """call timer"""
        return self.elapsed(return_format=return_format)

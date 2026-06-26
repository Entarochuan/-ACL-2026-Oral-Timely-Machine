"""Timer utility exposed to time-aware agents."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Literal

TimerMode = Literal["eval", "static", "dynamic"]
ReturnFormat = Literal["text", "value"]


@dataclass(slots=True)
class Timer:
    """Measure elapsed time with optional static or dynamic scaling."""

    mode: TimerMode = "eval"
    speed_factor: float = 1.0
    speed_factor_range: tuple[float, float] = (0.5, 2.0)
    noise_range: tuple[float, float] = (0.99, 1.01)
    start_time: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"eval", "static", "dynamic"}:
            raise ValueError("mode must be 'eval', 'static', or 'dynamic'")
        if self.speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        if self.noise_range[0] >= self.noise_range[1]:
            raise ValueError("noise_range must be an increasing pair")
        self.start_time = None

    def start(self) -> None:
        self.start_time = time.time()

    def elapsed(self, return_format: ReturnFormat = "text") -> str | float:
        if self.start_time is None:
            raise ValueError("Timer has not been started")

        real_elapsed = time.time() - self.start_time
        if self.mode == "eval":
            elapsed = real_elapsed
        elif self.mode == "static":
            elapsed = real_elapsed * self.speed_factor * self._noise()
        else:
            elapsed = real_elapsed * random.uniform(*self.speed_factor_range) * self._noise()

        if return_format == "text":
            return f"{elapsed:.2f} seconds."
        if return_format == "value":
            return elapsed
        raise ValueError("return_format must be 'text' or 'value'")

    def call(self, return_format: ReturnFormat = "text") -> str | float:
        return self.elapsed(return_format=return_format)

    def _noise(self) -> float:
        return random.uniform(*self.noise_range)

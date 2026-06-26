# LLM-Timer

A timer designed for LLM tool call.

## Installation

```bash
pip install -e .
```

## Usage

```python

from llm_timer import Timer
import time

print("=== Eval mode - real time ===")
timer = Timer(mode='eval')
timer.start()
time.sleep(1)
result = timer.call()  # Returns text
value = timer.call(return_format='value')  # Returns float
print(f"Text: {result}")
print(f"Value: {value}")

print("\n=== Static mode - fixed speed factor ===")
timer = Timer(mode='static', speed_factor=2.0)
timer.start()
time.sleep(1)
result = timer.call()
value = timer.call(return_format='value')
print(f"Text: {result}")
print(f"Value: {value}")

print("\n=== Dynamic mode - random speed factor ===")
timer = Timer(mode='dynamic', speed_factor_range=(0.5, 2.0), noise_range=(0.9, 1.1))
timer.start()
time.sleep(1)
result = timer.call()
value = timer.call(return_format='value')
print(f"Text: {result}")
print(f"Value: {value}")

```
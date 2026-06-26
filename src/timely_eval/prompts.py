"""Prompt templates and XML tool specifications used by Timely Eval."""

from __future__ import annotations

import json
from typing import Any

TIME_AWARE_GENERAL_SYSTEM = """Act as a Time-Aware Strategic Reasoner. Your objective is to solve complex reasoning tasks within a strictly enforced time window.

Treat time as a critical resource:
1. Initial Assessment: quickly estimate the problem complexity against the remaining time.
2. Cognitive Budgeting: allocate time for analysis, tool use, and final synthesis.
3. Dynamic Adjustment: if a path is too costly, switch to a faster heuristic or alternative.
4. Convergence: always provide a complete final answer before the deadline.

Output requirements:
- Put concise final reasoning in <summary>...</summary>.
- Put the final answer in <answer>...</answer>, for example <answer>\\boxed{100}</answer> or <answer>B</answer>.
"""

JUDGE_SYSTEM_PROMPT = """You judge whether a model answer is correct.

Given the question, the model generated answer, and the reference answer, reply in exactly this format:
<think>
Briefly compare the generated answer with the reference answer.
</think>
<judge_response>
yes/no
</judge_response>
"""

AGENTIC_ML_SYSTEM = """Act as a Time-Aware Strategic Reasoner for machine learning tasks.

Your goal is to build a valid solution under a strict time window. Prefer reliable, fast baselines over slow or fragile experiments when time is limited.

Write Python code in a fenced ```python block before calling the execution tool. The code must create ./submission.csv in the current working directory.

Output requirements:
- Record elapsed duration as <conclusion>total duration: {time} seconds</conclusion>.
- Summarize the best observed accuracy as <accuracy>{float between 0 and 1}</accuracy>.
"""

INTERACTIVE_SYSTEM = """Act as a Time-Aware Strategic Reasoner for an interactive text game.

Use tools to inspect and act in the environment. Balance exploration against the time limit, check score when useful, and end the game when you decide the current state is final.

Output requirements:
- Record elapsed duration as <conclusion>total duration: {time} seconds</conclusion>.
- Summarize final score as <score>{integer}</score>.
"""

TIME_TOOL = {
    "name": "get_duration",
    "description": "Get the total elapsed time in seconds since the start of the current request.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

AGENTIC_ML_TOOL = {
    "name": "execute_code_and_get_duration",
    "description": "Run the Python code in the response, evaluate submission.csv, and return elapsed duration.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. It must write ./submission.csv.",
            }
        },
        "required": ["code"],
    },
}

INTERACTIVE_TOOLS = [
    {
        "name": "step",
        "description": "Execute an action in the game world and return the textual response and immediate reward.",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "description": "Text command to execute."}},
            "required": ["action"],
        },
    },
    {
        "name": "get_available_actions",
        "description": "Return valid actions in the current state.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_score",
        "description": "Return the current game score.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_max_score",
        "description": "Return the maximum possible game score.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "end_game",
        "description": "Terminate the current game session.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def build_tool_prompt(tools: list[dict[str, Any]]) -> str:
    tool_lines = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return f"""# Tools

You may call one or more functions to assist with the user query.

Function signatures are provided inside <tools></tools>:
<tools>
{tool_lines}
</tools>

For each function call, return a JSON object with function name and arguments inside <tool_call></tool_call>:
<tool_call>
{{"name": "<function-name>", "arguments": {{}}}}
</tool_call>"""


TIME_TOOL_PROMPT = build_tool_prompt([TIME_TOOL])
AGENTIC_ML_TOOL_PROMPT = build_tool_prompt([AGENTIC_ML_TOOL])
INTERACTIVE_TOOL_PROMPT = build_tool_prompt(INTERACTIVE_TOOLS)

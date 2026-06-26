from timely_eval.parsing import (
    extract_answer_or_tool_call,
    extract_judge_result,
    extract_python_code,
    extract_tool_calls,
)


def test_extract_answer_or_tool_call_prefers_last_values() -> None:
    text = """
    <tool_call>{"name": "get_duration", "arguments": {}}</tool_call>
    <answer>wrong</answer>
    <answer>42</answer>
    """
    has_tool, has_answer, tool_call, answer = extract_answer_or_tool_call(text)
    assert has_tool is True
    assert has_answer is True
    assert "get_duration" in tool_call
    assert answer == "42"


def test_extract_tool_calls_json_and_xml() -> None:
    text = """
    <tool_call>{"name": "step", "arguments": {"action": "look"}}</tool_call>
    <tool_call><name>get_score</name><arguments>{}</arguments></tool_call>
    """
    calls = extract_tool_calls(text)
    assert calls == [
        {"name": "step", "arguments": {"action": "look"}},
        {"name": "get_score", "arguments": {}},
    ]


def test_extract_python_code() -> None:
    text = "```python\nprint('ok')\n```"
    assert extract_python_code(text) == "print('ok')"


def test_extract_judge_result() -> None:
    assert extract_judge_result("<judge_response>\nyes\n</judge_response>") == "yes"

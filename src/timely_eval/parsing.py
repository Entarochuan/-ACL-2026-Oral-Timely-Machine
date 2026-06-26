"""Parsing helpers for XML-style model outputs."""

from __future__ import annotations

import json
import re
from typing import Any

TAG_RE_TEMPLATE = r"<{tag}>\s*(.*?)\s*</{tag}>"
PYTHON_CODE_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_tag_values(text: str, tag: str) -> list[str]:
    pattern = re.compile(TAG_RE_TEMPLATE.format(tag=re.escape(tag)), re.DOTALL)
    return [match.strip() for match in pattern.findall(text or "")]


def extract_last_tag(text: str, tag: str) -> str | None:
    values = extract_tag_values(text, tag)
    return values[-1] if values else None


def extract_answer_or_tool_call(response_text: str) -> tuple[bool, bool, str | None, str | None]:
    """Return ``has_tool_call, has_answer, tool_call, answer`` from a response."""

    tool_calls = extract_tag_values(response_text, "tool_call")
    answers = extract_tag_values(response_text, "answer")
    return bool(tool_calls), bool(answers), tool_calls[-1] if tool_calls else None, answers[-1] if answers else None


def extract_judge_result(response_text: str) -> str | None:
    value = extract_last_tag(response_text, "judge_response")
    return value.lower() if value else None


def extract_python_code(response_text: str) -> str | None:
    match = PYTHON_CODE_RE.search(response_text or "")
    return match.group(1).strip() if match else None


def parse_tool_call_content(content: str) -> dict[str, Any] | None:
    """Parse JSON or simple XML content inside ``<tool_call>``."""

    content = (content or "").strip()
    if not content:
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "name" in parsed:
            parsed.setdefault("arguments", {})
            return parsed
    except json.JSONDecodeError:
        pass

    name_match = re.search(r"<name>\s*(.*?)\s*</name>", content, re.DOTALL)
    args_match = re.search(r"<arguments>\s*(.*?)\s*</arguments>", content, re.DOTALL)
    if not name_match:
        return None

    args: dict[str, Any] = {}
    if args_match:
        args_text = args_match.group(1).strip()
        if args_text:
            try:
                loaded = json.loads(args_text)
                if isinstance(loaded, dict):
                    args = loaded
            except json.JSONDecodeError:
                args = {}

    return {"name": name_match.group(1).strip(), "arguments": args}


def extract_tool_calls(response_text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for content in extract_tag_values(response_text, "tool_call"):
        parsed = parse_tool_call_content(content)
        if parsed:
            calls.append(parsed)
    return calls

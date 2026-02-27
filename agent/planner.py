"""
Call 1: Planning agent.

Takes the feature description + retrieved candidate functions + project overview
and outputs a structured implementation plan:
  - affected files
  - functions to modify / add
  - new files needed
  - step-by-step implementation plan
  - complexity level (medium/high)
"""

import json
import os
import re

from llm.client import LLMClient


SYSTEM_PROMPT_TEMPLATE = """\
你是一名资深 C 语言数据库内核工程师。
你的任务是分析开发者提出的特性需求，制定精确的代码修改计划。

# 项目架构概述
{project_overview}

# 关键调用链
{callchains}
"""

USER_PROMPT_TEMPLATE = """\
## 特性需求描述
{feature_desc}

## 候选函数列表（经过相关性过滤，按相关度排序）
{candidates}

## 任务
根据特性描述和候选函数，制定实现计划。

输出格式（严格 JSON，不包含任何 markdown 代码块标记）：
{{
  "complexity": "medium 或 high（影响范围跨 3+ 模块或涉及核心接口变更时为 high）",
  "affected_files": ["文件名.c（只写文件名，不含目录路径）", ...],
  "functions_to_modify": ["已有函数名（最多列出10个最核心的）", ...],
  "functions_to_add": [
    {{"name": "新函数名", "in_file": "文件名.c（只写文件名）"}}
  ],
  "new_files": ["新文件名.c（若无则为空列表）"],
  "header_changes": ["头文件名.h（只写文件名，若无则为空列表）"],
  "implementation_plan": "分步骤的实现说明，清晰描述每步修改的位置和内容"
}}

注意：
- affected_files 只写文件名（如 build.c），不含任何目录路径
- functions_to_modify 最多列出10个最关键的函数，不要重复
- 若修改了函数签名，必须在 header_changes 中列出对应头文件
- implementation_plan 要具体到函数级别
"""


def _strip_src_prefix(path: str) -> str:
    """Remove leading src/ or src\\ directory prefix from file paths."""
    path = path.replace("\\", "/")
    if path.startswith("src/"):
        path = path[4:]
    return path


def _parse_json_response(text: str) -> dict:
    """
    Extract and parse the JSON plan from the LLM response.
    Handles markdown fences, truncated JSON, and other common issues.
    """
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first complete { } block
    # Use a bracket-matching approach for robustness
    start = text.find('{')
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break

    # Last resort: try to fix truncated JSON by closing open structures
    fragment = text[start:]
    # Try progressively shorter fragments
    for end in range(len(fragment), max(10, len(fragment) - 500), -10):
        candidate = fragment[:end]
        # Try to close the JSON
        open_braces = candidate.count('{') - candidate.count('}')
        open_brackets = candidate.count('[') - candidate.count(']')
        if open_braces >= 0 and open_brackets >= 0:
            # Trim to last complete value (before any trailing comma)
            candidate = re.sub(r',\s*$', '', candidate.rstrip())
            closing = ']' * open_brackets + '}' * open_braces
            try:
                return json.loads(candidate + closing)
            except json.JSONDecodeError:
                continue

    return {}


def _normalize_plan(plan: dict) -> dict:
    """
    Post-process the plan dict to fix common LLM output issues:
    - Strip src/ prefix from file paths
    - Deduplicate functions_to_modify
    - Limit list sizes to avoid noise
    """
    if "affected_files" in plan:
        plan["affected_files"] = list(dict.fromkeys(
            _strip_src_prefix(f) for f in plan["affected_files"]
        ))

    if "functions_to_modify" in plan:
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for fn in plan["functions_to_modify"]:
            if fn not in seen:
                seen.add(fn)
                deduped.append(fn)
        plan["functions_to_modify"] = deduped[:20]  # cap at 20

    if "functions_to_add" in plan:
        for fn_add in plan["functions_to_add"]:
            if "in_file" in fn_add:
                fn_add["in_file"] = _strip_src_prefix(fn_add["in_file"])

    if "new_files" in plan:
        plan["new_files"] = [_strip_src_prefix(f) for f in plan["new_files"]]

    if "header_changes" in plan:
        plan["header_changes"] = list(dict.fromkeys(
            _strip_src_prefix(f) for f in plan["header_changes"]
        ))

    return plan


def plan_feature(
    feature_desc: str,
    candidates_text: str,
    project_overview: str,
    callchains_text: str,
    client: LLMClient,
    model: str = "",
) -> dict:
    """
    Call the LLM to produce an implementation plan.

    Args:
        feature_desc: user's feature description
        candidates_text: formatted candidate functions (from layer3_callgraph.format_candidates_for_prompt)
        project_overview: contents of project_overview.md
        callchains_text: formatted call chains (from callchain_builder.format_callchains_for_prompt)
        client: LLMClient instance
        model: model to use (empty string = use client default)

    Returns:
        Parsed plan dict with keys: complexity, affected_files, functions_to_modify,
        functions_to_add, new_files, header_changes, implementation_plan
    """
    model = model or client.default_model("main")

    system = SYSTEM_PROMPT_TEMPLATE.format(
        project_overview=project_overview,
        callchains=callchains_text,
    )
    user = USER_PROMPT_TEMPLATE.format(
        feature_desc=feature_desc,
        candidates=candidates_text,
    )

    raw = client.chat(model=model, system=system, user=user, max_tokens=8192)
    plan = _parse_json_response(raw)

    if not plan:
        print(f"[warn] planner JSON parse failed. Raw response snippet: {raw[:300]!r}")
        # Fallback: return raw text wrapped in a minimal structure
        return {
            "complexity": "unknown",
            "affected_files": [],
            "functions_to_modify": [],
            "functions_to_add": [],
            "new_files": [],
            "header_changes": [],
            "implementation_plan": raw,
            "_parse_error": True,
        }

    return _normalize_plan(plan)

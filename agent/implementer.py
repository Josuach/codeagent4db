"""
Call 2 (and optional Call 3): Implementation agent.

Takes the plan from Call 1 + full contents of affected files and generates
unified diff patches for all required changes.

For high-complexity features, a second implementation call handles header
files and cross-module interface consistency.
"""

import os
from typing import Optional

from llm.client import LLMClient

# Maximum characters per file in the implementation prompt.
# For large files (btree.c = 11K lines, vdbe.c = 9K lines), we extract only
# the relevant function sections rather than loading the full file.
MAX_FILE_CHARS = 30000


SYSTEM_PROMPT_IMPL = """\
你是一名资深 C 语言数据库内核工程师。
你的任务是根据实现计划，对提供的 C 源文件进行精确修改。

输出规范：
- 只输出 unified diff 格式的修改（--- a/... +++ b/... @@ ... @@），不输出任何解释文字
- 每个文件的修改用一个完整的 diff 块表示
- 新增文件用 --- /dev/null 和 +++ b/新文件路径 的格式
- diff 内容必须语法正确，可以直接用 patch -p1 应用
- 如果某个文件不需要修改，不要输出它的 diff
"""

SYSTEM_PROMPT_INTEGRATE = """\
你是一名资深 C 语言数据库内核工程师。
你的任务是审查已生成的代码修改，确保头文件声明和跨模块接口的一致性。

输出规范：
- 只输出 unified diff 格式，只包含头文件（.h）和接口相关的修改
- 不要重复 Call 2 已经生成的修改
- 如果接口已经正确，输出空字符串
"""

USER_PROMPT_IMPL_TEMPLATE = """\
## 实现计划
{implementation_plan}

## 相关结构体 / 联合体 / 枚举定义
{structs_content}

## 受影响文件的完整内容
{files_content}

请根据实现计划修改上述文件，只输出 unified diff。
"""

USER_PROMPT_INTEGRATE_TEMPLATE = """\
## 实现计划
{implementation_plan}

## 已生成的代码修改（Call 2 的输出）
{existing_diff}

## 头文件内容
{headers_content}

请检查上述修改，补充必要的头文件声明和跨模块接口调整，只输出 unified diff。
如果无需调整，输出空字符串。
"""


def _load_file_content(
    file_path: str,
    project_root: str,
    relevant_funcs: Optional[list[str]] = None,
    function_index: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Load a file and return (relative_path, content).
    For large files, extract only the relevant function sections to avoid
    exceeding the LLM context window.

    Args:
        file_path: relative path from project root
        project_root: absolute path to project root
        relevant_funcs: function names relevant to this file (from plan)
        function_index: function index for locating functions by line number

    Returns:
        (relative_path, content) — content may be truncated for large files
    """
    full_path = os.path.join(project_root, file_path)
    if not os.path.exists(full_path):
        return file_path, ""
    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # If file is small enough, return as-is
    if len(content) <= MAX_FILE_CHARS:
        return file_path, content

    # Large file: extract relevant sections
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    # Try to find relevant function locations from function_index
    sections_to_include: list[tuple[int, int]] = []  # (start, end) 0-indexed line numbers
    if relevant_funcs and function_index:
        for func_name in relevant_funcs:
            info = function_index.get(func_name, {})
            if info.get("file", "").replace("\\", "/") == file_path.replace("\\", "/"):
                start = max(0, info.get("start_line", 1) - 1 - 10)  # 10 lines before
                end = min(total_lines, info.get("end_line", info.get("start_line", 1)) + 5)
                sections_to_include.append((start, end))

    if not sections_to_include:
        # No specific functions found; include beginning (includes, struct defs) + first MAX_FILE_CHARS chars
        header_end = min(200, total_lines)  # first 200 lines usually has includes/structs
        sections_to_include = [(0, header_end)]

    # Merge overlapping sections and sort
    sections_to_include.sort()
    merged: list[tuple[int, int]] = []
    for s, e in sections_to_include:
        if merged and s <= merged[-1][1] + 5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Extract sections and join with ellipsis markers
    parts = [
        f"/* ... (lines 1-{merged[0][0]}) ... */\n" if merged[0][0] > 0 else ""
    ]
    for i, (s, e) in enumerate(merged):
        parts.append("".join(lines[s:e]))
        if i + 1 < len(merged):
            parts.append(f"\n/* ... (lines {e+1}-{merged[i+1][0]}) ... */\n")
        elif e < total_lines:
            parts.append(f"\n/* ... (lines {e+1}-{total_lines}, {total_lines - e} more lines) ... */\n")

    extracted = "".join(parts)

    # If still too large, truncate with a note
    if len(extracted) > MAX_FILE_CHARS:
        extracted = extracted[:MAX_FILE_CHARS] + f"\n/* ... truncated (file has {total_lines} lines total) ... */"

    return file_path, f"/* NOTE: Large file ({total_lines} lines). Showing relevant sections only. */\n" + extracted


def _format_files_block(files: list[tuple[str, str]]) -> str:
    """Format list of (path, content) into a block for the prompt."""
    parts = []
    for path, content in files:
        if not content:
            parts.append(f"=== {path} (文件不存在，需要新建) ===\n(空文件)")
        else:
            parts.append(f"=== {path} ===\n{content}")
    return "\n\n".join(parts)


def _build_structs_block(plan: dict, struct_index: Optional[dict]) -> str:
    """
    Build a formatted block of struct/union/enum definitions for the prompt.

    Uses two sources of struct names:
      1. plan["relevant_structs"] — as selected by the planner
      2. Struct names that appear as whole-word tokens in the implementation_plan text
         (catches structs the planner knew about but didn't list explicitly)
    """
    if not struct_index:
        return ""

    import re as _re

    # Gather names from the plan
    names: list = list(plan.get("relevant_structs", []))

    # Supplement with names found in the implementation_plan text
    plan_text = plan.get("implementation_plan", "")
    for name in struct_index:
        if name not in names and _re.search(r'\b' + _re.escape(name) + r'\b', plan_text):
            names.append(name)

    if not names:
        return ""

    parts = []
    for name in names:
        info = struct_index.get(name)
        if not info:
            continue
        parts.append(
            f"/* {info['kind']} {name}  ({info['file']}:{info['start_line']}) */\n"
            f"{info['body']}"
        )
    return "\n\n".join(parts)


def implement_feature(
    plan: dict,
    project_root: str,
    client: LLMClient,
    model: str = "",
    function_index: Optional[dict] = None,
    struct_index: Optional[dict] = None,
) -> str:
    """
    Call 2: Generate unified diffs for all affected source files.

    Args:
        plan: output from planner.plan_feature()
        project_root: absolute path to the C project root
        client: LLMClient instance
        model: model to use (empty string = use client default)
        function_index: optional function index for smart large-file extraction

    Returns:
        Unified diff string (may be empty if no changes needed)
    """
    model = model or client.default_model("main")
    # Collect all affected .c files (not headers — those are handled in Call 3)
    source_files = [f for f in plan.get("affected_files", []) if not f.endswith(".h")]
    # Also include files that need new functions
    for fn_add in plan.get("functions_to_add", []):
        target_file = fn_add.get("in_file", "")
        if target_file and target_file not in source_files:
            source_files.append(target_file)

    # Build per-file function lists for smart large-file extraction
    funcs_to_modify = plan.get("functions_to_modify", [])
    funcs_to_add = [fn.get("name", "") for fn in plan.get("functions_to_add", [])]
    all_plan_funcs = funcs_to_modify + funcs_to_add

    def _relevant_funcs_for_file(file_path: str) -> list[str]:
        if not function_index:
            return all_plan_funcs
        return [
            fn for fn in all_plan_funcs
            if function_index.get(fn, {}).get("file", "").replace("\\", "/")
               == file_path.replace("\\", "/")
        ] or all_plan_funcs  # fallback: pass all if no match found

    files_content = [
        _load_file_content(
            f, project_root,
            relevant_funcs=_relevant_funcs_for_file(f),
            function_index=function_index,
        )
        for f in source_files
    ]
    files_block = _format_files_block(files_content)

    # Build struct definitions block
    structs_block = _build_structs_block(plan, struct_index)

    user_msg = USER_PROMPT_IMPL_TEMPLATE.format(
        implementation_plan=plan.get("implementation_plan", ""),
        structs_content=structs_block or "(none)",
        files_content=files_block,
    )

    return client.chat(model=model, system=SYSTEM_PROMPT_IMPL, user=user_msg, max_tokens=8192).strip()


def integrate_interfaces(
    plan: dict,
    existing_diff: str,
    project_root: str,
    client: LLMClient,
    model: str = "",
) -> str:
    """
    Call 3 (for high-complexity features only): Check header files and
    cross-module interface consistency.

    Args:
        plan: output from planner.plan_feature()
        existing_diff: the diff produced by implement_feature() (Call 2)
        project_root: absolute path to the C project root
        client: LLMClient instance
        model: model to use (empty string = use client default)

    Returns:
        Additional unified diff for header changes (may be empty)
    """
    model = model or client.default_model("main")
    header_files = plan.get("header_changes", [])
    if not header_files:
        return ""

    headers_content = [_load_file_content(h, project_root) for h in header_files]
    headers_block = _format_files_block(headers_content)

    user_msg = USER_PROMPT_INTEGRATE_TEMPLATE.format(
        implementation_plan=plan.get("implementation_plan", ""),
        existing_diff=existing_diff,
        headers_content=headers_block,
    )

    result = client.chat(model=model, system=SYSTEM_PROMPT_INTEGRATE, user=user_msg, max_tokens=4096).strip()
    # If LLM says "no changes needed" in prose, return empty
    if len(result) < 20 and not result.startswith("---"):
        return ""
    return result

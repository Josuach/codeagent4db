"""
Call 2 (and optional Call 3): Implementation agent.

Takes the plan from Call 1 and generates unified diff patches, one file at a
time.  Each LLM call handles exactly one source or header file so that:
  - The output size per call is small and predictable (no truncation).
  - Partial failures affect only the current file, not the whole feature.

For high-complexity features, Call 3 iterates over header files the same way.
"""

import os
from typing import Optional

from llm.client import LLMClient

# Maximum characters per file in the implementation prompt.
# For large files (btree.c = 11K lines, vdbe.c = 9K lines), we extract only
# the relevant function sections rather than loading the full file.
MAX_FILE_CHARS = 30000


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_IMPL_FILE = """\
你是一名资深 C 语言数据库内核工程师。
你的任务是根据实现计划，对【单个文件】进行精确修改。

输出规范：
- 只输出该文件的 unified diff 格式修改（--- a/... +++ b/... @@ ... @@），不输出任何解释文字
- 新建文件用 --- /dev/null 和 +++ b/文件路径 的格式
- diff 内容必须语法正确，可以直接用 patch -p1 应用
- 如果该文件无需修改，输出空字符串
"""

SYSTEM_PROMPT_INTEGRATE = """\
你是一名资深 C 语言数据库内核工程师。
你的任务是审查已生成的代码修改，确保头文件声明和跨模块接口的一致性。

输出规范：
- 只输出 unified diff 格式，只包含头文件（.h）和接口相关的修改
- 不要重复 Call 2 已经生成的修改
- 如果接口已经正确，输出空字符串
"""

SYSTEM_PROMPT_FIX = """\
你是一名资深 C 语言数据库内核工程师。
你的任务是根据编译错误，修复一个有问题的 unified diff。

输出规范：
- 只输出修复后的完整 unified diff，不输出任何解释文字
- diff 必须可以直接用 patch -p1 应用
"""


# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

USER_PROMPT_IMPL_FILE_TEMPLATE = """\
## 总体实现计划
{implementation_plan}

## 当前任务：处理文件 `{file_path}`
- 需要修改的函数：{funcs_to_modify}
- 需要新增的函数：{funcs_to_add}

## 相关结构体 / 联合体 / 枚举定义
{structs_content}

## 文件内容
{file_content}

请输出 `{file_path}` 的 unified diff。如该文件无需修改，输出空字符串。
"""

USER_PROMPT_INTEGRATE_FILE_TEMPLATE = """\
## 实现计划
{implementation_plan}

## 已生成的源文件修改（Call 2 的输出，供参考）
{existing_diff}

## 当前头文件内容
{file_content}

请为 `{file_path}` 补充必要的声明和接口调整，只输出该文件的 unified diff。
如果无需调整，输出空字符串。
"""

USER_PROMPT_FIX_TEMPLATE = """\
## 实现计划
{implementation_plan}

## 当前 diff（编译失败）
{diff}

## 编译错误
{compile_errors}

请修复上述编译错误，输出修复后的完整 unified diff。
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_markdown_codeblock(text: str) -> str:
    """
    Remove ```diff ... ``` or ``` ... ``` wrappers that LLMs sometimes add
    around their diff output.
    """
    text = text.strip()
    for prefix in ("```diff\n", "```c\n", "```\n", "```diff", "```c", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith("\n```"):
        text = text[:-4]
    elif text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _is_valid_diff(text: str) -> bool:
    """Return True if text looks like a non-empty unified diff."""
    return bool(text) and "---" in text and "+++" in text


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


def _file_specific_ops(
    file_path: str,
    plan: dict,
    function_index: Optional[dict],
) -> tuple[list[str], list[str]]:
    """
    Return (funcs_to_modify, funcs_to_add) that belong to file_path.

    For funcs_to_modify we use function_index to filter by file; if no match
    is found we fall back to the full list (so the LLM still has context).
    For funcs_to_add we match on the in_file field.
    """
    all_mods = plan.get("functions_to_modify", [])
    fp = file_path.replace("\\", "/")

    if function_index:
        file_mods = [
            fn for fn in all_mods
            if function_index.get(fn, {}).get("file", "").replace("\\", "/") == fp
        ] or all_mods  # fallback: pass all when index has no match
    else:
        file_mods = all_mods

    file_adds = [
        fa.get("name", "")
        for fa in plan.get("functions_to_add", [])
        if fa.get("in_file", "").replace("\\", "/") == fp
    ]
    return file_mods, file_adds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Iterates over each source file individually to keep each LLM call small
    and focused, preventing truncated or empty responses on large features.

    Args:
        plan: output from planner.plan_feature()
        project_root: absolute path to the C project root
        client: LLMClient instance
        model: model to use (empty string = use client default)
        function_index: optional function index for smart large-file extraction
        struct_index: optional struct index for injecting struct definitions

    Returns:
        Concatenated unified diff string across all modified files.
    """
    model = model or client.default_model("main")

    # Collect source files to process (exclude .h — handled in Call 3)
    source_files = [f for f in plan.get("affected_files", []) if not f.endswith(".h")]
    for fn_add in plan.get("functions_to_add", []):
        target = fn_add.get("in_file", "")
        if target and not target.endswith(".h") and target not in source_files:
            source_files.append(target)

    # Pre-build the structs block (shared across all file calls)
    structs_block = _build_structs_block(plan, struct_index)

    all_diffs: list[str] = []
    print(f"[impl] {len(source_files)} source file(s) to process")

    for idx, file_path in enumerate(source_files, 1):
        print(f"  [{idx}/{len(source_files)}] {file_path} ...", end="", flush=True)

        file_mods, file_adds = _file_specific_ops(file_path, plan, function_index)

        _, content = _load_file_content(
            file_path, project_root,
            relevant_funcs=file_mods,
            function_index=function_index,
        )

        if content:
            file_content = f"=== {file_path} ===\n{content}"
        else:
            file_content = f"=== {file_path} (文件不存在，需要新建) ===\n(空文件)"

        user_msg = USER_PROMPT_IMPL_FILE_TEMPLATE.format(
            implementation_plan=plan.get("implementation_plan", ""),
            file_path=file_path,
            funcs_to_modify=", ".join(file_mods) or "(见实现计划)",
            funcs_to_add=", ".join(file_adds) or "(无)",
            structs_content=structs_block or "(none)",
            file_content=file_content,
        )

        raw = client.chat(
            model=model,
            system=SYSTEM_PROMPT_IMPL_FILE,
            user=user_msg,
            max_tokens=8192,
        ).strip()
        diff = _strip_markdown_codeblock(raw)

        if _is_valid_diff(diff):
            all_diffs.append(diff)
            print(f" {len(diff.splitlines())} lines")
        else:
            print(" (no changes)")

    return "\n\n".join(all_diffs)


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

    Iterates over each header file individually, passing the accumulated
    source-file diffs as context.

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

    all_diffs: list[str] = []
    print(f"[impl] {len(header_files)} header file(s) to integrate")

    for idx, file_path in enumerate(header_files, 1):
        print(f"  [{idx}/{len(header_files)}] {file_path} ...", end="", flush=True)

        _, content = _load_file_content(file_path, project_root)
        if content:
            file_content = f"=== {file_path} ===\n{content}"
        else:
            file_content = f"=== {file_path} (文件不存在，需要新建) ===\n(空文件)"

        user_msg = USER_PROMPT_INTEGRATE_FILE_TEMPLATE.format(
            implementation_plan=plan.get("implementation_plan", ""),
            existing_diff=existing_diff,
            file_path=file_path,
            file_content=file_content,
        )

        raw = client.chat(
            model=model,
            system=SYSTEM_PROMPT_INTEGRATE,
            user=user_msg,
            max_tokens=4096,
        ).strip()
        diff = _strip_markdown_codeblock(raw)

        if _is_valid_diff(diff):
            all_diffs.append(diff)
            print(f" {len(diff.splitlines())} lines")
        else:
            print(" (no changes)")

    return "\n\n".join(all_diffs)


def fix_diff(
    diff: str,
    compile_errors: str,
    plan: dict,
    project_root: str,
    client: LLMClient,
    model: str = "",
) -> str:
    """
    Ask the LLM to fix a diff that failed to compile.

    Args:
        diff: the current diff that produced compilation errors
        compile_errors: error messages from the compiler (pre-filtered)
        plan: the implementation plan for context
        project_root: absolute path to the C project root (unused, reserved)
        client: LLMClient instance
        model: model to use (empty string = use client default)

    Returns:
        A revised diff that hopefully resolves the compilation errors.
    """
    model = model or client.default_model("main")

    user_msg = USER_PROMPT_FIX_TEMPLATE.format(
        implementation_plan=plan.get("implementation_plan", ""),
        diff=diff,
        compile_errors=compile_errors,
    )

    raw = client.chat(
        model=model,
        system=SYSTEM_PROMPT_FIX,
        user=user_msg,
        max_tokens=8192,
    ).strip()
    return _strip_markdown_codeblock(raw)

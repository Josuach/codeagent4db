"""
Call 2 (and optional Call 3): Implementation agent.

Takes the plan from Call 1 and generates unified diff patches, one file at a
time.  Each LLM call handles exactly one source or header file so that:
  - The output size per call is small and predictable (no truncation).
  - Partial failures affect only the current file, not the whole feature.

Within each file call the prompt contains ONLY:
  - The specific function bodies that need to be modified (from function_index,
    with exact line numbers for diff generation)
  - Insertion-point context for new functions to add
  - Relevant struct/union/enum definitions from struct_index

This avoids sending large amounts of irrelevant code to the LLM.

For high-complexity features, Call 3 iterates over header files the same way.
"""

import os
from typing import Optional

from llm.client import LLMClient

# Fallback: maximum characters per file when function_index is unavailable.
MAX_FILE_CHARS = 30000

# Lines of context shown before/after each function body.
FUNC_CONTEXT_LINES = 5


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
- 代码片段中每行前面的数字是该行在文件中的真实行号，请用这些行号生成正确的 @@ 标记
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

## 文件相关代码片段（带行号）
{file_content}

请根据实现计划，生成 `{file_path}` 的 unified diff。
行号已在代码片段中标注（格式 "  NNN: 代码行"），请用这些行号生成正确的 @@ 标记。
如该文件无需修改，输出空字符串。
"""

USER_PROMPT_INTEGRATE_FILE_TEMPLATE = """\
## 实现计划
{implementation_plan}

## 已生成的源文件修改（Call 2 的输出，供参考）
{existing_diff}

## 当前头文件相关片段（带行号）
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


def _load_function_snippets(
    file_path: str,
    project_root: str,
    funcs_to_modify: list[str],
    funcs_to_add: list[str],
    function_index: dict,
    context_lines: int = FUNC_CONTEXT_LINES,
) -> str:
    """
    Extract only the relevant function bodies from a file, annotated with
    real line numbers so the LLM can generate precise unified diff hunks.

    Args:
        file_path: relative path from project root
        project_root: absolute path to project root
        funcs_to_modify: function names to be modified in this file
        funcs_to_add: function names to be newly added (show insertion point)
        function_index: preprocessed function index with start_line / end_line
        context_lines: extra lines shown before/after each function

    Returns:
        Formatted string with numbered code sections, or a "new file" notice.
    """
    full_path = os.path.join(project_root, file_path)
    if not os.path.exists(full_path):
        return f"(文件不存在，需要新建: {file_path})"

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    total_lines = len(all_lines)

    fp = file_path.replace("\\", "/")

    # Collect (start_0idx, end_0idx, label) sections
    sections: list[tuple[int, int, str]] = []

    for func_name in funcs_to_modify:
        info = function_index.get(func_name, {})
        if info.get("file", "").replace("\\", "/") != fp:
            continue
        s1 = info.get("start_line", 1)   # 1-based
        e1 = info.get("end_line", s1)    # 1-based
        start = max(0, s1 - 1 - context_lines)
        end = min(total_lines, e1 + context_lines)
        sections.append((start, end, f"[modify] {func_name}"))

    # For new functions: show the insertion point — end of the last known
    # function in this file, so the LLM knows where to append.
    if funcs_to_add:
        last_end_line = 0
        for info in function_index.values():
            if info.get("file", "").replace("\\", "/") == fp:
                last_end_line = max(last_end_line, info.get("end_line", 0))
        if last_end_line > 0:
            start = max(0, last_end_line - 1 - context_lines)
            end = min(total_lines, last_end_line + context_lines)
        else:
            start = max(0, total_lines - 30)
            end = total_lines
        label = "[add — insert after] " + ", ".join(funcs_to_add)
        sections.append((start, end, label))

    if not sections:
        # Fallback: show first 100 lines (includes, top-level declarations)
        sections = [(0, min(100, total_lines), "file header (no specific functions located)")]

    # Sort and merge adjacent / overlapping sections
    sections.sort()
    merged: list[list] = []
    for start, end, label in sections:
        if merged and start <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])

    # Build formatted output with line numbers
    parts = [f"/* File: {file_path}  (total: {total_lines} lines) */"]
    prev_end = 0
    for seg in merged:
        start, end, label = seg
        if start > prev_end:
            parts.append(f"\n/* ... (lines {prev_end + 1}–{start} omitted) ... */\n")
        parts.append(f"/* {label} */")
        for i, raw_line in enumerate(all_lines[start:end]):
            parts.append(f"{start + i + 1:6d}: {raw_line.rstrip()}")
        prev_end = end

    if prev_end < total_lines:
        parts.append(f"\n/* ... (lines {prev_end + 1}–{total_lines} omitted) ... */")

    return "\n".join(parts)


def _load_file_content(
    file_path: str,
    project_root: str,
    relevant_funcs: Optional[list[str]] = None,
    function_index: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Fallback file loader: returns (relative_path, content) with character-based
    truncation for large files.  Used for header files in integrate_interfaces()
    and as a last resort when function_index is unavailable.
    """
    full_path = os.path.join(project_root, file_path)
    if not os.path.exists(full_path):
        return file_path, ""
    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if len(content) <= MAX_FILE_CHARS:
        return file_path, content

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    sections_to_include: list[tuple[int, int]] = []
    if relevant_funcs and function_index:
        for func_name in relevant_funcs:
            info = function_index.get(func_name, {})
            if info.get("file", "").replace("\\", "/") == file_path.replace("\\", "/"):
                start = max(0, info.get("start_line", 1) - 1 - 10)
                end = min(total_lines, info.get("end_line", info.get("start_line", 1)) + 5)
                sections_to_include.append((start, end))

    if not sections_to_include:
        header_end = min(200, total_lines)
        sections_to_include = [(0, header_end)]

    sections_to_include.sort()
    merged: list[tuple[int, int]] = []
    for s, e in sections_to_include:
        if merged and s <= merged[-1][1] + 5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    parts = [f"/* ... (lines 1-{merged[0][0]}) ... */\n" if merged[0][0] > 0 else ""]
    for i, (s, e) in enumerate(merged):
        parts.append("".join(lines[s:e]))
        if i + 1 < len(merged):
            parts.append(f"\n/* ... (lines {e+1}-{merged[i+1][0]}) ... */\n")
        elif e < total_lines:
            parts.append(f"\n/* ... (lines {e+1}-{total_lines}, {total_lines - e} more lines) ... */\n")

    extracted = "".join(parts)
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

    names: list = list(plan.get("relevant_structs", []))
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


def _build_file_tasks(
    plan: dict,
    function_index: Optional[dict],
) -> dict[str, dict]:
    """
    Build a mapping {file_path: {"mods": [...], "adds": [...]}} from the plan.

    - functions_to_modify: resolved to their source file via function_index
      (falls back to affected_files when index is unavailable)
    - functions_to_add:    resolved via plan["functions_to_add"][i]["in_file"]
    - new_files:           added with empty mods/adds so new-file diffs are generated
    """
    file_tasks: dict[str, dict] = {}

    def _ensure(fp: str) -> dict:
        if fp not in file_tasks:
            file_tasks[fp] = {"mods": [], "adds": []}
        return file_tasks[fp]

    # functions_to_modify → resolved file via function_index
    if function_index:
        for func_name in plan.get("functions_to_modify", []):
            fp = function_index.get(func_name, {}).get("file", "").replace("\\", "/")
            if fp:
                _ensure(fp)["mods"].append(func_name)
            # if not found in index, silently skip (unknown location)
    else:
        # No index: fall back to affected_files, assign all mods to each file
        all_mods = plan.get("functions_to_modify", [])
        for f in plan.get("affected_files", []):
            if not f.endswith(".h"):
                _ensure(f.replace("\\", "/"))["mods"].extend(all_mods)

    # functions_to_add → resolved via in_file
    for fn_add in plan.get("functions_to_add", []):
        fp = fn_add.get("in_file", "").replace("\\", "/")
        if fp and not fp.endswith(".h"):
            _ensure(fp)["adds"].append(fn_add.get("name", ""))

    # new source files (may have no functions yet)
    for new_file in plan.get("new_files", []):
        fp = new_file.replace("\\", "/")
        if not fp.endswith(".h"):
            _ensure(fp)

    return file_tasks


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

    One LLM call per source file. Each call receives only the specific
    function bodies and struct definitions relevant to that file, with line
    numbers so the LLM can produce accurate diff hunks.

    Args:
        plan: output from planner.plan_feature()
        project_root: absolute path to the C project root
        client: LLMClient instance
        model: model to use (empty string = use client default)
        function_index: preprocessed function index (start_line / end_line per function)
        struct_index: preprocessed struct/union/enum index

    Returns:
        Concatenated unified diff string across all modified files.
    """
    model = model or client.default_model("main")

    # Build {file_path: {mods, adds}} from plan's function lists (not affected_files)
    file_tasks = _build_file_tasks(plan, function_index)

    if not file_tasks:
        print("[impl] no source files to process")
        return ""

    # Pre-build the structs block — shared across all per-file calls
    structs_block = _build_structs_block(plan, struct_index)

    all_diffs: list[str] = []
    file_list = list(file_tasks.items())
    print(f"[impl] {len(file_list)} source file(s) to process")

    for idx, (file_path, tasks) in enumerate(file_list, 1):
        file_mods = tasks["mods"]
        file_adds = tasks["adds"]
        print(
            f"  [{idx}/{len(file_list)}] {file_path}"
            f"  (modify: {len(file_mods)}, add: {len(file_adds)}) ...",
            end="", flush=True,
        )

        # Use targeted snippet extraction when function_index is available;
        # fall back to character-limited full-file loading otherwise.
        if function_index:
            file_content = _load_function_snippets(
                file_path, project_root,
                funcs_to_modify=file_mods,
                funcs_to_add=file_adds,
                function_index=function_index,
            )
        else:
            _, raw = _load_file_content(file_path, project_root,
                                        relevant_funcs=file_mods)
            file_content = f"=== {file_path} ===\n{raw}" if raw else \
                           f"(文件不存在，需要新建: {file_path})"

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

    One LLM call per header file, with the accumulated source-file diffs
    passed as context.

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

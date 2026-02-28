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
- 代码片段中的 /* ... (lines N–M omitted) ... */ 是省略标记，不要将其输出到 diff 中
- 函数实现必须完整，不得使用 TODO、占位符或省略号截断
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

SYSTEM_PROMPT_REVIEW = """\
你是一名资深 C 语言数据库内核工程师，负责代码 diff 审查。
你的任务是逐步检查生成的 unified diff 是否完整、正确地实现了计划中的每个步骤。

审查重点：
1. 每个计划步骤是否在 diff 中有对应的实际代码改动
2. 新增函数是否有完整逻辑实现（不能是空桩或 TODO 占位符）
3. 修改的函数是否真正实现了步骤所描述的功能
4. 函数签名、返回值、错误处理是否合理
5. 跨步骤的依赖关系是否一致（如新函数被正确调用）
"""

SYSTEM_PROMPT_REFINE = """\
你是一名资深 C 语言数据库内核工程师。
根据审查意见，补全并修正 unified diff，使其完整实现所有计划步骤。

输出规范：
- 只输出修正后的完整 unified diff，不输出任何解释文字
- diff 必须可以直接用 patch -p1 应用
- 保留原有正确的修改，只补充缺失或修正错误的部分
- 代码片段中的 /* ... (lines N–M omitted) ... */ 是省略标记，不要将其输出到 diff 中
- 函数实现必须完整，不得使用 TODO、占位符或省略号截断
"""


# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

USER_PROMPT_IMPL_STEP_TEMPLATE = """\
## 整体特性概要
{feature_summary}

## 当前步骤 [{step_index}/{step_total}]
{step_description}

- 需要修改的函数：{funcs_to_modify}
- 需要新增的函数：{funcs_to_add}

## 相关结构体 / 联合体 / 枚举定义
{structs_content}

## 相关代码片段（带行号）
{file_content}

请根据当前步骤的要求生成 unified diff。
代码片段中每行前的数字是该行在文件中的真实行号，请用这些行号生成正确的 @@ 标记。
如该步骤无需代码改动，输出空字符串。
"""

USER_PROMPT_INTEGRATE_FILE_TEMPLATE = """\
## 实现计划步骤摘要
{implementation_plan}

## 已生成的源文件修改（供参考）
{existing_diff}

## 当前头文件相关片段（带行号）
{file_content}

请为 `{file_path}` 补充必要的声明和接口调整，只输出该文件的 unified diff。
如果无需调整，输出空字符串。
"""

USER_PROMPT_FIX_TEMPLATE = """\
## 实现计划步骤摘要
{implementation_plan}

## 当前 diff（编译失败）
{diff}

## 编译错误
{compile_errors}

请修复上述编译错误，输出修复后的完整 unified diff。
"""

USER_PROMPT_REVIEW_TEMPLATE = """\
## 实现计划（共 {step_total} 步）
{steps_list}

## 生成的 diff
{diff}

请逐步检查上述 diff 是否完整实现了每个计划步骤，以 JSON 格式输出审查结果。
"""

USER_PROMPT_REFINE_TEMPLATE = """\
## 实现计划（共 {step_total} 步）
{steps_list}

## 当前 diff（审查未通过）
{diff}

## 审查意见
{issues}

## 尚未实现的步骤
{unimplemented_steps}

## 相关结构体 / 联合体 / 枚举定义
{structs_content}

## 相关代码片段（带行号）
{code_context}

请输出补全修正后的完整 unified diff。
"""

# JSON schema for structured review output
_REVIEW_SCHEMA_NAME = "diff_review"
_REVIEW_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {
            "type": "boolean",
            "description": "True if the diff correctly and completely implements all required steps",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of concrete issues found (empty list if approved)",
        },
        "unimplemented_steps": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-based indices of plan steps not yet implemented in the diff",
        },
    },
    "required": ["approved", "issues", "unimplemented_steps"],
}


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


def _load_step_snippets(
    step: dict,
    project_root: str,
    function_index: Optional[dict],
    context_lines: int = FUNC_CONTEXT_LINES,
) -> str:
    """
    Collect all code snippets needed for a single plan step.

    Groups the step's functions_to_modify and functions_to_add by file,
    then calls _load_function_snippets for each file involved.

    Returns a combined snippet string across all files in the step.
    """
    funcs_to_modify = step.get("functions_to_modify", [])
    funcs_to_add = step.get("functions_to_add", [])

    if not function_index:
        # No index: return a placeholder
        return "(function_index not available — cannot extract targeted snippets)"

    # Group by file
    file_mods: dict[str, list[str]] = {}
    file_adds: dict[str, list[str]] = {}

    for func_name in funcs_to_modify:
        fp = function_index.get(func_name, {}).get("file", "").replace("\\", "/")
        if fp:
            file_mods.setdefault(fp, []).append(func_name)

    for fn_add in funcs_to_add:
        fp = fn_add.get("in_file", "").replace("\\", "/")
        name = fn_add.get("name", "")
        if fp and name:
            file_adds.setdefault(fp, []).append(name)

    all_fps = sorted(set(file_mods) | set(file_adds))

    if not all_fps:
        return "(no functions located in index for this step)"

    parts = []
    for fp in all_fps:
        snippet = _load_function_snippets(
            fp, project_root,
            funcs_to_modify=file_mods.get(fp, []),
            funcs_to_add=file_adds.get(fp, []),
            function_index=function_index,
            context_lines=context_lines,
        )
        parts.append(snippet)

    return "\n\n".join(parts)


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
    Call 2: Generate unified diffs step by step.

    Iterates over plan["steps"] (each step = one atomic code change).
    For each step, retrieves only the function bodies involved in that step
    (via function_index), then makes one LLM call to produce the diff.

    Args:
        plan: output from planner.plan_feature()
        project_root: absolute path to the C project root
        client: LLMClient instance
        model: model to use (empty string = use client default)
        function_index: preprocessed function index (start_line / end_line per function)
        struct_index: preprocessed struct/union/enum index

    Returns:
        Concatenated unified diff string across all steps.
    """
    model = model or client.default_model("main")
    steps = plan.get("steps", [])

    if not steps:
        print("[impl] plan has no steps to process")
        return ""

    # Build a short feature summary from step descriptions for context
    feature_summary = "\n".join(
        f"  {i}. {s.get('description', '')}"
        for i, s in enumerate(steps, 1)
    )

    # Pre-build the structs block — shared across all step calls
    structs_block = _build_structs_block(plan, struct_index)

    all_diffs: list[str] = []
    print(f"[impl] {len(steps)} step(s) to process")

    for idx, step in enumerate(steps, 1):
        desc = step.get("description", f"Step {idx}")
        step_mods = step.get("functions_to_modify", [])
        step_adds = step.get("functions_to_add", [])
        add_names = [fa.get("name", "") for fa in step_adds]

        print(
            f"  [{idx}/{len(steps)}] {desc[:70]}"
            f"  (modify: {len(step_mods)}, add: {len(add_names)}) ...",
            end="", flush=True,
        )

        # Load code snippets for all functions involved in this step
        if function_index:
            file_content = _load_step_snippets(
                step, project_root, function_index,
            )
        else:
            # Fallback: show first 100 lines of each affected file
            affected = plan.get("affected_files", [])
            parts = []
            for fpath in affected:
                if not fpath.endswith(".h"):
                    _, raw = _load_file_content(fpath, project_root)
                    if raw:
                        parts.append(f"=== {fpath} ===\n{raw[:5000]}")
            file_content = "\n\n".join(parts) or "(no files found)"

        user_msg = USER_PROMPT_IMPL_STEP_TEMPLATE.format(
            feature_summary=feature_summary,
            step_index=idx,
            step_total=len(steps),
            step_description=desc,
            funcs_to_modify=", ".join(step_mods) or "(无)",
            funcs_to_add=", ".join(add_names) or "(无)",
            structs_content=structs_block or "(none)",
            file_content=file_content,
        )

        raw = client.chat(
            model=model,
            system=SYSTEM_PROMPT_IMPL_FILE,
            user=user_msg,
            max_tokens=16384,
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

        steps_summary = "\n".join(
            f"  {i}. {s.get('description', '')}"
            for i, s in enumerate(plan.get("steps", []), 1)
        ) or "(no steps)"
        user_msg = USER_PROMPT_INTEGRATE_FILE_TEMPLATE.format(
            implementation_plan=steps_summary,
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

    steps_summary = "\n".join(
        f"  {i}. {s.get('description', '')}"
        for i, s in enumerate(plan.get("steps", []), 1)
    ) or "(no steps)"
    user_msg = USER_PROMPT_FIX_TEMPLATE.format(
        implementation_plan=steps_summary,
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


# Max characters of diff sent to the review prompt (avoid overwhelming context).
_REVIEW_DIFF_CHARS = 40000


def review_and_refine_diff(
    diff: str,
    plan: dict,
    client: LLMClient,
    model: str = "",
    max_rounds: int = 1,
    function_index: Optional[dict] = None,
    struct_index: Optional[dict] = None,
    project_root: str = "",
) -> str:
    """
    Post-generation LLM review loop.

    After implement_feature() produces a diff, this function:
      1. Asks the LLM to review the diff against the plan steps (structured JSON).
      2. If issues are found, collects relevant code snippets for the problematic
         steps (when function_index + project_root are provided) and asks the LLM
         to produce a corrected diff.
      3. Repeats up to max_rounds times.

    Args:
        diff:           The combined unified diff from Call 2 (+ Call 3 if any).
        plan:           The implementation plan dict.
        client:         LLMClient instance.
        model:          Model to use (empty = client default).
        max_rounds:     Maximum number of review+refine cycles (0 = skip review).
        function_index: Preprocessed function index (for snippet extraction in refine).
        struct_index:   Preprocessed struct index (for struct context in refine).
        project_root:   Absolute path to C project root (for snippet extraction).

    Returns:
        The final (possibly refined) diff string.
    """
    if max_rounds <= 0 or not diff:
        return diff

    model = model or client.default_model("main")
    steps = plan.get("steps", [])

    steps_list = "\n".join(
        f"  {i}. {s.get('description', '')}"
        for i, s in enumerate(steps, 1)
    )

    current_diff = diff

    for round_num in range(1, max_rounds + 1):
        print(f"[review] Round {round_num}/{max_rounds}: reviewing diff ...", flush=True)

        # Truncate diff if too large for review prompt
        diff_for_prompt = current_diff
        if len(diff_for_prompt) > _REVIEW_DIFF_CHARS:
            diff_for_prompt = (
                current_diff[:_REVIEW_DIFF_CHARS]
                + f"\n\n/* ... diff truncated at {_REVIEW_DIFF_CHARS} chars for review ... */"
            )

        review_user = USER_PROMPT_REVIEW_TEMPLATE.format(
            step_total=len(steps),
            steps_list=steps_list,
            diff=diff_for_prompt,
        )

        review = client.chat_structured(
            model=model,
            system=SYSTEM_PROMPT_REVIEW,
            user=review_user,
            max_tokens=4096,
            schema_name=_REVIEW_SCHEMA_NAME,
            json_schema=_REVIEW_JSON_SCHEMA,
        )

        if not review:
            print("[review] Review returned empty result; stopping review loop.")
            break

        approved = review.get("approved", False)
        issues = review.get("issues", [])
        unimplemented = review.get("unimplemented_steps", [])

        status = "APPROVED" if approved else f"NOT APPROVED ({len(issues)} issue(s))"
        print(f"[review] {status}  unimplemented steps: {unimplemented or '(none)'}")
        for issue in issues:
            print(f"         - {issue}")

        if approved:
            break

        if round_num >= max_rounds:
            print("[review] Max rounds reached; using current diff.")
            break

        # --- Refine ---
        print("[review] Refining diff ...", flush=True)

        issues_text = "\n".join(f"- {iss}" for iss in issues) or "(none listed)"
        unimpl_text = "\n".join(
            f"- Step {i}: {steps[i - 1].get('description', '')}"
            for i in unimplemented
            if 1 <= i <= len(steps)
        ) or "(none)"

        # Build code context for problematic steps
        code_context = "(no function index provided)"
        if function_index and project_root:
            # Load snippets for unimplemented steps; fall back to all steps if none flagged
            target_indices = [i for i in unimplemented if 1 <= i <= len(steps)] \
                             or list(range(1, len(steps) + 1))
            snippet_parts = []
            for si in target_indices:
                step = steps[si - 1]
                snippet = _load_step_snippets(step, project_root, function_index)
                if snippet and "(no functions" not in snippet and "(function_index" not in snippet:
                    snippet_parts.append(f"=== Step {si}: {step.get('description', '')} ===\n{snippet}")
            code_context = "\n\n".join(snippet_parts) or "(no relevant snippets found)"

        structs_content = _build_structs_block(plan, struct_index) or "(none)"

        refine_user = USER_PROMPT_REFINE_TEMPLATE.format(
            step_total=len(steps),
            steps_list=steps_list,
            diff=diff_for_prompt,
            issues=issues_text,
            unimplemented_steps=unimpl_text,
            structs_content=structs_content,
            code_context=code_context,
        )

        raw = client.chat(
            model=model,
            system=SYSTEM_PROMPT_REFINE,
            user=refine_user,
            max_tokens=16384,
        ).strip()
        refined = _strip_markdown_codeblock(raw)

        if _is_valid_diff(refined):
            current_diff = refined
            print(f"[review] Refined diff: {len(refined.splitlines())} lines")
        else:
            print("[review] Refinement produced no valid diff; keeping current.")
            break

    return current_diff

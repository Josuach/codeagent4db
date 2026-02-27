"""
LLM batch function summarizer.

Sends batches of C functions to the LLM and collects structured summaries
including: one-line description, subsystem, call scenario, and key data structures.

The rich summary format is critical — it determines retrieval quality downstream.

Usage:
    python batch_summarizer.py --root /path/to/project --output cache/function_index.json
"""

import json
import os
import re
import time
from dataclasses import asdict
from typing import Optional

from llm.client import LLMClient

from preprocess.c_parser import FunctionInfo, build_call_graph, build_reverse_call_graph, parse_project

# How many functions per LLM batch. Tune based on average function body length.
# At ~30 lines/function, 40 functions ≈ 1200 lines ≈ ~8k tokens input.
DEFAULT_BATCH_SIZE = 40

# Max characters of function body to include per function.
# Very long functions are truncated to keep batch size manageable.
MAX_BODY_CHARS = 2000

SYSTEM_PROMPT = """你是 C 语言数据库内核代码分析专家，负责为代码检索系统构建高质量的函数索引。

你的输出将直接用于检索——当开发者描述一个特性时，系统会通过你的摘要找到需要修改的函数。
因此，摘要必须包含足够的语义信息，不能只描述"做了什么动作"，还要描述"在什么场景下被使用"。

输出格式要求：严格 JSON，不包含任何 markdown 代码块标记，不包含任何解释文字。"""

BATCH_PROMPT_TEMPLATE = """为以下 C 函数生成结构化摘要，用于构建代码检索索引。

输出格式（严格 JSON object，key 为函数名）：
{{
  "函数名": {{
    "summary": "一句话功能描述（≤40字）",
    "subsystem": "所属子系统（如 storage/executor/optimizer/transaction/index/buffer/wal/lock/catalog 等）",
    "scenario": "被谁调用、在什么业务场景下使用（≤60字）",
    "data_structures": ["涉及的核心数据结构或概念"]
  }},
  ...
}}

摘要质量要求：
- summary 描述功能目的，不只是实现动作
- scenario 要包含调用场景（如"被 exec_insert 和 COPY 写入路径调用"）
- data_structures 包含函数操作的核心结构体名称或数据库概念（如 WAL、MVCC、checkpoint）

函数列表：
{functions_block}"""


def _format_function_block(
    func: FunctionInfo,
    callers: list[str],
) -> str:
    """Format a single function for inclusion in the batch prompt."""
    body = func.body
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n    /* ... truncated ... */"

    caller_note = ""
    if callers:
        caller_note = f"\n// 已知调用者: {', '.join(callers[:5])}"

    comment_block = ""
    if func.comment:
        comment_block = func.comment + "\n"

    return (
        f"--- {func.name} ({func.file}:{func.start_line}) ---\n"
        f"{comment_block}"
        f"{func.signature}{caller_note}\n"
        f"{body}\n"
    )


def _parse_llm_json(response_text: str) -> dict:
    """
    Extract JSON from LLM response, handling cases where the model
    wraps it in markdown code fences despite instructions not to.
    """
    text = response_text.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def summarize_batch(
    client: LLMClient,
    functions: list[FunctionInfo],
    reverse_call_graph: dict[str, list[str]],
    model: str = "",
    retries: int = 2,
) -> dict[str, dict]:
    """
    Call the LLM to summarize one batch of functions.
    Returns a dict {func_name: {summary, subsystem, scenario, data_structures}}.
    """
    model = model or client.default_model("summarizer")
    functions_block = "\n".join(
        _format_function_block(f, reverse_call_graph.get(f.name, []))
        for f in functions
    )
    prompt = BATCH_PROMPT_TEMPLATE.format(functions_block=functions_block)

    for attempt in range(retries + 1):
        try:
            raw_text = client.chat(model=model, system=SYSTEM_PROMPT, user=prompt, max_tokens=8192)
            parsed = _parse_llm_json(raw_text)
            if parsed:
                return parsed
            print(f"[warn] empty JSON parse on attempt {attempt + 1}, retrying...")
        except Exception as e:
            print(f"[error] LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)

    # Return empty stubs so the caller can continue without crashing
    return {f.name: {"summary": "", "subsystem": "", "scenario": "", "data_structures": []}
            for f in functions}


CHECKPOINT_INTERVAL = 10  # Save index to disk every N batches


def build_function_index(
    file_map: dict[str, list[FunctionInfo]],
    client: LLMClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = "",
    existing_index: Optional[dict] = None,
    skip_names: Optional[set[str]] = None,
    checkpoint_path: Optional[str] = None,
) -> dict[str, dict]:
    """
    Build the complete function index by batching LLM summarization calls.

    Args:
        file_map: {file_path: [FunctionInfo]} from parse_project()
        client: LLMClient instance
        batch_size: functions per LLM call
        model: LLM model to use (empty string = use client default)
        existing_index: pre-existing index to merge into (for incremental updates)
        skip_names: function names to skip (already in existing_index)
        checkpoint_path: if set, save partial index here every CHECKPOINT_INTERVAL batches

    Returns:
        {func_name: {file, signature, start_line, summary, subsystem, scenario, data_structures}}
    """
    call_graph = build_call_graph(file_map)
    reverse_cg = build_reverse_call_graph(call_graph)

    # Collect all functions, skipping already-indexed ones
    all_funcs: list[FunctionInfo] = []
    for funcs in file_map.values():
        for f in funcs:
            if skip_names and f.name in skip_names:
                continue
            all_funcs.append(f)

    total = len(all_funcs)
    print(f"[summarizer] {total} functions to summarize in batches of {batch_size}")

    index = dict(existing_index) if existing_index else {}

    for batch_start in range(0, total, batch_size):
        batch = all_funcs[batch_start: batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total)
        batch_num = batch_start // batch_size + 1
        print(f"[summarizer] batch {batch_num}: "
              f"functions {batch_start + 1}-{batch_end}/{total}")

        summaries = summarize_batch(client, batch, reverse_cg, model=model)

        # Merge LLM summaries with static metadata from FunctionInfo
        for f in batch:
            llm_data = summaries.get(f.name, {})
            index[f.name] = {
                "file": f.file,
                "signature": f.signature,
                "start_line": f.start_line,
                "end_line": f.end_line,
                "summary": llm_data.get("summary", ""),
                "subsystem": llm_data.get("subsystem", ""),
                "scenario": llm_data.get("scenario", ""),
                "data_structures": llm_data.get("data_structures", []),
                "direct_calls": list(call_graph.get(f.name, [])),
                "known_callers": list(reverse_cg.get(f.name, []))[:10],
            }

        # Checkpoint: save partial index periodically so we can resume if interrupted
        if checkpoint_path and batch_num % CHECKPOINT_INTERVAL == 0:
            save_index(index, checkpoint_path)
            print(f"[summarizer] checkpoint saved ({len(index)} functions indexed)")

        # Brief pause to respect rate limits
        if batch_end < total:
            time.sleep(0.5)

    return index


def build_file_index(
    file_map: dict[str, list[FunctionInfo]],
    function_index: dict[str, dict],
) -> dict[str, dict]:
    """
    Build a file-level index by aggregating function summaries.
    No additional LLM calls needed — derived from function_index.

    Returns:
        {file_path: {subsystems, function_names, description}}
    """
    file_index: dict[str, dict] = {}
    for file_path, funcs in file_map.items():
        subsystems = set()
        func_names = []
        summary_fragments = []
        for f in funcs:
            fi = function_index.get(f.name, {})
            if fi.get("subsystem"):
                subsystems.add(fi["subsystem"])
            func_names.append(f.name)
            if fi.get("summary"):
                summary_fragments.append(fi["summary"])

        # Use the first few function summaries as a file description
        description = "；".join(summary_fragments[:5])
        if len(summary_fragments) > 5:
            description += f"（共 {len(funcs)} 个函数）"

        file_index[file_path] = {
            "subsystems": sorted(subsystems),
            "function_count": len(funcs),
            "function_names": func_names,
            "description": description,
        }
    return file_index


def save_index(index: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[summarizer] saved index ({len(index)} entries) to {path}")


def load_index(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLM batch function summarizer")
    p.add_argument("--root", required=True, help="Project root directory")
    p.add_argument("--output", default="cache/function_index.json")
    p.add_argument("--file-output", default="cache/file_index.json")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--incremental", action="store_true",
                   help="Skip functions already in the output index")
    args = p.parse_args()

    from llm.client import LLMConfig
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY (or LLM_API_KEY) environment variable not set")

    client = LLMClient(LLMConfig(api_key=api_key))

    print(f"[summarizer] parsing project at {args.root} ...")
    file_map = parse_project(args.root)
    total_funcs = sum(len(v) for v in file_map.values())
    print(f"[summarizer] found {total_funcs} functions across {len(file_map)} files")

    existing = {}
    skip_names = None
    if args.incremental:
        existing = load_index(args.output)
        skip_names = set(existing.keys())
        print(f"[summarizer] incremental mode: skipping {len(skip_names)} already-indexed functions")

    func_index = build_function_index(
        file_map, client,
        batch_size=args.batch_size,
        model=args.model,
        existing_index=existing,
        skip_names=skip_names,
    )
    save_index(func_index, args.output)

    file_index = build_file_index(file_map, func_index)
    save_index(file_index, args.file_output)
    print(f"[summarizer] done. {len(func_index)} functions indexed.")

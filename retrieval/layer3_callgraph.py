"""
Layer 3 retrieval: call graph expansion.

Given a set of seed functions (from Layer 2 BM25), expands the candidate set
by walking the call graph: callee expansion (find what they call) and
caller expansion (find who calls them).

Uses the pre-built call graph from c_parser.py / callchain_builder.py.
Applies a utility-function blacklist to avoid exponential expansion.
"""

from preprocess.c_parser import UTILITY_BLACKLIST  # noqa: E402


def expand_with_callgraph(
    seed_funcs: list[str],
    call_graph: dict[str, list[str]],
    reverse_call_graph: dict[str, list[str]],
    callee_depth: int = 2,
    caller_depth: int = 1,
    max_seeds: int = 5,
    max_expand_per_seed: int = 10,
    blacklist: set[str] | None = None,
) -> list[str]:
    """
    Expand a seed set by following call graph edges.

    Strategy:
    - Callee expansion (depth=2): find what the seed functions call.
      Good for finding helper functions that need to change.
    - Caller expansion (depth=1): find who calls the seed functions.
      Good for estimating blast radius / finding integration points.
    - Blacklist: skip utility functions (palloc, elog, etc.) to avoid explosion.

    Args:
        seed_funcs: top-k functions from BM25 (ordered by relevance)
        call_graph: {func: [callees]}
        reverse_call_graph: {func: [callers]}
        callee_depth: how deep to follow callees
        caller_depth: how deep to follow callers
        max_seeds: only expand the top N seeds (rest kept as-is)
        max_expand_per_seed: max new functions to add per seed
        blacklist: function names to skip during expansion

    Returns:
        Expanded list of function names (seed_funcs + newly discovered).
        Seeds appear first, preserving their rank order.
    """
    if blacklist is None:
        blacklist = UTILITY_BLACKLIST

    result: list[str] = list(seed_funcs)
    seen: set[str] = set(seed_funcs)

    def _expand_callees(func: str, depth: int, added: list[str]):
        if depth == 0 or len(added) >= max_expand_per_seed:
            return
        for callee in call_graph.get(func, []):
            if callee in blacklist or callee in seen:
                continue
            seen.add(callee)
            added.append(callee)
            _expand_callees(callee, depth - 1, added)

    def _expand_callers(func: str, depth: int, added: list[str]):
        if depth == 0 or len(added) >= max_expand_per_seed:
            return
        for caller in reverse_call_graph.get(func, []):
            if caller in blacklist or caller in seen:
                continue
            seen.add(caller)
            added.append(caller)
            _expand_callers(caller, depth - 1, added)

    for seed in seed_funcs[:max_seeds]:
        new_funcs: list[str] = []
        _expand_callees(seed, callee_depth, new_funcs)
        _expand_callers(seed, caller_depth, new_funcs)
        result.extend(new_funcs[:max_expand_per_seed])

    return result


def format_candidates_for_prompt(
    candidate_funcs: list[str],
    function_index: dict,
    max_entries: int = 40,
) -> str:
    """
    Format the candidate function list for inclusion in the LLM planning prompt.

    Output format (one function per line):
        storage/heap.c:234  heap_insert(HeapFile*, Tuple*)
          → 向堆文件写入元组；storage 子系统；被 exec_insert 调用；涉及 WAL
    """
    lines = []
    for func_name in candidate_funcs[:max_entries]:
        info = function_index.get(func_name)
        if not info:
            lines.append(f"  (unknown)  {func_name}")
            continue

        file_loc = f"{info.get('file', '?')}:{info.get('start_line', '?')}"
        sig = info.get("signature", func_name)
        # Shorten very long signatures
        if len(sig) > 80:
            sig = sig[:77] + "..."

        summary = info.get("summary", "")
        subsystem = info.get("subsystem", "")
        scenario = info.get("scenario", "")

        detail_parts = []
        if summary:
            detail_parts.append(summary)
        if subsystem:
            detail_parts.append(f"{subsystem} 子系统")
        if scenario:
            detail_parts.append(scenario)

        detail = "；".join(detail_parts)

        lines.append(f"{file_loc:<35}  {sig}")
        if detail:
            lines.append(f"  → {detail}")

    if len(candidate_funcs) > max_entries:
        lines.append(f"  ... (共 {len(candidate_funcs)} 个候选，仅显示前 {max_entries} 个)")

    return "\n".join(lines)

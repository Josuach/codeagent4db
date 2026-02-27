"""
Call chain builder.
Starting from user-specified entry functions, performs BFS/DFS traversal
of the project call graph to produce nested call trees.

Usage:
    python callchain_builder.py --root /path/to/project --config callchain_entries.yaml
"""

import json
import os
import yaml
from collections import deque
from typing import Optional

from preprocess.c_parser import (
    FunctionInfo,
    build_call_graph,
    build_reverse_call_graph,
    parse_project,
    UTILITY_BLACKLIST,
)


def build_call_tree(
    entry: str,
    call_graph: dict[str, list[str]],
    max_depth: int = 8,
    blacklist: Optional[set[str]] = None,
) -> dict:
    """
    BFS traversal from `entry`, producing a nested dict call tree.

    Example output:
        {
          "exec_insert": {
            "heap_insert": {
              "get_free_page": {},
              "write_page": {}
            },
            "update_catalog": {}
          }
        }

    Cycles are broken by tracking visited nodes per path (not globally,
    so the same function CAN appear in different branches).
    To avoid explosive duplication, once a function has been expanded at
    any depth it is not expanded again (mark-once strategy).

    Args:
        entry: name of the root function
        call_graph: {func: [callees]}
        max_depth: maximum recursion depth
        blacklist: function names to treat as leaves (not expanded)
    """
    if blacklist is None:
        blacklist = UTILITY_BLACKLIST

    expanded: set[str] = set()  # functions already expanded (mark-once)

    def _recurse(func: str, depth: int) -> dict:
        if depth >= max_depth:
            return {"__truncated__": True}
        if func in blacklist:
            return {}
        if func in expanded:
            return {"__ref__": func}   # already shown elsewhere in tree
        expanded.add(func)

        children = call_graph.get(func, [])
        result = {}
        for callee in children:
            result[callee] = _recurse(callee, depth + 1)
        return result

    tree = {entry: _recurse(entry, 0)}
    return tree


def flatten_call_tree(tree: dict, prefix: str = "", depth: int = 0) -> list[str]:
    """
    Flatten a call tree into a list of strings for human-readable display.
    Example:
        exec_insert
          heap_insert
            get_free_page
            write_page
          update_catalog
    """
    lines = []
    for func, subtree in tree.items():
        indent = "  " * depth
        if isinstance(subtree, dict) and subtree.get("__truncated__"):
            lines.append(f"{indent}{func} [depth limit]")
        elif isinstance(subtree, dict) and "__ref__" in subtree:
            lines.append(f"{indent}{func} -> (see {subtree['__ref__']})")
        else:
            lines.append(f"{indent}{func}")
            if isinstance(subtree, dict):
                lines.extend(flatten_call_tree(subtree, prefix, depth + 1))
    return lines


def get_all_functions_in_tree(tree: dict) -> set[str]:
    """
    Collect every function name that appears in a call tree (excluding markers).
    """
    result: set[str] = set()

    def _walk(node: dict):
        for key, val in node.items():
            if key.startswith("__"):
                continue
            result.add(key)
            if isinstance(val, dict):
                _walk(val)

    _walk(tree)
    return result


def build_all_callchains(
    entries_config: list[dict],
    call_graph: dict[str, list[str]],
    max_depth: int = 8,
    blacklist: Optional[set[str]] = None,
) -> dict:
    """
    Build call trees for all entries defined in the config.

    Args:
        entries_config: list of {"name": str, "entry": str} dicts
        call_graph: project-wide call graph
        max_depth: passed to build_call_tree
        blacklist: utility functions to skip

    Returns:
        {
          "write_path": { "exec_insert": { ... } },
          "query_path": { "exec_query": { ... } },
          ...
        }
    """
    result = {}
    for item in entries_config:
        name = item["name"]
        entry_func = item["entry"]
        if entry_func not in call_graph:
            print(f"[warn] entry function '{entry_func}' not found in call graph")
            result[name] = {entry_func: {"__not_found__": True}}
        else:
            result[name] = build_call_tree(entry_func, call_graph, max_depth, blacklist)
    return result


def load_entries_config(yaml_path: str) -> list[dict]:
    """Load callchain_entries.yaml."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("entries", [])


def save_callchains(callchains: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(callchains, f, ensure_ascii=False, indent=2)
    print(f"[callchain] saved to {output_path}")


def format_callchains_for_prompt(callchains: dict, max_lines_per_chain: int = 40) -> str:
    """
    Render call chains into a compact text block suitable for inclusion in LLM prompts.
    Truncates very deep chains to keep token count manageable.
    """
    parts = []
    for chain_name, tree in callchains.items():
        lines = flatten_call_tree(tree)
        if len(lines) > max_lines_per_chain:
            lines = lines[:max_lines_per_chain] + [f"  ... ({len(lines) - max_lines_per_chain} more lines truncated)"]
        parts.append(f"### {chain_name}\n" + "\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build call chains from C project")
    parser.add_argument("--root", required=True, help="Project root directory")
    parser.add_argument("--config", default="callchain_entries.yaml",
                        help="YAML file with entry function definitions")
    parser.add_argument("--output", default="cache/callchains.json",
                        help="Output JSON file")
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    print(f"[callchain] parsing project at {args.root} ...")
    file_map = parse_project(args.root)
    total_funcs = sum(len(v) for v in file_map.values())
    print(f"[callchain] found {total_funcs} functions across {len(file_map)} files")

    call_graph = build_call_graph(file_map)
    entries = load_entries_config(args.config)
    chains = build_all_callchains(entries, call_graph, max_depth=args.max_depth)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_callchains(chains, args.output)

    # Print summary
    for chain_name, tree in chains.items():
        funcs_in_chain = get_all_functions_in_tree(tree)
        print(f"  {chain_name}: {len(funcs_in_chain)} functions reachable")

"""
Struct / union / enum definition indexer for C projects.

During preprocessing, walks all .h and .c files and builds a flat index of
every struct/union/enum definition found. The index is saved to
struct_index.json in the cache directory.

During generation, the implementer loads this index and injects the
definitions of relevant structs into the implementation prompt so the LLM
can reference correct field names and types.
"""

import json
import os

from preprocess.c_parser import parse_project_structs


def build_struct_index(project_root: str) -> dict:
    """
    Walk all .h and .c files under project_root and extract every
    struct/union/enum definition.

    Returns:
        {
            "StructName": {
                "file":       "relative/path/to/file.h",
                "start_line": 42,
                "end_line":   65,
                "body":       "typedef struct StructName { ... } StructName;",
                "kind":       "struct"  # or "union" / "enum"
            },
            ...
        }
    First occurrence wins when the same name appears in multiple files.
    """
    file_map = parse_project_structs(project_root, extensions=(".h", ".c"))
    index: dict = {}
    for _rel_path, structs in file_map.items():
        for s in structs:
            if s.name not in index:
                index[s.name] = {
                    "file":       s.file,
                    "start_line": s.start_line,
                    "end_line":   s.end_line,
                    "body":       s.body,
                    "kind":       s.kind,
                }
    return index


def save_struct_index(index: dict, path: str) -> None:
    """Persist the struct index to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_struct_index(path: str) -> dict:
    """Load a previously saved struct index, returning {} if the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_relevant_structs(
    plan: dict,
    struct_index: dict,
    max_structs: int = 20,
) -> list:
    """
    Identify struct names that are relevant to the given implementation plan.

    Two-pass heuristic:
      1. Structs defined in files that match the plan's affected_files basenames
         (e.g. if btree.c is affected, structs from btree.c and btree.h are included)
      2. Struct names that appear as whole-word tokens in the plan text

    Returns a list of struct names (no duplicates, capped at max_structs).
    """
    if not struct_index:
        return []

    # Pass 1 – structs whose file base-name matches an affected file
    affected_bases = {
        os.path.splitext(os.path.basename(f))[0]
        for f in plan.get("affected_files", [])
    }
    pass1 = [
        name for name, info in struct_index.items()
        if os.path.splitext(os.path.basename(info.get("file", "")))[0] in affected_bases
    ]

    # Pass 2 – struct names appearing in the plan text
    plan_text = " ".join([
        plan.get("implementation_plan", ""),
        " ".join(plan.get("functions_to_modify", [])),
        " ".join(fn.get("name", "") for fn in plan.get("functions_to_add", [])),
    ])
    import re
    pass2 = [
        name for name in struct_index
        if name not in pass1 and re.search(r'\b' + re.escape(name) + r'\b', plan_text)
    ]

    combined = pass1 + pass2
    return combined[:max_structs]


def format_structs_for_prompt(names: list, struct_index: dict) -> str:
    """
    Build a formatted block of struct definitions for inclusion in an LLM prompt.

    Args:
        names: list of struct names to include
        struct_index: the loaded struct index

    Returns:
        Multi-section string, one fenced code block per struct, or "" if empty.
    """
    if not names or not struct_index:
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


def format_struct_names_for_prompt(struct_index: dict, max_names: int = 60) -> str:
    """
    Return a compact listing of all known struct/union/enum names for use in
    the planner prompt so the LLM knows what types exist in the codebase.

    Returns a plain comma-separated string, or "" if the index is empty.
    """
    if not struct_index:
        return ""
    names = sorted(struct_index.keys())[:max_names]
    suffix = f" ... ({len(struct_index)} total)" if len(struct_index) > max_names else ""
    return ", ".join(names) + suffix

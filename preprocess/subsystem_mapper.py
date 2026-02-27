"""
Subsystem mapper.

Sends the project directory tree to the LLM and asks it to generate
semantic labels and keyword lists for each directory. The result is
cached as subsystem_map.json and used in Layer 1 retrieval (zero LLM calls
at query time — pure keyword matching).

Usage:
    python subsystem_mapper.py --root /path/to/project --output cache/subsystem_map.json
"""

import json
import os
from typing import Optional

from llm.client import LLMClient

SUBSYSTEM_SCHEMA_NAME = "subsystem_map"

SUBSYSTEM_JSON_SCHEMA = {
    "type": "object",
    "description": "Map of directory path (forward slashes, no trailing slash) to its subsystem metadata",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Human-readable subsystem name",
            },
            "description": {
                "type": "string",
                "description": "What this subsystem is responsible for",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search keywords in both English and Chinese; include synonyms",
            },
        },
        "required": ["name", "description", "keywords"],
    },
}

SYSTEM_PROMPT = """\
You are a database systems architecture expert, familiar with the code organization of PostgreSQL, MySQL, TiDB, and similar databases."""

MAPPING_PROMPT_TEMPLATE = """\
Below is the directory structure of a C database project.
For each non-empty directory, generate semantic labels to build a subsystem index for code retrieval.

Directory structure:
{dir_tree}

Guidelines:
- keywords must include synonyms in both English and Chinese so developers can search in either language
- Use medium granularity: too fine (one entry per file) or too coarse (one entry for all of src/) both reduce retrieval quality
- If a directory name is already self-explanatory (e.g. executor/), keywords should capture multiple ways to express its function
- Use forward slashes in directory paths, no trailing slash
- Example of a good entry for a storage directory:
    name: "Storage Engine"
    description: "Handles disk page read/write, heap file management, and free space tracking"
    keywords: ["storage", "heap", "page", "buffer", "fsm", "disk", "block", "存储", "堆文件", "页面"]

Return the result as a JSON object mapping each directory path to its subsystem metadata.
"""


def _collect_dir_tree(root: str, max_depth: int = 4, extensions: tuple = (".c", ".h")) -> str:
    """
    Walk the project directory and produce a tree string.
    Only includes directories that contain C/H source files.
    """
    lines = []

    def _walk(dirpath: str, depth: int, prefix: str):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(dirpath))
        except PermissionError:
            return

        dirs = [e for e in entries if os.path.isdir(os.path.join(dirpath, e))
                and not e.startswith(".")]
        c_files = [e for e in entries if any(e.endswith(ext) for ext in extensions)]

        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel == ".":
            rel = "(root)"

        if c_files:
            lines.append(f"{prefix}{rel}/  [{len(c_files)} files]")
        else:
            lines.append(f"{prefix}{rel}/")

        for d in dirs:
            _walk(os.path.join(dirpath, d), depth + 1, prefix + "  ")

    _walk(root, 0, "")
    return "\n".join(lines)



def generate_subsystem_map(
    project_root: str,
    client: LLMClient,
    model: str = "",
) -> dict:
    """
    Call the LLM once (or a few times for large projects) to generate
    the subsystem semantic map for the project directory tree.
    """
    model = model or client.default_model("summarizer")
    dir_tree = _collect_dir_tree(project_root)
    print(f"[subsystem] directory tree ({dir_tree.count(chr(10))} lines)")

    prompt = MAPPING_PROMPT_TEMPLATE.format(dir_tree=dir_tree)

    parsed = client.chat_structured(
        model=model,
        system=SYSTEM_PROMPT,
        user=prompt,
        max_tokens=4096,
        schema_name=SUBSYSTEM_SCHEMA_NAME,
        json_schema=SUBSYSTEM_JSON_SCHEMA,
    )

    if not parsed:
        print("[warn] subsystem map structured output returned empty dict")

    # Normalize paths: ensure forward slashes, strip trailing slash
    normalized = {}
    for k, v in parsed.items():
        k = k.replace("\\", "/").rstrip("/")
        normalized[k] = v

    return normalized


def save_subsystem_map(subsystem_map: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subsystem_map, f, ensure_ascii=False, indent=2)
    print(f"[subsystem] saved {len(subsystem_map)} subsystem entries to {output_path}")


def load_subsystem_map(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate subsystem semantic map from project directory")
    p.add_argument("--root", required=True, help="Project root directory")
    p.add_argument("--output", default="cache/subsystem_map.json")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = p.parse_args()

    from llm.client import LLMConfig
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY (or LLM_API_KEY) environment variable not set")

    client = LLMClient(LLMConfig(api_key=api_key))
    smap = generate_subsystem_map(args.root, client, model=args.model)
    save_subsystem_map(smap, args.output)

    print("\nGenerated subsystem map:")
    for path, info in smap.items():
        print(f"  {path}: {info.get('name', '')} — {', '.join(info.get('keywords', [])[:4])}")

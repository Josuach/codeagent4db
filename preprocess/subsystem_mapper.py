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
import re
from typing import Optional

from llm.client import LLMClient

SYSTEM_PROMPT = """你是数据库系统架构专家，熟悉 PostgreSQL、MySQL、TiDB 等数据库的代码组织方式。"""

MAPPING_PROMPT_TEMPLATE = """以下是一个 C 语言数据库项目的目录结构。
请为每个非空目录生成语义标签，用于构建代码检索的子系统索引。

目录结构：
{dir_tree}

输出格式（严格 JSON，key 为目录路径，路径使用正斜杠分隔）：
{{
  "src/storage": {{
    "name": "存储引擎",
    "description": "负责磁盘页面读写、堆文件管理、空闲空间追踪",
    "keywords": ["存储", "堆文件", "页面", "缓冲", "storage", "heap", "page", "buffer", "fsm"]
  }},
  "src/optimizer": {{
    "name": "查询优化器",
    "description": "将逻辑查询计划转换为最优物理执行计划，包含代价估算和连接顺序选择",
    "keywords": ["优化", "代价", "连接", "索引选择", "optimizer", "cost", "join", "plan", "Columbia"]
  }},
  ...
}}

注意：
- keywords 要包含中英文同义词（开发者可能用中文或英文描述需求）
- 粒度适中：太细（每个文件一个条目）或太粗（整个 src 一个条目）都会降低检索效果
- 若目录名已经很清晰（如 executor/），keywords 要包含其功能的多种表述"""


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


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


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

    raw = client.chat(model=model, system=SYSTEM_PROMPT, user=prompt, max_tokens=4096)
    parsed = _parse_llm_json(raw)

    if not parsed:
        print("[warn] subsystem map LLM response could not be parsed as JSON")
        print("Raw response:", raw[:500])

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

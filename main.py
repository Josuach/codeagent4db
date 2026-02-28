"""
Code Agent main entry point.

Two modes:
  1. preprocess  — Build all cache artifacts (function index, call chains, subsystem map)
  2. generate    — Given a feature description, generate unified diffs

Usage:
    # First time (or after code changes): run preprocessing
    python main.py preprocess --project /path/to/db_project

    # Generate code for a feature (reads from the same cache dir)
    python main.py generate --project /path/to/db_project --feature "描述特性"

    # Use a custom cache directory (so one codeagent instance handles multiple projects)
    python main.py preprocess --project /path/to/db_project --cache-dir /data/sqlite_cache
    python main.py generate  --project /path/to/db_project --cache-dir /data/sqlite_cache --feature "描述特性"

    # Incremental preprocessing (only re-process changed files)
    python main.py preprocess --project /path/to/db_project --incremental
"""

import argparse
import json
import os
import sys

from llm.client import LLMClient, LLMConfig, create_client_from_env

# ---------------------------------------------------------------------------
# Path setup: allow imports from preprocess/ and retrieval/ as packages
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from preprocess.c_parser import (
    parse_project,
    build_call_graph,
    build_reverse_call_graph,
)
from preprocess.callchain_builder import (
    build_all_callchains,
    load_entries_config,
    save_callchains,
    format_callchains_for_prompt,
)
from preprocess.batch_summarizer import (
    build_function_index,
    build_file_index,
    save_index,
    load_index,
)
from preprocess.subsystem_mapper import (
    generate_subsystem_map,
    save_subsystem_map,
    load_subsystem_map,
)
from preprocess.incremental import find_changed_files
from preprocess.struct_indexer import (
    build_struct_index,
    save_struct_index,
    load_struct_index,
)

from retrieval.layer1_subsystem import locate_subsystems, filter_functions_by_subsystem
from retrieval.layer2_bm25 import bm25_search_scoped
from retrieval.layer3_callgraph import expand_with_callgraph, format_candidates_for_prompt

from agent.planner import plan_feature, plan_feature_with_feedback, format_plan_for_display
from agent.implementer import implement_feature, integrate_interfaces
from agent.checker import compile_check_and_fix


# ---------------------------------------------------------------------------
# Default paths (relative to codeagent directory)
# ---------------------------------------------------------------------------

_AGENT_DIR = os.path.dirname(__file__)
DEFAULT_CACHE_DIR = os.path.join(_AGENT_DIR, "cache")


def _cache_paths(cache_dir: str) -> dict:
    """Return a dict of all cache file paths rooted at cache_dir."""
    return {
        "function_index": os.path.join(cache_dir, "function_index.json"),
        "file_index":     os.path.join(cache_dir, "file_index.json"),
        "subsystem_map":  os.path.join(cache_dir, "subsystem_map.json"),
        "callchains":     os.path.join(cache_dir, "callchains.json"),
        "file_hashes":    os.path.join(cache_dir, "file_hashes.json"),
        "struct_index":   os.path.join(cache_dir, "struct_index.json"),
    }


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def cmd_preprocess(args, client: LLMClient):
    project_root = os.path.abspath(args.project)
    cache_dir = os.path.abspath(args.cache_dir)
    config_dir = os.path.abspath(args.config_dir)
    callchain_entries = os.path.join(config_dir, "callchain_entries.yaml")
    os.makedirs(cache_dir, exist_ok=True)
    paths = _cache_paths(cache_dir)

    print(f"[main] preprocessing project: {project_root}")
    print(f"[main] cache directory: {cache_dir}")
    print(f"[main] config directory: {config_dir}")

    # --- Step 1: Parse C source files ---
    if args.incremental:
        changed_files, tracker = find_changed_files(project_root, paths["file_hashes"])
        print(f"[main] incremental mode: {len(changed_files)} changed files")
        if not changed_files:
            print("[main] nothing changed, skipping re-parse")
        # Still need full file_map for call graph (can't build partial call graph)
    else:
        tracker = None

    print("[main] parsing C source files ...")
    file_map = parse_project(project_root)
    total_funcs = sum(len(v) for v in file_map.values())
    print(f"[main] found {total_funcs} functions across {len(file_map)} files")

    # --- Step 2: Build call graph and call chains ---
    print("[main] building call graph ...")
    call_graph = build_call_graph(file_map)

    if os.path.exists(callchain_entries):
        entries = load_entries_config(callchain_entries)
        print(f"[main] building call chains for {len(entries)} entry points ...")
        chains = build_all_callchains(entries, call_graph)
        save_callchains(chains, paths["callchains"])
    else:
        print(f"[warn] {callchain_entries} not found, skipping call chain build")
        chains = {}

    # --- Step 3: LLM batch function summarization ---
    existing_index = {}
    skip_names = None
    if getattr(args, 'resume', False):
        # Resume from checkpoint: skip all functions already in the checkpoint index
        existing_index = load_index(paths["function_index"])
        if existing_index:
            skip_names = {name for name, info in existing_index.items() if info.get("summary")}
            print(f"[main] resume mode: skipping {len(skip_names)} already-summarized functions")
        else:
            print("[main] resume mode: no checkpoint found, starting fresh")
    elif args.incremental:
        existing_index = load_index(paths["function_index"])
        if tracker is not None:
            # Only re-summarize functions from changed files
            changed_set = {os.path.relpath(f, project_root).replace("\\", "/")
                          for f in changed_files}
            skip_names = {
                name for name, info in existing_index.items()
                if info.get("file") not in changed_set
            }
            print(f"[main] skipping {len(skip_names)} already-indexed functions")

    print("[main] generating LLM function summaries ...")
    func_index = build_function_index(
        file_map, client,
        batch_size=args.batch_size,
        model=args.summarizer_model,
        existing_index=existing_index,
        skip_names=skip_names,
        checkpoint_path=paths["function_index"],
    )
    save_index(func_index, paths["function_index"])

    # --- Step 4: Build file index (derived, no LLM calls) ---
    file_index = build_file_index(file_map, func_index)
    save_index(file_index, paths["file_index"])

    # --- Step 5: Generate subsystem map (LLM, one-time) ---
    if not os.path.exists(paths["subsystem_map"]) or args.regen_subsystem:
        print("[main] generating subsystem semantic map ...")
        smap = generate_subsystem_map(project_root, client, model=args.summarizer_model)
        save_subsystem_map(smap, paths["subsystem_map"])
    else:
        print("[main] subsystem map already exists, skipping (use --regen-subsystem to refresh)")

    # --- Update file hashes ---
    if tracker is not None:
        all_files = [
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(project_root)
            for fname in fnames
            if fname.endswith((".c", ".h"))
        ]
        tracker.update(all_files)
        tracker.save()

    # --- Step 6: Build struct / union / enum index (no LLM calls) ---
    print("[main] building struct/union/enum index ...")
    struct_idx = build_struct_index(project_root)
    save_struct_index(struct_idx, paths["struct_index"])
    print(f"  Struct index:   {len(struct_idx)} definitions → {paths['struct_index']}")

    print("[main] preprocessing complete.")
    print(f"  Function index: {len(func_index)} functions  →  {paths['function_index']}")
    print(f"  File index:     {len(file_index)} files      →  {paths['file_index']}")
    print(f"  Subsystem map:  {paths['subsystem_map']}")
    print(f"  Call chains:    {paths['callchains']}")


# ---------------------------------------------------------------------------
# Interactive plan review helper
# ---------------------------------------------------------------------------

def _interactive_plan_review(
    plan: dict,
    feature_desc: str,
    project_overview: str,
    callchains_text: str,
    client: LLMClient,
    model: str,
) -> dict:
    """
    Display the current plan and enter an interactive loop:
      - Empty input / 'y' / 'yes' / 'ok'  → accept plan and return it
      - 'abort' / 'quit'                   → return None (caller should exit)
      - Any other text                     → treat as feedback, re-plan, repeat

    Returns the accepted plan dict, or None if the user aborted.
    """
    while True:
        print()
        print(format_plan_for_display(plan))
        print()
        print("Press Enter to proceed with this plan, type feedback to revise it,")
        print("or type 'abort' to quit.")
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if user_input.lower() in ("", "y", "yes", "ok", "proceed", "continue"):
            return plan

        if user_input.lower() in ("abort", "quit", "exit", "q"):
            return None

        # User gave feedback — re-plan
        print(f"\n[agent] Revising plan based on feedback ...")
        plan = plan_feature_with_feedback(
            feature_desc=feature_desc,
            project_overview=project_overview,
            callchains_text=callchains_text,
            client=client,
            feedback=user_input,
            prev_plan=plan,
            model=model,
        )


# ---------------------------------------------------------------------------
# Feature generation pipeline
# ---------------------------------------------------------------------------

def cmd_generate(args, client: LLMClient):
    project_root = os.path.abspath(args.project)
    cache_dir = os.path.abspath(args.cache_dir)
    config_dir = os.path.abspath(args.config_dir)
    project_overview_path = os.path.join(config_dir, "project_overview.md")
    paths = _cache_paths(cache_dir)
    feature_desc = args.feature

    print(f"[main] feature: {feature_desc}")
    print(f"[main] cache directory: {cache_dir}")
    print(f"[main] config directory: {config_dir}")

    # Load cache artifacts
    func_index = load_index(paths["function_index"])
    subsystem_map = load_subsystem_map(paths["subsystem_map"])
    struct_index = load_struct_index(paths["struct_index"])
    if struct_index:
        print(f"[main] loaded struct index: {len(struct_index)} definitions")
    else:
        print("[main] struct_index.json not found — run preprocess to build it")

    if not func_index:
        sys.exit(
            f"[error] function_index.json not found in {cache_dir}.\n"
            f"        Run: python main.py preprocess --project {args.project} --cache-dir {args.cache_dir}"
        )

    callchains_text = ""
    if os.path.exists(paths["callchains"]):
        with open(paths["callchains"], "r", encoding="utf-8") as f:
            chains = json.load(f)
        callchains_text = format_callchains_for_prompt(chains)

    project_overview = ""
    if os.path.exists(project_overview_path):
        with open(project_overview_path, "r", encoding="utf-8") as f:
            project_overview = f.read()

    # Build reverse call graph from function index
    call_graph = {name: info.get("direct_calls", []) for name, info in func_index.items()}
    reverse_cg: dict[str, list[str]] = {}
    for caller, callees in call_graph.items():
        for callee in callees:
            reverse_cg.setdefault(callee, []).append(caller)

    # --- Layer 1: Subsystem localization ---
    candidate_dirs = locate_subsystems(feature_desc, subsystem_map)
    print(f"[retrieval] Layer 1 matched directories: {candidate_dirs}")

    # --- Layer 2: Scoped BM25 ---
    scoped_index = filter_functions_by_subsystem(func_index, candidate_dirs)
    if not scoped_index:
        print("[retrieval] Layer 1 returned no matches, falling back to full index")
        scoped_index = func_index

    bm25_results = bm25_search_scoped(feature_desc, scoped_index, top_k=25)
    print(f"[retrieval] Layer 2 BM25 top results: {bm25_results[:5]} ...")

    # --- Layer 3: Call graph expansion ---
    expanded = expand_with_callgraph(bm25_results, call_graph, reverse_cg)
    print(f"[retrieval] Layer 3 expanded to {len(expanded)} candidate functions")

    candidates_text = format_candidates_for_prompt(expanded, func_index)

    # --- Call 1: Planning ---
    print("[agent] Call 1: planning ...")
    plan = plan_feature(
        feature_desc=feature_desc,
        candidates_text=candidates_text,
        project_overview=project_overview,
        callchains_text=callchains_text,
        client=client,
        model=args.model,
        struct_index=struct_index,
    )

    # --- Interactive plan review ---
    if not getattr(args, "no_interactive", False):
        plan = _interactive_plan_review(
            plan=plan,
            feature_desc=feature_desc,
            project_overview=project_overview,
            callchains_text=callchains_text,
            client=client,
            model=args.model,
        )
        if plan is None:
            print("[main] Aborted by user.")
            return
    else:
        print(format_plan_for_display(plan))

    # --- Call 2: Implementation ---
    print("[agent] Call 2: generating code ...")
    diff_main = implement_feature(plan, project_root, client, model=args.model,
                                  function_index=func_index, struct_index=struct_index)

    # --- Call 3 (if high complexity): Interface integration ---
    diff_headers = ""
    if plan.get("complexity") == "high" and plan.get("header_changes"):
        print("[agent] Call 3: integrating interfaces (high complexity feature) ...")
        diff_headers = integrate_interfaces(plan, diff_main, project_root, client, model=args.model)

    # --- Output ---
    if args.output:
        output_path = args.output
    elif args.diff_name:
        output_path = os.path.join(cache_dir, f"{args.diff_name}.diff")
    else:
        safe_name = feature_desc[:20].replace(" ", "_").replace("/", "_")
        output_path = os.path.join(cache_dir, f"output_{safe_name}.diff")

    full_diff = diff_main
    if diff_headers:
        full_diff = full_diff + "\n\n" + diff_headers

    # --- Optional compile check ---
    if getattr(args, "compile_check", False):
        print("\n[checker] Running compile check ...")
        full_diff, compile_ok, compile_log = compile_check_and_fix(
            diff_text=full_diff,
            plan=plan,
            project_root=project_root,
            compile_cmd=getattr(args, "compile_cmd", ""),
            client=client,
            model=args.model,
        )
        print(compile_log)
        if compile_ok:
            print("[checker] ✓ Compile check passed.")
        else:
            print("[checker] ✗ Compile check failed (diff may still have issues).")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_diff)

    print(f"\n[main] Done. Diff written to: {output_path}")
    print(f"       Apply with: patch -p1 --fuzz=5 < {output_path}")

    # Also print plan summary
    print("\n=== Implementation Plan ===")
    print(plan.get("implementation_plan", ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based code agent for C database projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Global LLM provider options ---
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "deepseek"),
        choices=["deepseek", "anthropic", "openai_compatible"],
        help="LLM provider to use (default: deepseek; env: LLM_PROVIDER)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key (overrides env vars). "
            "Env fallback: DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / LLM_API_KEY"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL"),
        help="Base URL for OpenAI-compatible providers (env: LLM_BASE_URL)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- preprocess command ---
    prep = sub.add_parser("preprocess", help="Build preprocessing cache (function index, call chains, subsystem map)")
    prep.add_argument("--project", required=True, help="Path to the C project root")
    prep.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR,
        help=(
            "Directory to store all cache files (function_index.json, callchains.json, etc.). "
            f"Default: <codeagent>/cache. Use the same value for 'generate' to reuse this index."
        ),
    )
    prep.add_argument("--incremental", action="store_true",
                      help="Only re-process changed files")
    prep.add_argument("--resume", action="store_true",
                      help="Resume from checkpoint: skip functions already in the index")
    prep.add_argument("--regen-subsystem", action="store_true",
                      help="Regenerate subsystem map even if it exists")
    prep.add_argument("--batch-size", type=int, default=40,
                      help="Functions per LLM batch (default: 40)")
    prep.add_argument("--summarizer-model", default="",
                      help="Model for preprocessing (default: provider's summarizer default)")
    prep.add_argument(
        "--config-dir", default=_AGENT_DIR,
        help="Directory containing callchain_entries.yaml and project_overview.md (default: codeagent root)",
    )

    # --- generate command ---
    gen = sub.add_parser("generate", help="Generate code changes for a feature")
    gen.add_argument("--project", required=True, help="Path to the C project root")
    gen.add_argument("--feature", required=True, help="Feature description")
    gen.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR,
        help=(
            "Directory containing the preprocessing cache (must match the value used in 'preprocess'). "
            f"Default: <codeagent>/cache. Output diff is also written here unless --output is set."
        ),
    )
    gen.add_argument("--output", default=None,
                     help="Output diff file path (default: <cache-dir>/output_<feature>.diff)")
    gen.add_argument("--diff-name", default=None,
                     help="Short name for the output diff (saved as <cache-dir>/<name>.diff); ignored if --output is set")
    gen.add_argument("--model", default="",
                     help="Model for planning and implementation (default: provider's main default)")
    gen.add_argument(
        "--config-dir", default=_AGENT_DIR,
        help="Directory containing project_overview.md (default: codeagent root)",
    )
    gen.add_argument("--compile-check", action="store_true",
                     help="After generating diff, apply it to a temp copy and compile to verify correctness")
    gen.add_argument("--compile-cmd", default="",
                     help="Compile command to run for --compile-check (e.g. 'gcc -c mdb.c', 'make'); "
                          "if empty, tries 'make' then falls back to compiling .c files with gcc")
    gen.add_argument("--no-interactive", action="store_true",
                     help="Skip the interactive plan review step and proceed directly to implementation")

    args = parser.parse_args()

    # --- Build LLM client ---
    if args.api_key:
        config = LLMConfig(provider=args.provider, api_key=args.api_key, base_url=args.base_url)
        client = LLMClient(config)
    else:
        client = create_client_from_env(provider=args.provider, base_url=args.base_url)

    print(f"[main] using provider: {args.provider} / model defaults: "
          f"main={client.default_model('main')}, summarizer={client.default_model('summarizer')}")

    if args.command == "preprocess":
        cmd_preprocess(args, client)
    elif args.command == "generate":
        cmd_generate(args, client)


if __name__ == "__main__":
    main()

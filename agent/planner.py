"""
Call 1: Planning agent.

Takes the feature description + retrieved candidate functions + project overview
and outputs a structured implementation plan:
  - affected files
  - new files / header changes
  - ordered steps (each step: description + functions_to_modify + functions_to_add)
  - complexity level (medium/high)
"""

from llm.client import LLMClient


PLAN_SCHEMA_NAME = "implementation_plan"

PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity": {
            "type": "string",
            "description": "Complexity level: 'medium' or 'high'. Use 'high' when changes span 3+ modules or modify core interfaces.",
        },
        "affected_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All file names touched by this feature (no directory path), e.g. 'build.c'",
        },
        "new_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New file names to create (empty list if none)",
        },
        "header_changes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Header file names that need changes (empty list if none)",
        },
        "relevant_structs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of structs/unions/enums whose definitions are needed for implementation (choose from the Available Structs list; empty list if none)",
        },
        "steps": {
            "type": "array",
            "description": "Ordered list of implementation steps. Each step is an atomic code change.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Concrete description of what this step implements, specific down to the function level",
                    },
                    "functions_to_modify": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Existing function names to modify in this step (no duplicates)",
                    },
                    "functions_to_add": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "in_file": {"type": "string", "description": "File name only, no path"},
                            },
                            "required": ["name", "in_file"],
                        },
                        "description": "New functions to add in this step",
                    },
                },
                "required": ["description", "functions_to_modify", "functions_to_add"],
            },
        },
    },
    "required": [
        "complexity", "affected_files", "new_files", "header_changes",
        "relevant_structs", "steps",
    ],
}

SYSTEM_PROMPT_TEMPLATE = """\
You are a senior C database kernel engineer.
Your task is to analyze the developer's feature request and produce a precise code modification plan.

# Project Architecture Overview
{project_overview}

# Key Call Chains
{callchains}
"""

USER_PROMPT_TEMPLATE = """\
## Feature Request
{feature_desc}

## Candidate Functions (filtered by relevance, ordered by score)
{candidates}

## Available Structs / Unions / Enums in this codebase
{structs_summary}

## Task
Produce an implementation plan in JSON format based on the feature description and candidate functions above.

Guidelines:
- affected_files: all file names touched (e.g. build.c), no directory paths
- steps: break the work into ordered atomic steps; each step must specify:
    - description: what this step does, concrete and function-level specific
    - functions_to_modify: existing functions changed in THIS step only
    - functions_to_add: new functions introduced in THIS step only
- If a function signature changes, include the corresponding header in header_changes
- relevant_structs: struct/union/enum names (from Available Structs above) needed by the implementer
"""

REFINE_USER_PROMPT_TEMPLATE = """\
## Feature Request
{feature_desc}

## Current Implementation Plan
{current_plan_json}

## User Feedback
{feedback}

## Task
Revise the implementation plan based on the user's feedback.
Keep all unchanged parts intact; only adjust what the feedback requires.
Produce the revised plan in the same JSON format.
"""


def _strip_src_prefix(path: str) -> str:
    """Remove leading src/ or src\\ directory prefix from file paths."""
    path = path.replace("\\", "/")
    if path.startswith("src/"):
        path = path[4:]
    return path


def _normalize_step(step: dict) -> dict:
    """Normalize a single step dict: fix types, strip src/ prefixes, deduplicate."""
    # functions_to_modify: flatten nested lists, deduplicate
    seen: set = set()
    deduped = []
    for fn in step.get("functions_to_modify", []):
        if isinstance(fn, list):
            fn = fn[0] if fn else ""
        fn = str(fn).strip()
        if fn and fn not in seen:
            seen.add(fn)
            deduped.append(fn)
    step["functions_to_modify"] = deduped

    # functions_to_add: coerce strings → {name, in_file=""}, strip src/ prefix
    normalized_adds = []
    for fn_add in step.get("functions_to_add", []):
        if isinstance(fn_add, str):
            fn_add = {"name": fn_add.strip(), "in_file": ""}
        elif not isinstance(fn_add, dict):
            continue
        if "in_file" in fn_add:
            fn_add["in_file"] = _strip_src_prefix(str(fn_add["in_file"]))
        normalized_adds.append(fn_add)
    step["functions_to_add"] = normalized_adds

    # description: ensure string
    if not isinstance(step.get("description"), str):
        step["description"] = str(step.get("description", ""))

    return step


def _normalize_plan(plan: dict) -> dict:
    """
    Post-process the plan dict to fix common LLM output issues:
    - Strip src/ prefix from file paths
    - Normalize each step
    - Ensure required fields exist
    """
    if "affected_files" in plan:
        plan["affected_files"] = list(dict.fromkeys(
            _strip_src_prefix(f) for f in plan["affected_files"]
        ))

    if "new_files" in plan:
        plan["new_files"] = [_strip_src_prefix(f) for f in plan["new_files"]]

    if "header_changes" in plan:
        plan["header_changes"] = list(dict.fromkeys(
            _strip_src_prefix(f) for f in plan["header_changes"]
        ))

    # Normalize each step
    steps = plan.get("steps", [])
    if isinstance(steps, list):
        plan["steps"] = [_normalize_step(s) for s in steps if isinstance(s, dict)]
    else:
        plan["steps"] = []

    # Back-compat: derive top-level functions_to_modify / functions_to_add
    # so callers that still read these fields continue to work.
    all_mods: list = []
    seen_mods: set = set()
    all_adds: list = []
    seen_adds: set = set()
    for step in plan["steps"]:
        for fn in step.get("functions_to_modify", []):
            if fn not in seen_mods:
                seen_mods.add(fn)
                all_mods.append(fn)
        for fa in step.get("functions_to_add", []):
            key = fa.get("name", "")
            if key and key not in seen_adds:
                seen_adds.add(key)
                all_adds.append(fa)
    plan["functions_to_modify"] = all_mods
    plan["functions_to_add"] = all_adds

    return plan


def _structs_summary(struct_index: dict) -> str:
    """Return a compact list of known struct names for the planner prompt."""
    if not struct_index:
        return "(no struct index available)"
    names = sorted(struct_index.keys())
    MAX = 80
    suffix = f" ... ({len(names)} total)" if len(names) > MAX else ""
    return ", ".join(names[:MAX]) + suffix


def plan_feature(
    feature_desc: str,
    candidates_text: str,
    project_overview: str,
    callchains_text: str,
    client: LLMClient,
    model: str = "",
    struct_index: dict = None,
) -> dict:
    """
    Call the LLM to produce an implementation plan.

    Returns:
        Parsed plan dict with keys: complexity, affected_files, new_files,
        header_changes, relevant_structs, steps (+ derived functions_to_modify /
        functions_to_add for backward compatibility)
    """
    model = model or client.default_model("main")

    system = SYSTEM_PROMPT_TEMPLATE.format(
        project_overview=project_overview,
        callchains=callchains_text,
    )
    user = USER_PROMPT_TEMPLATE.format(
        feature_desc=feature_desc,
        candidates=candidates_text,
        structs_summary=_structs_summary(struct_index or {}),
    )

    plan = client.chat_structured(
        model=model,
        system=system,
        user=user,
        max_tokens=8192,
        schema_name=PLAN_SCHEMA_NAME,
        json_schema=PLAN_JSON_SCHEMA,
    )

    if not plan:
        print("[warn] planner structured output returned empty dict; using fallback.")
        return {
            "complexity": "unknown",
            "affected_files": [],
            "new_files": [],
            "header_changes": [],
            "relevant_structs": [],
            "steps": [],
            "functions_to_modify": [],
            "functions_to_add": [],
            "_parse_error": True,
        }

    return _normalize_plan(plan)


def plan_feature_with_feedback(
    feature_desc: str,
    project_overview: str,
    callchains_text: str,
    client: LLMClient,
    feedback: str,
    prev_plan: dict,
    model: str = "",
) -> dict:
    """
    Re-run planning with user feedback to refine the previous plan.

    Returns:
        Revised plan dict (falls back to prev_plan if the LLM call fails)
    """
    import json as _json
    model = model or client.default_model("main")

    system = SYSTEM_PROMPT_TEMPLATE.format(
        project_overview=project_overview,
        callchains=callchains_text,
    )
    user = REFINE_USER_PROMPT_TEMPLATE.format(
        feature_desc=feature_desc,
        current_plan_json=_json.dumps(prev_plan, ensure_ascii=False, indent=2),
        feedback=feedback,
    )

    plan = client.chat_structured(
        model=model,
        system=system,
        user=user,
        max_tokens=8192,
        schema_name=PLAN_SCHEMA_NAME,
        json_schema=PLAN_JSON_SCHEMA,
    )

    if not plan:
        print("[warn] plan refinement returned empty dict; keeping previous plan.")
        return prev_plan

    return _normalize_plan(plan)


def format_plan_for_display(plan: dict) -> str:
    """Return a human-readable summary of the implementation plan."""
    lines = ["=" * 60, "  Implementation Plan", "=" * 60]
    lines.append(f"Complexity   : {plan.get('complexity', 'unknown')}")

    affected = plan.get("affected_files", [])
    lines.append(f"Affected files ({len(affected)}): {', '.join(affected) or '(none)'}")

    new_files = plan.get("new_files", [])
    if new_files:
        lines.append(f"New files    : {', '.join(new_files)}")

    headers = plan.get("header_changes", [])
    if headers:
        lines.append(f"Header changes: {', '.join(headers)}")

    structs = plan.get("relevant_structs", [])
    if structs:
        lines.append(f"Relevant structs: {', '.join(structs)}")

    steps = plan.get("steps", [])
    lines.append(f"\nSteps ({len(steps)}):")
    for i, step in enumerate(steps, 1):
        desc = step.get("description", "(no description)")
        mods = step.get("functions_to_modify", [])
        adds = [fa.get("name", "?") for fa in step.get("functions_to_add", [])]
        lines.append(f"  {i}. {desc}")
        if mods:
            lines.append(f"     modify : {', '.join(mods)}")
        if adds:
            lines.append(f"     add    : {', '.join(adds)}")

    lines.append("=" * 60)
    return "\n".join(lines)

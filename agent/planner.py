"""
Call 1: Planning agent.

Takes the feature description + retrieved candidate functions + project overview
and outputs a structured implementation plan:
  - affected files
  - functions to modify / add
  - new files needed
  - step-by-step implementation plan
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
            "description": "File names only (no directory path), e.g. 'build.c'",
        },
        "functions_to_modify": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Existing function names to modify (at most 10 most critical, no duplicates)",
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
        "implementation_plan": {
            "type": "string",
            "description": "Step-by-step implementation instructions, specific down to the function level",
        },
    },
    "required": [
        "complexity", "affected_files", "functions_to_modify",
        "functions_to_add", "new_files", "header_changes",
        "relevant_structs", "implementation_plan",
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
- affected_files: file names only (e.g. build.c), no directory paths
- functions_to_modify: at most 10 most critical existing functions, no duplicates
- If a function signature changes, include the corresponding header in header_changes
- relevant_structs: list struct/union/enum names (from Available Structs above) whose definitions the implementer will need; empty list if none
- implementation_plan: concrete steps down to the function level
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



def _normalize_plan(plan: dict) -> dict:
    """
    Post-process the plan dict to fix common LLM output issues:
    - Strip src/ prefix from file paths
    - Deduplicate functions_to_modify
    - Limit list sizes to avoid noise
    """
    if "affected_files" in plan:
        plan["affected_files"] = list(dict.fromkeys(
            _strip_src_prefix(f) for f in plan["affected_files"]
        ))

    if "functions_to_modify" in plan:
        # Flatten any nested lists and deduplicate while preserving order
        seen = set()
        deduped = []
        for fn in plan["functions_to_modify"]:
            if isinstance(fn, list):
                fn = fn[0] if fn else ""
            fn = str(fn)
            if fn and fn not in seen:
                seen.add(fn)
                deduped.append(fn)
        plan["functions_to_modify"] = deduped[:20]  # cap at 20

    if "implementation_plan" in plan:
        ip = plan["implementation_plan"]
        if isinstance(ip, list):
            plan["implementation_plan"] = "\n".join(str(item) for item in ip)

    if "functions_to_add" in plan:
        for fn_add in plan["functions_to_add"]:
            if "in_file" in fn_add:
                fn_add["in_file"] = _strip_src_prefix(fn_add["in_file"])

    if "new_files" in plan:
        plan["new_files"] = [_strip_src_prefix(f) for f in plan["new_files"]]

    if "header_changes" in plan:
        plan["header_changes"] = list(dict.fromkeys(
            _strip_src_prefix(f) for f in plan["header_changes"]
        ))

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

    Args:
        feature_desc: user's feature description
        candidates_text: formatted candidate functions (from layer3_callgraph.format_candidates_for_prompt)
        project_overview: contents of project_overview.md
        callchains_text: formatted call chains (from callchain_builder.format_callchains_for_prompt)
        client: LLMClient instance
        model: model to use (empty string = use client default)

    Returns:
        Parsed plan dict with keys: complexity, affected_files, functions_to_modify,
        functions_to_add, new_files, header_changes, implementation_plan
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
            "functions_to_modify": [],
            "functions_to_add": [],
            "new_files": [],
            "header_changes": [],
            "relevant_structs": [],
            "implementation_plan": "(structured output failed)",
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

    Args:
        feature_desc: original feature description
        project_overview: contents of project_overview.md
        callchains_text: formatted call chains
        client: LLMClient instance
        feedback: user's revision instructions
        prev_plan: the plan dict produced by the previous planning call
        model: model to use (empty string = use client default)

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

    mods = plan.get("functions_to_modify", [])
    lines.append(f"Modify ({len(mods)}): {', '.join(mods) or '(none)'}")

    adds = plan.get("functions_to_add", [])
    if adds:
        lines.append(f"Add ({len(adds)}):")
        for fa in adds:
            lines.append(f"  + {fa.get('name', '?')}  in {fa.get('in_file', '?')}")

    new_files = plan.get("new_files", [])
    if new_files:
        lines.append(f"New files    : {', '.join(new_files)}")

    headers = plan.get("header_changes", [])
    if headers:
        lines.append(f"Header changes: {', '.join(headers)}")

    structs = plan.get("relevant_structs", [])
    if structs:
        lines.append(f"Relevant structs: {', '.join(structs)}")

    lines.append("")
    lines.append("Steps:")
    impl = plan.get("implementation_plan", "(empty)")
    if isinstance(impl, list):
        impl = "\n".join(str(item) for item in impl)
    lines.append(str(impl))
    lines.append("=" * 60)
    return "\n".join(lines)

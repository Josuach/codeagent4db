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
        "implementation_plan": {
            "type": "string",
            "description": "Step-by-step implementation instructions, specific down to the function level",
        },
    },
    "required": [
        "complexity", "affected_files", "functions_to_modify",
        "functions_to_add", "new_files", "header_changes", "implementation_plan",
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

## Task
Produce an implementation plan in JSON format based on the feature description and candidate functions above.

Guidelines:
- affected_files: file names only (e.g. build.c), no directory paths
- functions_to_modify: at most 10 most critical existing functions, no duplicates
- If a function signature changes, include the corresponding header in header_changes
- implementation_plan: concrete steps down to the function level
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
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for fn in plan["functions_to_modify"]:
            if fn not in seen:
                seen.add(fn)
                deduped.append(fn)
        plan["functions_to_modify"] = deduped[:20]  # cap at 20

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


def plan_feature(
    feature_desc: str,
    candidates_text: str,
    project_overview: str,
    callchains_text: str,
    client: LLMClient,
    model: str = "",
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
            "implementation_plan": "(structured output failed)",
            "_parse_error": True,
        }

    return _normalize_plan(plan)

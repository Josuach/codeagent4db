"""
C source code static analyzer.
Extracts function definitions, signatures, bodies, and direct call relationships
using regex + bracket-depth matching. No external tools required.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional, List

# C keywords and common macros that look like function calls — exclude from call lists
C_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "return",
    "sizeof", "typeof", "alignof", "offsetof", "defined",
    "break", "continue", "goto", "default",
    # Common macros that are definitely not functions
    "Assert", "AssertArg", "AssertState", "StaticAssertStmt",
    "likely", "unlikely", "PG_TRY", "PG_CATCH", "PG_END_TRY",
}

# Patterns for identifiers that are almost certainly utility/logging, not business logic
UTILITY_BLACKLIST = {
    "palloc", "palloc0", "pfree", "repalloc",
    "malloc", "calloc", "realloc", "free",
    "memcpy", "memset", "memmove", "memcmp",
    "strcpy", "strncpy", "strcmp", "strncmp", "strlen", "strcat",
    "sprintf", "snprintf", "fprintf", "printf",
    "elog", "ereport", "errmsg", "errcode", "errdetail",
    "Assert", "MemSet",
}


@dataclass
class FunctionInfo:
    name: str
    signature: str          # full declaration: return_type name(params)
    file: str               # relative path from project root
    start_line: int
    end_line: int
    body: str               # full function body including braces
    calls: list[str] = field(default_factory=list)   # function names called inside body
    comment: str = ""       # leading comment block (if any)


def _strip_comments_for_parsing(source: str) -> tuple[str, dict[int, str]]:
    """
    Remove /* */ and // comments from source for structural parsing,
    but preserve line count by replacing comment content with spaces/newlines.
    Returns (stripped_source, {line_no: comment_text}) where comment blocks
    that appear immediately before a function are captured.
    """
    result = []
    i = 0
    n = len(source)
    while i < n:
        # Block comment
        if source[i:i+2] == "/*":
            end = source.find("*/", i + 2)
            if end == -1:
                end = n - 2
            # Preserve newlines so line numbers stay correct
            block = source[i:end+2]
            result.append("\n" * block.count("\n"))
            i = end + 2
        # Line comment
        elif source[i:i+2] == "//":
            end = source.find("\n", i)
            if end == -1:
                end = n
            result.append("\n")
            i = end
        # String literal — skip contents
        elif source[i] == '"':
            result.append('"')
            i += 1
            while i < n and source[i] != '"':
                if source[i] == '\\':
                    result.append(source[i:i+2])
                    i += 2
                else:
                    result.append(source[i])
                    i += 1
            if i < n:
                result.append('"')
                i += 1
        # Char literal
        elif source[i] == "'":
            result.append("'")
            i += 1
            while i < n and source[i] != "'":
                if source[i] == '\\':
                    result.append(source[i:i+2])
                    i += 2
                else:
                    result.append(source[i])
                    i += 1
            if i < n:
                result.append("'")
                i += 1
        else:
            result.append(source[i])
            i += 1
    return "".join(result)


def _find_matching_brace(source: str, open_pos: int) -> int:
    """
    Given the position of an opening '{', find its matching '}'.
    Returns the index of '}', or -1 if not found.
    """
    depth = 0
    i = open_pos
    n = len(source)
    while i < n:
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _extract_calls_from_body(body: str) -> list[str]:
    """
    Extract all function call names from a function body.
    Matches: identifier( pattern, excluding C keywords and obvious macros.
    """
    # Pattern: word character sequence followed immediately by '('
    raw_calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', body)
    seen = set()
    result = []
    for name in raw_calls:
        if name not in C_KEYWORDS and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _extract_leading_comment(source: str, func_start: int) -> str:
    """
    Look backwards from func_start to find a /* */ or // comment block
    that immediately precedes the function (allowing whitespace).
    """
    before = source[:func_start].rstrip()
    if before.endswith("*/"):
        start = before.rfind("/*")
        if start != -1:
            return before[start:].strip()
    # Check for consecutive // lines
    lines = before.split("\n")
    comment_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            comment_lines.insert(0, stripped)
        elif stripped == "":
            continue
        else:
            break
    if comment_lines:
        return "\n".join(comment_lines)
    return ""


# Regex for C function definition header.
# Strategy: find  name(params)  patterns where the closing ) is NOT followed
# by a semicolon (declaration) — then verify that { comes after.
# We match from the start of a line to catch top-level definitions.
# The return type + qualifiers may span multiple lines (K&R / Linux style).
_FUNC_DEF_RE = re.compile(
    r'^'                                       # start of line (MULTILINE)
    r'(?:(?:static|inline|extern|__inline__|__attribute__\s*\([^)]*\))\s+)*'  # optional qualifiers
    r'(?:[a-zA-Z_][a-zA-Z0-9_\s\*]*?)\s*'    # return type (permissive, non-greedy)
    r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*'         # (GROUP 1) function name
    r'\('                                      # opening paren
    r'([^;{]*?)'                              # (GROUP 2) params — no semicolon or brace
    r'\)\s*'                                   # closing paren (whitespace OK after)
    r'(?=[\s{])',                              # lookahead: followed by whitespace or {
    re.MULTILINE,
)


def extract_functions(filepath: str, project_root: str = "") -> list[FunctionInfo]:
    """
    Parse a C source file and return a list of FunctionInfo for every
    function definition found.

    Args:
        filepath: absolute or relative path to the .c file
        project_root: if provided, file paths in FunctionInfo are relative to this
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            original_source = f.read()
    except OSError:
        return []

    stripped = _strip_comments_for_parsing(original_source)
    lines = original_source.split("\n")  # for line number calculation

    rel_path = os.path.relpath(filepath, project_root) if project_root else filepath
    rel_path = rel_path.replace("\\", "/")

    functions: list[FunctionInfo] = []

    for m in _FUNC_DEF_RE.finditer(stripped):
        func_name = m.group(1)
        params_raw = m.group(2).strip()

        # The function body must start with '{' shortly after the header.
        # We look in the stripped source from match end, skipping whitespace.
        search_start = m.end()
        rest = stripped[search_start:]
        brace_offset = rest.find("{")
        if brace_offset == -1:
            continue
        # Make sure there's no ';' before the '{' (would be a declaration, not definition)
        before_brace = rest[:brace_offset]
        if ";" in before_brace:
            continue

        open_brace_pos = search_start + brace_offset
        close_brace_pos = _find_matching_brace(stripped, open_brace_pos)
        if close_brace_pos == -1:
            continue

        # Line numbers (1-based)
        start_line = stripped[:m.start()].count("\n") + 1
        end_line = stripped[:close_brace_pos].count("\n") + 1

        # Extract body from ORIGINAL source (with comments) for LLM context
        body = original_source[open_brace_pos:close_brace_pos + 1]

        # Build signature from original (not stripped) for readability
        sig_start = m.start()
        # Reconstruct signature from original source using same offsets
        orig_header = original_source[sig_start:search_start + brace_offset].strip()
        # Clean up: remove trailing whitespace/newlines, keep single line
        signature = " ".join(orig_header.split())

        calls = _extract_calls_from_body(stripped[open_brace_pos:close_brace_pos + 1])
        comment = _extract_leading_comment(original_source, m.start())

        functions.append(FunctionInfo(
            name=func_name,
            signature=signature,
            file=rel_path,
            start_line=start_line,
            end_line=end_line,
            body=body,
            calls=calls,
            comment=comment,
        ))

    return functions


def parse_project(project_root: str, extensions: tuple[str, ...] = (".c",)) -> dict[str, list[FunctionInfo]]:
    """
    Walk a project directory and extract all functions from matching source files.

    Returns:
        {relative_file_path: [FunctionInfo, ...]}
    """
    result: dict[str, list[FunctionInfo]] = {}
    for dirpath, _dirnames, filenames in os.walk(project_root):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                full_path = os.path.join(dirpath, fname)
                funcs = extract_functions(full_path, project_root)
                if funcs:
                    rel = os.path.relpath(full_path, project_root).replace("\\", "/")
                    result[rel] = funcs
    return result


def build_call_graph(file_map: dict[str, list[FunctionInfo]]) -> dict[str, list[str]]:
    """
    Build a project-wide call graph: {func_name: [called_func_names]}.
    Only includes calls to functions that are defined somewhere in the project
    (filters out stdlib calls etc.).

    Note: if two functions share the same name in different files, they are
    merged in this flat map (limitation of static analysis without full type info).
    """
    # Collect all known function names in the project
    all_defined: set[str] = set()
    for funcs in file_map.values():
        for f in funcs:
            all_defined.add(f.name)

    call_graph: dict[str, list[str]] = {}
    for funcs in file_map.values():
        for f in funcs:
            # Only keep calls to project-internal functions
            internal_calls = [c for c in f.calls if c in all_defined and c != f.name]
            call_graph[f.name] = internal_calls

    return call_graph


def build_reverse_call_graph(call_graph: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Invert the call graph: {func_name: [functions_that_call_it]}.
    """
    reverse: dict[str, list[str]] = {}
    for caller, callees in call_graph.items():
        for callee in callees:
            reverse.setdefault(callee, []).append(caller)
    return reverse


def get_function_by_name(name: str, file_map: dict[str, list[FunctionInfo]]) -> Optional[FunctionInfo]:
    """Look up a FunctionInfo by function name across all files."""
    for funcs in file_map.values():
        for f in funcs:
            if f.name == name:
                return f
    return None


# ---------------------------------------------------------------------------
# Struct / union / enum extraction
# ---------------------------------------------------------------------------

@dataclass
class StructInfo:
    name: str           # typedef alias or tag name
    kind: str           # "struct", "union", or "enum"
    file: str           # relative path from project root
    start_line: int
    end_line: int
    body: str           # full definition text (may be truncated for large defs)


# Matches the beginning of a struct/union/enum definition.
# Group 1: "typedef " prefix (optional)
# Group 2: kind (struct / union / enum)
# Group 3: optional tag name
_STRUCT_RE = re.compile(
    r'\b(typedef\s+)?'
    r'(struct|union|enum)\s+'
    r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\s*)?',
    re.MULTILINE,
)

# Max characters to store per struct body (avoid storing huge structs verbatim)
_MAX_STRUCT_BODY_CHARS = 3000


def extract_structs(filepath: str, project_root: str = "") -> List[StructInfo]:
    """
    Parse a C source or header file and return StructInfo for every
    struct/union/enum definition found.

    Args:
        filepath: absolute or relative path to the .c / .h file
        project_root: if provided, file paths in StructInfo are relative to this
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            original_source = f.read()
    except OSError:
        return []

    stripped = _strip_comments_for_parsing(original_source)
    orig_lines = original_source.split("\n")

    rel_path = os.path.relpath(filepath, project_root) if project_root else filepath
    rel_path = rel_path.replace("\\", "/")

    structs: List[StructInfo] = []
    seen_names: set = set()

    for m in _STRUCT_RE.finditer(stripped):
        is_typedef = bool(m.group(1))
        kind = m.group(2)        # "struct" / "union" / "enum"
        tag_name = m.group(3)    # may be None

        # Find the opening brace after the match
        brace_pos = stripped.find("{", m.end())
        if brace_pos == -1:
            continue

        # Skip forward declarations like "struct Foo;" or "typedef struct Foo Bar;"
        between = stripped[m.end():brace_pos]
        if ";" in between:
            continue

        close_brace_pos = _find_matching_brace(stripped, brace_pos)
        if close_brace_pos == -1:
            continue

        # For typedefs: the alias name follows the closing brace
        alias_name: Optional[str] = None
        alias_suffix_len = 0
        if is_typedef:
            after = stripped[close_brace_pos + 1:]
            alias_m = re.match(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*;', after)
            if alias_m:
                alias_name = alias_m.group(1)
                alias_suffix_len = alias_m.end()

        name = alias_name or tag_name
        if not name:
            continue  # anonymous with no typedef alias — skip
        if name in seen_names:
            continue  # take first definition only
        seen_names.add(name)

        # Compute line numbers from stripped (line counts are preserved)
        start_line = stripped[:m.start()].count("\n") + 1
        if alias_name:
            end_pos = close_brace_pos + 1 + alias_suffix_len
        else:
            end_pos = close_brace_pos + 1
        end_line = stripped[:end_pos].count("\n") + 1

        # Extract body from original source using line numbers (safe since
        # _strip_comments_for_parsing preserves newline count per line)
        body_lines = orig_lines[start_line - 1: end_line]
        body = "\n".join(body_lines)
        if len(body) > _MAX_STRUCT_BODY_CHARS:
            body = body[:_MAX_STRUCT_BODY_CHARS] + "\n    /* ... truncated ... */"

        structs.append(StructInfo(
            name=name,
            kind=kind,
            file=rel_path,
            start_line=start_line,
            end_line=end_line,
            body=body,
        ))

    return structs


def parse_project_structs(
    project_root: str,
    extensions: tuple = (".h", ".c"),
) -> dict[str, list[StructInfo]]:
    """
    Walk a project directory and extract all struct/union/enum definitions
    from matching files.

    Returns:
        {relative_file_path: [StructInfo, ...]}
    """
    result: dict[str, list[StructInfo]] = {}
    for dirpath, _dirnames, filenames in os.walk(project_root):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                full_path = os.path.join(dirpath, fname)
                structs = extract_structs(full_path, project_root)
                if structs:
                    rel = os.path.relpath(full_path, project_root).replace("\\", "/")
                    result[rel] = structs
    return result

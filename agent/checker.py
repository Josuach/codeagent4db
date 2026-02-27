"""
Compile checker for generated diffs.

Applies a unified diff to a temporary copy of the project, then compiles it
to verify correctness. If compilation fails, reports the errors so they can
be fed back to the LLM for a fix attempt.
"""

import os
import shutil
import subprocess
import tempfile


def _find_gcc() -> str:
    """Find a usable gcc/cc executable."""
    for candidate in ["gcc", "cc", "/c/msys64/ucrt64/bin/gcc"]:
        try:
            subprocess.run(
                [candidate, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return ""


def _default_compile_cmd(project_root: str) -> str:
    """
    Try to find a sensible compile command.
    Prefers 'make' if a Makefile exists, otherwise falls back to
    compiling all .c files with gcc.
    """
    if os.path.exists(os.path.join(project_root, "Makefile")):
        return "make"
    gcc = _find_gcc()
    if gcc:
        c_files = [f for f in os.listdir(project_root) if f.endswith(".c")]
        if c_files:
            return f"{gcc} -c " + " ".join(c_files)
    return ""


def apply_and_compile(
    diff_text: str,
    project_root: str,
    compile_cmd: str = "",
) -> tuple[bool, str]:
    """
    Apply a unified diff to a temporary copy of project_root and compile it.

    Args:
        diff_text:    The unified diff string to apply.
        project_root: Absolute path to the C project.
        compile_cmd:  Shell command to run for compilation. If empty, auto-detected.

    Returns:
        (success, output) where:
          - success: True if diff applied and compiled without errors
          - output:  Combined stdout+stderr from patch and compile steps
    """
    # Find patch executable
    patch_exe = shutil.which("patch") or "patch"

    # Write diff to a temp file (patch needs to read it by path on Windows)
    diff_fd, diff_path = tempfile.mkstemp(suffix=".diff")
    try:
        with os.fdopen(diff_fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(diff_text)
    except Exception:
        os.close(diff_fd)
        raise

    # Create a temp copy of the project
    tmpdir = tempfile.mkdtemp(prefix="codeagent_compile_")
    try:
        # Copy project files to tmpdir
        for fname in os.listdir(project_root):
            src = os.path.join(project_root, fname)
            if os.path.isfile(src):
                shutil.copy2(src, tmpdir)

        log_parts = []

        # Apply the diff
        patch_result = subprocess.run(
            [patch_exe, "--fuzz=5", "-p1", "-i", diff_path],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        log_parts.append("=== patch output ===")
        log_parts.append(patch_result.stdout)
        if patch_result.stderr:
            log_parts.append(patch_result.stderr)

        if patch_result.returncode != 0:
            return False, "\n".join(log_parts)

        # Resolve compile command
        if not compile_cmd:
            compile_cmd = _default_compile_cmd(tmpdir)
        if not compile_cmd:
            return False, "No compile command found. Specify --compile-cmd."

        # Run compile
        compile_result = subprocess.run(
            compile_cmd,
            cwd=tmpdir,
            shell=True,
            capture_output=True,
            text=True,
        )
        log_parts.append("\n=== compile output ===")
        log_parts.append(compile_result.stdout)
        if compile_result.stderr:
            log_parts.append(compile_result.stderr)

        success = compile_result.returncode == 0
        return success, "\n".join(log_parts)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            os.unlink(diff_path)
        except OSError:
            pass


def compile_check_and_fix(
    diff_text: str,
    plan: dict,
    project_root: str,
    compile_cmd: str,
    client,
    model: str = "",
    max_retries: int = 2,
) -> tuple[str, bool, str]:
    """
    Apply the diff, compile, and attempt to fix compilation errors automatically.

    Args:
        diff_text:    The generated unified diff.
        plan:         The planner output dict (for context in fix prompts).
        project_root: Absolute path to the C project.
        compile_cmd:  Shell compile command (empty = auto-detect).
        client:       LLMClient instance.
        model:        Model name (empty = use client default).
        max_retries:  How many fix attempts to make.

    Returns:
        (final_diff, success, log) where final_diff is the (possibly fixed) diff,
        success indicates whether the last compile attempt passed, and log
        contains all diagnostic output.
    """
    from agent.implementer import fix_diff  # avoid circular import at module level

    current_diff = diff_text
    full_log = []

    for attempt in range(max_retries + 1):
        label = "initial" if attempt == 0 else f"fix attempt {attempt}"
        print(f"[checker] {label}: applying diff and compiling ...")

        success, output = apply_and_compile(current_diff, project_root, compile_cmd)
        full_log.append(f"\n--- {label} ---\n{output}")

        if success:
            print(f"[checker] compilation succeeded on {label}")
            break

        # Extract only error lines to keep the fix prompt concise
        error_lines = [
            line for line in output.splitlines()
            if "error:" in line or "FAILED" in line.lower()
        ]
        error_summary = "\n".join(error_lines[:60])  # cap at 60 lines
        print(f"[checker] compilation failed ({len(error_lines)} errors). "
              f"{'Attempting fix...' if attempt < max_retries else 'Max retries reached.'}")

        if attempt >= max_retries:
            break

        # Ask the LLM to fix the diff
        print(f"[checker] calling LLM to fix errors ...")
        current_diff = fix_diff(
            diff=current_diff,
            compile_errors=error_summary,
            plan=plan,
            project_root=project_root,
            client=client,
            model=model,
        )

    return current_diff, success, "\n".join(full_log)

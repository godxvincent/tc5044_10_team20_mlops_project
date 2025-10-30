#!/usr/bin/env python3
"""
Run Black and Flake8, capture outputs, and save TXT reports into tools/lint_reports
with filenames like <YYYYMMDD_HHMMSS>_<user>_<branch>_black.txt / _flake8.txt.

- Includes current Git branch name in the filename.
- Supports --check and --verbose flags for Black.
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ---------- low-level runner ----------


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr) as text."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out.strip(), err.strip()


def run_cmd_and_save(tool: str, cmd: List[str], out_dir: Path, prefix: str) -> int:
    """Run a tool, write a TXT report, mirror basic info to console, return exit code."""
    code, out, err = run_cmd(cmd)
    report_path = out_dir / f"{prefix}_{tool}.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Command: " + " ".join(cmd) + "\n")
        f.write(f"# Exit code: {code}\n\n")
        if out:
            f.write(out)
        if err:
            f.write("\n# STDERR\n" + err)

    print(f"\n--- {tool.capitalize()} report saved to: {report_path}")
    if out:
        print(f"\n[{tool.upper()} STDOUT]\n{out}")
    if err:
        print(f"\n[{tool.upper()} STDERR]\n{err}")

    return code


# ---------- individual tools ----------


def build_black_cmd(check: bool, verbose: bool) -> List[str]:
    cmd = ["black", "."]
    if check:
        cmd = ["black", "--check", "."]
    if verbose:
        cmd.insert(1, "--verbose")
    return cmd


def run_black(check: bool, verbose: bool, out_dir: Path, prefix: str) -> int:
    cmd = build_black_cmd(check=check, verbose=verbose)
    print(f"\nRunning: {' '.join(cmd)}")
    return run_cmd_and_save("black", cmd, out_dir, prefix)


def run_flake8(out_dir: Path, prefix: str) -> int:
    cmd = ["flake8", "."]
    print(f"\nRunning: {' '.join(cmd)}")
    return run_cmd_and_save("flake8", cmd, out_dir, prefix)


# ---------- helper functions ----------


def get_git_branch() -> str:
    """Return current Git branch name, or 'no-branch' if not in a Git repo."""
    try:
        code, out, _ = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if code == 0 and out:
            return out.replace("/", "_")  # sanitize for filenames
    except Exception:
        pass
    return "no-branch"


def make_prefix(user_from_cli: str | None) -> str:
    user = user_from_cli or getpass.getuser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = get_git_branch()
    return f"{timestamp}_{user}_{branch}"


def ensure_output_dir(relative_dir_name: str) -> Path:
    """
    Always create the output folder next to this script:
    tools/<relative_dir_name>
    """
    base_dir = Path(__file__).parent  # tools/
    out_dir = base_dir / relative_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Black & Flake8 and save TXT reports with user + git branch.")
    parser.add_argument(
        "--out-dir",
        default="lint_reports",
        help="Folder (relative to this script) to store reports. Default: lint_reports",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User name for file naming. Default: current OS user",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run Black in --check mode (no changes; non-zero exit if reformat needed).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Add --verbose to Black to list files it touches.",
    )
    return parser.parse_args()


# ---------- main orchestration ----------


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.out_dir)
    prefix = make_prefix(args.user)

    black_code = run_black(check=args.check, verbose=args.verbose, out_dir=out_dir, prefix=prefix)
    flake8_code = run_flake8(out_dir=out_dir, prefix=prefix)

    if black_code != 0 or flake8_code != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

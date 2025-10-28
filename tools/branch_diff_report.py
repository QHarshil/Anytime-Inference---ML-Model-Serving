"""Utility to summarize the differences between two Git branches.

This helper is useful for quickly understanding what the current feature
branch changes relative to the baseline (typically ``main``).  It prints a
summary of file change counts, the high level directory impact, and optionally
emits the raw ``git diff --stat`` output so you can inspect the magnitude of the
change.
"""
from __future__ import annotations

import argparse
import collections
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple


StatusMap = Dict[str, List[Tuple[str, str]]]


def _run_git_command(args: Iterable[str]) -> str:
    """Run a git command and return its stdout as text.

    Parameters
    ----------
    args:
        The git sub-command arguments, e.g. ["diff", "--stat", "main..work"].
    """
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_name_status(output: str) -> StatusMap:
    """Parse ``git diff --name-status`` output.

    Returns a mapping from the primary status code to a list of tuples.
    For add/modify/delete entries the tuple contains ``(path, path)``.
    For renames it contains ``(old_path, new_path)``.
    """
    summary: StatusMap = collections.defaultdict(list)
    if not output:
        return summary

    for line in output.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        code = status[0]
        if code == "R" and len(parts) == 3:
            summary[code].append((parts[1], parts[2]))
        elif len(parts) >= 2:
            summary[code].append((parts[1], parts[1]))
    return summary


def _format_section(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(lines)
    if not body:
        return ""
    return f"{title}\n{'-' * len(title)}\n{body}\n"


def generate_report(base: str, head: str, show_stat: bool) -> str:
    """Generate a textual report summarising ``head`` relative to ``base``."""
    name_status_output = _run_git_command(["diff", "--name-status", f"{base}..{head}"])
    status_summary = _parse_name_status(name_status_output)

    additions = len(status_summary.get("A", []))
    modifications = len(status_summary.get("M", []))
    deletions = len(status_summary.get("D", []))
    renames = len(status_summary.get("R", []))

    header_lines = [
        f"Comparing '{head}' against '{base}'",
        f"Added files      : {additions}",
        f"Modified files   : {modifications}",
        f"Deleted files    : {deletions}",
        f"Renamed files    : {renames}",
    ]

    # Compute the first-level directory impact for a quick overview.
    dir_counter: Dict[str, int] = collections.Counter()
    for entries in status_summary.values():
        for _, new_path in entries:
            top_level = new_path.split("/", 1)[0]
            dir_counter[top_level] += 1

    directory_lines = [f"{directory}: {count}" for directory, count in sorted(dir_counter.items())]

    sections: List[str] = [
        _format_section("Summary", header_lines),
        _format_section("Impacted top-level directories", directory_lines),
    ]

    if show_stat:
        diff_stat = _run_git_command(["diff", "--stat", f"{base}..{head}"])
        sections.append(_format_section("git diff --stat", diff_stat.splitlines()))

    return "\n".join(section for section in sections if section).strip()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="main", help="Baseline branch or commit (default: main)")
    parser.add_argument("head", nargs="?", default="work", help="Comparison branch or commit (default: work)")
    parser.add_argument(
        "--stat",
        action="store_true",
        help="Include git diff --stat output for additional detail.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = generate_report(args.base, args.head, args.stat)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive fallback
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        return exc.returncode or 1

    if report:
        print(report)
    else:
        print(f"No differences found between {args.base!r} and {args.head!r}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

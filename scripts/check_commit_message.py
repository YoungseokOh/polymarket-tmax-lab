#!/usr/bin/env python3
"""Validate commit message headers against the repository convention."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

ALLOWED_TYPES: Final[tuple[str, ...]] = (
    "feat",
    "fix",
    "docs",
    "refactor",
    "test",
    "chore",
    "ci",
    "perf",
)
SKIP_PREFIXES: Final[tuple[str, ...]] = ("Merge ", "Revert ", "fixup! ", "squash! ")

_TYPE_PATTERN = "|".join(re.escape(item) for item in ALLOWED_TYPES)
HEADER_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<type>{_TYPE_PATTERN})(?P<breaking>!)?: "
    r"(?P<summary>[^\s].+)$"
)


def load_header(path: Path) -> str:
    """Return the first non-empty, non-comment line from a commit message file."""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return ""


def validate_header(header: str) -> tuple[bool, str]:
    """Validate the commit-message header and return a status plus error text."""

    if not header:
        return False, "Commit message is empty."
    if header.startswith(SKIP_PREFIXES):
        return True, ""
    match = HEADER_RE.fullmatch(header)
    if match is None:
        return False, "Commit header must follow 'type: subject'."
    summary = match.group("summary")
    if summary.endswith("."):
        return False, "Commit summary must not end with a period."
    return True, ""


def main(argv: list[str]) -> int:
    """CLI entrypoint for the commit-msg hook."""

    if len(argv) != 2:
        print("Usage: check_commit_message.py <commit-msg-file>", file=sys.stderr)
        return 2
    header = load_header(Path(argv[1]))
    is_valid, error = validate_header(header)
    if is_valid:
        return 0

    print(error, file=sys.stderr)
    print("", file=sys.stderr)
    print("Expected format: type: subject", file=sys.stderr)
    print(f"Allowed types: {', '.join(ALLOWED_TYPES)}", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    print("  feat: add hourly max calibration features", file=sys.stderr)
    print("  fix: parse HKO finalization clause", file=sys.stderr)
    print("  docs: add folder ownership guide", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

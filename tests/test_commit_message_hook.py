from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_commit_message.py"


def run_hook(message: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    commit_file = tmp_path / "COMMIT_EDITMSG"
    commit_file.write_text(message, encoding="utf-8")
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), str(commit_file)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_valid_commit_message_passes(tmp_path: Path) -> None:
    result = run_hook("feat: add hourly max calibration features\n", tmp_path)

    assert result.returncode == 0
    assert result.stderr == ""


def test_invalid_commit_message_fails_with_guidance(tmp_path: Path) -> None:
    result = run_hook("update stuff\n", tmp_path)

    assert result.returncode == 1
    assert "type: subject" in result.stderr
    assert "Allowed types" in result.stderr


def test_merge_commit_is_accepted(tmp_path: Path) -> None:
    result = run_hook("Merge branch 'main' into feature/docs\n", tmp_path)

    assert result.returncode == 0

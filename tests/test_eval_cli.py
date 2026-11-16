"""The command lines must start. Regression: a crash at startup when the module had no docstring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "src.eval.evaluate",
        "src.eval.relevance",
        "src.eval.check_manifests",
        "src.eval.sanity_check",
        "src.eval.clap_baseline",
        "src.eval.ttd_prep",
        "src.eval.onset_baseline",
        "src.eval.ttd_render",
        "src.eval.transcribe_baseline",
    ],
)
def test_help_runs(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout

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
        "src.eval.compare",
        "src.eval.sloppiness",
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


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "src.eval.transcribe_baseline", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_avp_data_gives_instructions_not_a_file_error(tmp_path):
    result = _run("train", "--avp-root", str(tmp_path / "nope"))
    assert result.returncode == 1
    assert "tools.download_datasets" in result.stderr
    (tmp_path / "avp").mkdir()
    result = _run(
        "tune", "--avp-root", str(tmp_path / "avp"), "--avp-index", str(tmp_path / "x.csv")
    )
    assert result.returncode == 1
    assert "index_avp" in result.stderr


def test_running_before_training_says_to_train_first(tmp_path):
    result = _run("run", "--classifier", str(tmp_path / "missing.npz"))
    assert result.returncode == 1
    assert "train command first" in result.stderr

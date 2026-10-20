"""Smoke test: the package layout imports and the core dependencies are installed."""

import importlib

import pytest

SUBPACKAGES = ["music", "library", "queries", "models", "baselines", "eval"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_src_subpackages_import(name):
    importlib.import_module(f"src.{name}")


@pytest.mark.parametrize("name", ["numpy", "pandas", "pretty_midi", "soundfile", "yaml"])
def test_core_dependencies_import(name):
    importlib.import_module(name)

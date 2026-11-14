from __future__ import annotations

import numpy as np
import pytest

from src.eval.evaluate import EvalError
from src.eval.sanity_check import oracle_scores, run_sanity


def _fake_key(n_patterns: int = 150, kits: int = 2):
    ## each pattern is rendered on every kit, so each query has `kits` relevant clips
    library_ids = [f"p{p}_k{k}" for p in range(n_patterns) for k in range(kits)]
    relevance = {f"q{p}": [f"p{p}_k{k}" for k in range(kits)] for p in range(n_patterns)}
    return library_ids, relevance


def test_oracle_scores_put_relevant_clips_first_without_ties():
    library_ids, relevance = _fake_key(n_patterns=3)
    query_ids = sorted(relevance)
    scores = oracle_scores(query_ids, library_ids, relevance)
    row = scores[query_ids.index("q1")]
    top_two = np.argsort(-row)[:2]
    assert sorted(library_ids[i] for i in top_two) == ["p1_k0", "p1_k1"]
    assert len(set(row[top_two])) == 2


def test_sanity_passes_on_a_consistent_key(tmp_path):
    library_ids, relevance = _fake_key()
    result = run_sanity(library_ids, relevance, tmp_path)
    assert result["oracle_mrr"] == 1.0
    assert result["passed"]
    assert abs(result["random_mrr"] - result["chance_mrr"]) < 0.05
    assert (tmp_path / "oracle.npz").exists()
    assert (tmp_path / "random.npz").exists()


def test_chance_value_matches_the_pool_size(tmp_path):
    ## 2 relevant clips of 300 is a much lower floor than 2 of 10
    big = run_sanity(*_fake_key(n_patterns=150), tmp_path / "big")
    small = run_sanity(*_fake_key(n_patterns=5), tmp_path / "small")
    assert big["chance_mrr"] < small["chance_mrr"]


def test_answer_key_naming_a_clip_outside_the_library_is_an_error(tmp_path):
    library_ids, relevance = _fake_key(n_patterns=5)
    relevance["q0"] = ["not_in_library"]
    with pytest.raises(KeyError):
        run_sanity(library_ids, relevance, tmp_path)


def test_empty_key_is_an_error(tmp_path):
    library_ids, _ = _fake_key(n_patterns=2)
    with pytest.raises((EvalError, ValueError, IndexError)):
        run_sanity(library_ids, {}, tmp_path)

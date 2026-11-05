## default test suite for src.eval.evaluate

from __future__ import annotations

import numpy as np
import pytest

from src.eval.evaluate import (
    EvalError,
    bootstrap_ci,
    chance_rr,
    evaluate,
    load_system,
    query_metrics,
    rank_relevant,
    summarize,
)


def _save_scores(path, scores, query_ids=None, library_ids=None):
    scores = np.asarray(scores, dtype=float)
    np.savez(
        path,
        query_ids=np.array(query_ids or [f"q{i}" for i in range(scores.shape[0])]),
        library_ids=np.array(library_ids or [f"c{i}" for i in range(scores.shape[1])]),
        scores=scores,
    )


## Rank and metric calculations


def test_ranks_of_relevant_clips():
    row = np.array([0.9, 0.8, 0.7, 0.6])
    assert rank_relevant(row, np.array([1, 3])).tolist() == [2, 4]


def test_ties_count_against_the_system():
    ## A constant-score system must not benefit from where the relevant clip sits.
    row = np.zeros(4)
    for position in range(4):
        assert rank_relevant(row, np.array([position])).tolist() == [4]


def test_metrics_hand_example():
    ## Relevant clips at ranks 2 and 4: RR = 1/2, AP = (1/2 + 2/4) / 2 = 1/2.
    m = query_metrics(np.array([2, 4]))
    assert m["first_rank"] == 2
    assert m["rr"] == 0.5
    assert m["ap"] == pytest.approx(0.5)
    assert (m["hit@1"], m["hit@5"], m["hit@10"]) == (0.0, 1.0, 1.0)


def test_chance_rr_matches_enumeration():
    ## One relevant clip of 3: ranks 1, 2, 3 equally likely -> (1 + 1/2 + 1/3) / 3.
    assert chance_rr(3, 1) == pytest.approx(11 / 18)
    ## Two relevant of 4: P(first = 1, 2, 3) = 1/2, 1/3, 1/6 -> 1/2 + 1/6 + 1/18.
    assert chance_rr(4, 2) == pytest.approx(13 / 18)
    assert chance_rr(5, 5) == pytest.approx(1.0)


def test_chance_rr_rejects_impossible_counts():
    with pytest.raises(ValueError):
        chance_rr(3, 4)
    with pytest.raises(ValueError):
        chance_rr(3, 0)


## Whole file evaluations


def test_perfect_system_scores_one(tmp_path):
    scores = np.eye(3)
    _save_scores(tmp_path / "s.npz", scores)
    relevance = {"q0": ["c0"], "q1": ["c1"], "q2": ["c2"]}
    table, counts = evaluate(tmp_path / "s.npz", relevance)
    assert table["rr"].tolist() == [1.0, 1.0, 1.0]
    assert counts["n_queries_dropped"] == 0


def test_embeddings_and_scores_agree(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4, 8))
    c = rng.normal(size=(6, 8))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    ids = {"query_ids": np.array(list("abcd")), "library_ids": np.array(list("uvwxyz"))}
    np.savez(tmp_path / "emb.npz", query_emb=q, library_emb=c, **ids)
    np.savez(tmp_path / "sc.npz", scores=q @ c.T, **ids)
    assert np.allclose(load_system(tmp_path / "emb.npz")[2], load_system(tmp_path / "sc.npz")[2])


def test_unnormalized_embeddings_are_rejected(tmp_path):
    np.savez(
        tmp_path / "bad.npz",
        query_ids=np.array(["q"]),
        library_ids=np.array(["c"]),
        query_emb=np.array([[2.0, 0.0]]),
        library_emb=np.array([[1.0, 0.0]]),
    )
    with pytest.raises(EvalError, match="L2-normalized"):
        load_system(tmp_path / "bad.npz")


def test_wrong_shape_and_nan_are_rejected(tmp_path):
    np.savez(
        tmp_path / "shape.npz",
        query_ids=np.array(["q"]),
        library_ids=np.array(["a", "b"]),
        scores=np.zeros((1, 3)),
    )
    with pytest.raises(EvalError, match="shape"):
        load_system(tmp_path / "shape.npz")
    _save_scores(tmp_path / "nan.npz", [[0.0, np.nan]])
    with pytest.raises(EvalError, match="NaN"):
        load_system(tmp_path / "nan.npz")


def test_queries_without_relevant_clips_are_dropped_and_counted(tmp_path):
    _save_scores(tmp_path / "s.npz", np.eye(3))
    table, counts = evaluate(tmp_path / "s.npz", {"q0": ["c0"], "q1": [], "ghost": ["c1"]})
    assert table["query_id"].tolist() == ["q0"]
    assert counts["n_queries_dropped"] == 2
    assert counts["n_answer_key_queries_not_in_system"] == 1


def test_answer_key_clip_missing_from_library_is_an_error(tmp_path):
    _save_scores(tmp_path / "s.npz", np.eye(2))
    with pytest.raises(EvalError, match="not in the library"):
        evaluate(tmp_path / "s.npz", {"q0": ["nope"]})


def test_random_system_lands_near_chance(tmp_path):
    rng = np.random.default_rng(1)
    n_queries, n_library = 400, 50
    _save_scores(tmp_path / "s.npz", rng.random((n_queries, n_library)))
    relevance = {f"q{i}": [f"c{i % n_library}", f"c{(i + 7) % n_library}"] for i in range(n_queries)}
    table, _ = evaluate(tmp_path / "s.npz", relevance)
    assert table["rr"].mean() == pytest.approx(chance_rr(n_library, 2), abs=0.04)


## Statistics


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    values = np.random.default_rng(2).random(300)
    low, high = bootstrap_ci(values, n_boot=2000, seed=5)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_ci(values, n_boot=2000, seed=5)


def test_participant_bootstrap_is_wider_when_queries_cluster():
    # 10 participants, each with 20 queries that all score the same: queries are far from
    # independent, so resampling participants must give a wider interval than queries.
    values = np.repeat(np.linspace(0.1, 0.9, 10), 20)
    groups = np.repeat(np.arange(10), 20)
    plain = bootstrap_ci(values, n_boot=2000, seed=0)
    clustered = bootstrap_ci(values, groups, n_boot=2000, seed=0)
    assert (clustered[1] - clustered[0]) > 2 * (plain[1] - plain[0])


def test_summary_reports_every_metric(tmp_path):
    _save_scores(tmp_path / "s.npz", np.eye(3))
    table, _ = evaluate(tmp_path / "s.npz", {"q0": ["c0"], "q1": ["c1"], "q2": ["c2"]})
    summary = summarize(table, n_boot=200)
    assert {"mrr", "map", "hit@1", "hit@5", "hit@10", "chance_mrr"} <= set(summary)
    assert summary["mrr"]["mean"] == 1.0

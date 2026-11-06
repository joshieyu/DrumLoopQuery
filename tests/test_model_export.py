"""The embedding file writer (src/models/export.py) against D's scorer (src/eval/evaluate.py)."""

import numpy as np
import pytest

from src.eval.evaluate import evaluate, load_system
from src.models.export import embedding_path, write_embeddings


def unit_rows(n, d=8, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_file_round_trips_through_ds_loader(tmp_path):
    q, lib = unit_rows(3), unit_rows(5, seed=1)
    path = write_embeddings(tmp_path / "s_t.npz", ["q0", "q1", "q2"], q, list("abcde"), lib)
    query_ids, library_ids, scores = load_system(path)
    assert list(query_ids) == ["q0", "q1", "q2"]
    assert list(library_ids) == list("abcde")
    np.testing.assert_allclose(scores, q @ lib.T, atol=1e-6)


def test_perfect_embeddings_score_mrr_one(tmp_path):
    lib = np.eye(4, dtype=np.float32)
    path = write_embeddings(tmp_path / "s_t.npz", ["q0", "q1"], lib[[2, 0]], list("abcd"), lib)
    table, counts = evaluate(path, {"q0": ["c"], "q1": ["a"]})
    assert table["rr"].tolist() == [1.0, 1.0]
    assert counts["n_queries_scored"] == 2


def test_embedding_path_follows_the_contract(tmp_path):
    assert embedding_path("r1_supcon", "syn_beatbox", tmp_path) == (
        tmp_path / "r1_supcon_syn_beatbox.npz"
    )
    with pytest.raises(ValueError, match="without slashes"):
        embedding_path("r1/supcon", "syn", tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        embedding_path("", "syn", tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(query_emb=2 * unit_rows(2)), "length 1"),
        (dict(query_emb=np.full((2, 8), np.nan)), "NaN"),
        (dict(query_emb=unit_rows(3)), "shape"),
        (dict(library_emb=unit_rows(3, d=4)), "dimensions"),
        (dict(query_ids=["q", "q"]), "duplicates"),
        (dict(library_ids=["a", "", "c"]), "empty id"),
        (dict(query_ids=[], query_emb=np.zeros((0, 8))), "empty"),
    ],
)
def test_bad_files_are_refused(tmp_path, change, message):
    args = dict(
        query_ids=["q0", "q1"],
        query_emb=unit_rows(2),
        library_ids=["a", "b", "c"],
        library_emb=unit_rows(3, seed=1),
    )
    args.update(change)
    with pytest.raises(ValueError, match=message):
        write_embeddings(tmp_path / "s_t.npz", **args)
    assert not (tmp_path / "s_t.npz").exists()


def test_file_name_must_end_in_npz(tmp_path):
    with pytest.raises(ValueError, match=".npz"):
        write_embeddings(tmp_path / "s_t.npy", ["q"], unit_rows(1), ["a"], unit_rows(1))

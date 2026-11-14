from __future__ import annotations

import pandas as pd
import pytest

from src.eval.check_manifests import check_manifests


@pytest.fixture
def library():
    return pd.DataFrame(
        [
            ("sa_k1", "A", "T1", "test", "salamander", "perfA:bar0"),
            ("sa_k2", "A", "T1", "test", "muldjordkit", "perfA:bar0"),
            ("sb_k1", "B", "T2", "test", "salamander", "perfB:bar0"),
            ("sb_k2", "B", "T2", "test", "muldjordkit", "perfB:bar0"),
            ("tc_k1", "C", "T3", "train", "drskit", "perfC:bar0"),
            ("td_k1", "D", "T4", "validation", "drskit", "perfD:bar0"),
        ],
        columns=[
            "clip_id",
            "pattern_id",
            "dualized_pattern_id",
            "split",
            "kit_id",
            "source_clip_id",
        ],
    )


@pytest.fixture
def queries():
    return pd.DataFrame(
        [
            ("q1", "A", "T1", "beatbox", "synthetic", "test", "perfA:bar0", "P10"),
            ("q2", "B", "T2", "beatbox", "synthetic", "test", "perfB:bar0", "P10"),
            ("q3", "C", "T3", "beatbox", "synthetic", "train", "perfC:bar0", "P1"),
        ],
        columns=[
            "query_id",
            "pattern_id",
            "dualized_pattern_id",
            "query_type",
            "source",
            "split",
            "source_clip_id",
            "participant_id",
        ],
    )


@pytest.fixture
def participants():
    return pd.DataFrame([("P1", "train"), ("P10", "test")], columns=["participant_id", "split"])


def _fails(findings):
    return [f.text for f in findings if f.level == "FAIL"]


def _warns(findings):
    return [f.text for f in findings if f.level == "WARN"]


def test_consistent_manifests_have_no_failures(library, queries, participants):
    findings = check_manifests(library, queries, participants)
    assert _fails(findings) == []
    assert _warns(findings) == []


def test_duplicate_ids_fail(library, queries, participants):
    findings = check_manifests(pd.concat([library, library.iloc[:1]]), queries, participants)
    assert any("duplicate clip_id" in t for t in _fails(findings))
    findings = check_manifests(library, pd.concat([queries, queries.iloc[:1]]), participants)
    assert any("duplicate query_id" in t for t in _fails(findings))


def test_test_kit_used_in_training_fails(library, queries, participants):
    library.loc[library.clip_id == "tc_k1", "kit_id"] = "salamander"
    findings = check_manifests(library, queries, participants)
    assert any("kits used in both test and train" in t for t in _fails(findings))


def test_shared_pattern_is_a_warning_not_a_failure(library, queries, participants):
    ## the train clip and its train query both take pattern A, so only the leakage warning fires
    library.loc[library.clip_id == "tc_k1", "pattern_id"] = "A"
    queries.loc[queries.query_id == "q3", "pattern_id"] = "A"
    findings = check_manifests(library, queries, participants)
    assert _fails(findings) == []
    assert any("also appear in other splits" in t for t in _warns(findings))


def test_query_pointing_at_missing_source_clip_fails(library, queries, participants):
    queries.loc[0, "source_clip_id"] = "perfZ:bar9"
    findings = check_manifests(library, queries, participants)
    assert any("not in the library" in t for t in _fails(findings))


def test_query_with_wrong_pattern_or_split_fails(library, queries, participants):
    queries.loc[0, "pattern_id"] = "B"
    findings = check_manifests(library, queries, participants)
    assert any("different pattern_id" in t for t in _fails(findings))
    queries.loc[0, "pattern_id"] = "A"
    queries.loc[0, "split"] = "train"
    findings = check_manifests(library, queries, participants)
    assert any("different split from their source clip" in t for t in _fails(findings))


def test_source_clip_with_two_patterns_fails(library, queries, participants):
    library.loc[library.clip_id == "sa_k2", "pattern_id"] = "X"
    findings = check_manifests(library, queries, participants)
    assert any("more than one pattern_id" in t for t in _fails(findings))


def test_participant_in_the_wrong_split_fails(library, queries, participants):
    queries.loc[0, "participant_id"] = "P1"
    findings = check_manifests(library, queries, participants)
    assert any("different split from their participant" in t for t in _fails(findings))
    queries.loc[0, "participant_id"] = "P99"
    findings = check_manifests(library, queries, participants)
    assert any("not in the split file" in t for t in _fails(findings))


def test_synthetic_query_without_pattern_fails(library, queries, participants):
    queries.loc[0, "pattern_id"] = None
    findings = check_manifests(library, queries, participants)
    assert any("no pattern_id" in t for t in _fails(findings))


def test_queries_without_a_source_clip_are_skipped_not_failed(library, queries, participants):
    real = pd.DataFrame(
        [("r1", None, "T1", "beatbox", "team", "test", None, "member1")],
        columns=queries.columns,
    )
    findings = check_manifests(library, pd.concat([queries, real]), participants)
    assert _fails(findings) == []
    assert any("no source_clip_id" in f.text for f in findings)


def test_missing_pool_split_stops_early(library, queries, participants):
    findings = check_manifests(library, queries, participants, pool_split="nope")
    assert any("no clips with split" in t for t in _fails(findings))

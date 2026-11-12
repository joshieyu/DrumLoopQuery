"""Production manifest boundaries and resumable output integrity."""

import json

import pandas as pd
import pytest

from src.build_dataset import Receipts, validate_manifests
from src.music.spec import load_spec


def tables():
    library = pd.DataFrame(
        [
            {"clip_id": "l1", "pattern_id": "p1", "split": "train", "kit_id": "avl_black_pearl"},
            {"clip_id": "l2", "pattern_id": "p2", "split": "test", "kit_id": "salamander"},
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": "q1", "pattern_id": "p1", "split": "train", "participant_id": "P1"},
            {"query_id": "q2", "pattern_id": "p2", "split": "test", "participant_id": "P2"},
        ]
    )
    assignments = pd.DataFrame({"participant_id": ["P1", "P2"], "split": ["train", "test"]})
    return library, queries, assignments


def test_valid_manifest_and_multiple_relevant_kits():
    library, queries, assignments = tables()
    extra = library.iloc[[0]].assign(clip_id="l3", kit_id="drskit")
    validate_manifests(pd.concat([library, extra]), queries, assignments, load_spec())


@pytest.mark.parametrize("mistake", ["pattern", "kit", "participant", "target", "duplicate"])
def test_reject_invalid_handoff(mistake):
    library, queries, assignments = tables()
    if mistake == "pattern":
        library.loc[1, "pattern_id"] = "p1"
    elif mistake == "kit":
        library.loc[0, "kit_id"] = "salamander"
    elif mistake == "participant":
        queries.loc[0, "participant_id"] = "P2"
    elif mistake == "target":
        queries.loc[0, "pattern_id"] = "unmatched"
    else:
        queries.loc[1, "query_id"] = "q1"
    with pytest.raises(ValueError):
        validate_manifests(library, queries, assignments, load_spec())


def test_resume_rechecks_audio_hash_and_missing_file(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"rendered audio")
    receipts = Receipts(tmp_path, "query")
    row = {"audio_path": str(audio), "query_id": "q1"}
    receipts.add("q1", row, events=[{"played": True}])
    resumed = Receipts(tmp_path, "query")
    assert resumed.find("q1")["row"] == row
    assert json.loads(receipts.path.read_text())["events"] == [{"played": True}]
    audio.write_bytes(b"damaged")
    assert resumed.find("q1") is None
    audio.unlink()
    assert resumed.find("q1") is None
    assert resumed.find("unknown") is None


def test_resume_recovers_only_a_truncated_final_receipt(tmp_path):
    path = tmp_path / "query_receipts.jsonl"
    path.write_text('{"id":"q1","row":{}}\n{"id":"q2"')
    assert set(Receipts(tmp_path, "query").rows) == {"q1"}
    assert path.read_text() == '{"id":"q1","row":{}}\n'
    path.write_text('broken\n{"id":"q2","row":{}}\n')
    with pytest.raises(json.JSONDecodeError):
        Receipts(tmp_path, "query")

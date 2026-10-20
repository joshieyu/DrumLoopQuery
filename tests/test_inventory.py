"""Checks for the inventory math on small synthetic performances."""

import numpy as np
import pandas as pd
import pytest

from tools.download_datasets import Dataset, file_digest, verify
from tools.inventory import gmd


def make_hits(beats, pitch=42, perf_id="p1", bpm=120.0):
    beats = np.asarray(beats, dtype=float)
    return pd.DataFrame(
        {
            "id": perf_id,
            "pitch": pitch,
            "velocity": 80,
            "beat": beats,
            "sec": beats * 60.0 / bpm,
            "bpm": bpm,
            "instrument": gmd.GMD_NOTES[pitch][1],
            "label": gmd.note_label(pitch),
        }
    )


def test_swing_estimate_straight_vs_triplet():
    n = 32
    straight = make_hits(np.concatenate([np.arange(n), np.arange(n) + 0.5]), perf_id="straight")
    swung = make_hits(np.concatenate([np.arange(n), np.arange(n) + 2 / 3]), perf_id="swung")
    sw = gmd.swing_estimates(pd.concat([straight, swung], ignore_index=True))
    assert sw.loc["straight", "pos8"] == pytest.approx(0.5)
    assert sw.loc["swung", "pos8"] == pytest.approx(2 / 3)
    assert sw.loc["swung", "ratio8"] == pytest.approx(2.0)


def test_swing_estimate_ignores_kick_and_short_performances():
    kick = make_hits(np.arange(40) + 2 / 3, pitch=36, perf_id="kick_only")
    short = make_hits([0.5, 1.5], perf_id="short")
    sw = gmd.swing_estimates(pd.concat([kick, short], ignore_index=True), min_hits=16)
    assert "kick_only" not in sw.index
    assert np.isnan(sw.loc["short", "pos8"])


def test_bar_windows_assigns_early_downbeat_to_next_bar():
    # One hit per bar for 4 bars; the bar-3 downbeat is played a hair early (7.99 beats).
    hits = make_hits([0.0, 4.0, 7.99, 12.0], pitch=36)
    hits = hits.assign(split="train", beat_type="beat", style_primary="rock", time_signature="4-4")
    perf = pd.DataFrame(
        {
            "id": ["p1"],
            "beat_type": ["beat"],
            "time_signature": ["4-4"],
            "split": ["train"],
            "style_primary": ["rock"],
        }
    )
    w2 = gmd.bar_windows(hits, perf, bars_per_window=2, grid=12)
    assert w2["window"].tolist() == [0, 1]
    assert w2["n_36"].tolist() == [2, 2]


def test_grid_collisions_counts_merged_hits():
    hits = make_hits([0.0, 0.01, 1.0], pitch=38)  # first two merge on any coarse grid
    coll = gmd.grid_collisions(hits, [4, 480])
    assert coll.loc["Snare", "4 steps/beat"] == pytest.approx(100 / 3)
    assert coll.loc["Snare", "480 steps/beat"] == 0


def test_verify_rejects_wrong_checksum(tmp_path):
    f = tmp_path / "a.zip"
    f.write_bytes(b"hello")
    good = Dataset("x", "x", "1", "u", "a.zip", 5, "l", "p", md5=file_digest(f, "md5"))
    assert verify(f, good) == file_digest(f, "sha256")
    bad = Dataset("x", "x", "1", "u", "a.zip", 5, "l", "p", sha256="0" * 64)
    with pytest.raises(ValueError):
        verify(f, bad)

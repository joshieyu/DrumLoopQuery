"""Practice data rules (src/models/practice.py)."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")  # part of the optional "model" extra

from src.models.data import load_clips  # noqa: E402
from src.models.practice import (  # noqa: E402
    LEAD_IN_S,
    Loop,
    QuerySloppiness,
    add_noise,
    build_practice,
    draw_sloppiness,
    loop_hits,
    render_tempo,
    select_loops,
    timed_hits,
)
from src.music.rhythm import Note  # noqa: E402
from src.music.spec import load_spec  # noqa: E402
from tools.render.kits import KITS_DIR  # noqa: E402

SPEC = load_spec()
KICK, SNARE, HAT, TOM = 36, 38, 42, 48  # GMD notes (spec drum_map)


def make_loop(clip_id="d/s/1:bar0", pattern="P", split="train", notes=(), bpm=100.0):
    return Loop(clip_id, pattern, "D" + pattern, split, bpm, 8.0, tuple(notes), "m.mid", "d/s/1", 0)


def test_select_loops_skips_patterns_in_two_splits_and_is_reproducible():
    loops = [make_loop(f"t{i}", f"P{i}", "train") for i in range(5)]
    loops += [make_loop("v0", "P0", "validation"), make_loop("v1", "V1", "validation")]
    picked = select_loops(loops, {"train": 4, "validation": 1}, seed=0)
    assert all(lp.pattern_id != "P0" for lp in picked)  # P0 is in train and validation
    assert [lp.split for lp in picked].count("train") == 4
    assert picked == select_loops(loops, {"train": 4, "validation": 1}, seed=0)
    with pytest.raises(ValueError, match="only 4 are usable"):
        select_loops(loops, {"train": 5}, seed=0)


def test_loop_hits_follow_the_nearest_step_boundary():
    # Half a grid step is 1/24 beat (0.0417). An early downbeat stays; the next loop's doesn't.
    notes = [Note(KICK, 100, b) for b in (-0.05, -0.04, 0.0, 7.95, 7.97, 8.5)]
    beats = [b for b, _, _ in loop_hits(make_loop(notes=notes), SPEC)]
    assert beats == [-0.04, 0.0, 7.95]


@pytest.mark.parametrize("bpm", [72.0, 100.0, 158.0])
def test_render_tempo_stays_inside_the_spec(bpm):
    lo, hi = SPEC.rendering.tempo_bpm
    pct = SPEC.rendering.max_tempo_change_pct / 100
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = render_tempo(bpm, SPEC, rng)
        assert lo <= t <= hi and bpm * (1 - pct) <= t <= bpm * (1 + pct)


def test_sloppiness_draws_stay_inside_the_spec_ranges():
    rng = np.random.default_rng(0)
    s = SPEC.sloppiness
    for _ in range(100):
        d = draw_sloppiness(SPEC, rng)
        assert s.offset_ms[0] <= d.offset_ms <= s.offset_ms[1]
        assert s.jitter_ms[0] <= d.jitter_ms <= s.jitter_ms[1]
        assert s.snr_db[0] <= d.snr_db <= s.snr_db[1]
        for voice, (lo, hi) in s.drop_prob.items():
            assert lo <= d.drop_prob[voice] <= hi


def test_timed_hits_timing_offsets_and_drops():
    hits = [(0.0, KICK, 100), (1.0, HAT, 80), (2.0, TOM, 90), (3.0, SNARE, 90)]
    rng = np.random.default_rng(0)
    clean = timed_hits(hits, 120.0, SPEC, rng)
    assert [round(t, 6) for t, _, _ in clean] == [LEAD_IN_S + b * 0.5 for b in (0, 1, 2, 3)]

    shifted = timed_hits(
        hits, 120.0, SPEC, rng, QuerySloppiness(10.0, 0.0, {"kick": 0, "snare": 0, "hats": 1}, 20)
    )
    notes = [n for _, n, _ in shifted]
    assert HAT not in notes and TOM in notes  # hats dropped; toms aren't part of the rhythm
    assert shifted[0][0] == pytest.approx(LEAD_IN_S + 0.010)


def test_add_noise_hits_the_requested_snr():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 2000, 44100)).astype(np.float32)
    noisy = add_noise(clean, 20.0, rng)
    noise = noisy - clean
    snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    assert snr == pytest.approx(20.0, abs=0.3)


KITS = ("avl_black_pearl", "avl_blonde_bop")


@pytest.mark.skipif(
    not all((KITS_DIR / k).exists() for k in KITS),
    reason="needs the two AVL kits (python -m tools.download_datasets avl_black_pearl ...)",
)
def test_build_practice_writes_audio_manifests_and_answer_key(tmp_path):
    groove = [Note(KICK, 100, 0.0), Note(HAT, 80, 0.5), Note(SNARE, 100, 1.0)]
    loops = [
        make_loop("d/s/1:bar0", "A", "train", groove),
        make_loop("d/s/1:bar2", "B", "test", groove[:2] + [Note(SNARE, 90, 3.0)]),
    ]
    summary = build_practice(loops, SPEC, 22050, tmp_path, views_per_query=2, log=lambda *_: None)
    assert summary["library_clips"] == 2 and summary["queries"] == 4

    m = tmp_path / "manifests"
    clips = load_clips(m / "library_manifest.csv", m / "query_manifest.csv")
    assert len(clips) == 6 and all(p.endswith(".flac") for p in clips.audio_path)
    relevance = json.loads((m / "relevance.json").read_text())
    assert relevance["practice:d/s/1:bar2:q1"] == ["d/s/1:bar2@avl_black_pearl"]
    assert (tmp_path / "listen.html").read_text().count("<audio") == 3  # 1 test loop + 2 queries

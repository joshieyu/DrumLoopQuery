"""Tests for the recording beat picker (tools/listen/beat_picker.py).

Records are synthetic catalog rows, so nothing here needs GMD or the kits.
"""

import json

import pytest

from src.music.spec import load_spec
from tools.listen import beat_picker as bp
from tools.render.kits import kit_by_id

SPEC = load_spec()
SPB = SPEC.same_rhythm.steps_per_beat
BEATS = 8


def record(
    hits: dict[str, list[int]], *, bpm=120.0, clip_id="d1/s1/1:bar0", pid="p1", split="test"
):
    steps = BEATS * SPB
    onsets = [[s in hits.get(v, []) for s in range(steps)] for v in SPEC.voices]
    return {
        "clip_id": clip_id,
        "pattern_id": pid,
        "split": split,
        "tempo_bpm": bpm,
        "midi_path": "drummer1/session1/1_funk-groove1_120_beat_4-4.mid",
        "source_performance": "drummer1/session1/1",
        "start_bar": 0,
        "start_beat": 0.0,
        "end_beat": float(BEATS),
        "grid": {"voices": list(SPEC.voices), "steps_per_beat": SPB, "onsets": onsets},
    }


def straight_8ths():
    return {
        "kick": [0, 4 * SPB],
        "snare": [SPB, 3 * SPB, 5 * SPB, 7 * SPB],
        "hats": list(range(0, BEATS * SPB, SPB // 2)),
    }


def test_style_of():
    assert bp.style_of("drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid") == "funk-groove1"
    assert bp.style_of("odd.mid") == ""


def test_straight_eighths():
    s = bp.loop_stats(record(straight_8ths()), SPEC)
    assert s["sounds"] == 2 * BEATS  # kick and snare share steps with the hats
    assert (s["straight_off"], s["triplet_off"], s["other_off"]) == (BEATS, 0, 0)
    assert s["sounds_per_s"] == pytest.approx(2 * BEATS / (BEATS * 60 / 120))
    assert s["min_gap_ms"] == 250  # an 8th note at 120 BPM
    assert s["bar_beats"] == BEATS / SPEC.loops.bars
    assert s["hits"]["kick"] == [0, 4 * SPB]


def test_swung_and_loose_offbeats():
    swung = [b * SPB + k for b in range(BEATS) for k in (0, 2 * SPB // 3)]
    s = bp.loop_stats(record({"kick": [0], "snare": [SPB], "hats": swung + [2]}), SPEC)
    assert (s["straight_off"], s["triplet_off"], s["other_off"]) == (0, BEATS, 1)
    assert s["min_gap_ms"] == round(2 * 60_000 / 120 / SPB)  # steps 0 and 2


def test_single_sound_has_no_gap():
    assert bp.loop_stats(record({"kick": [0]}), SPEC)["min_gap_ms"] is None


def test_grid_without_16th_and_triplet_steps_is_rejected():
    r = record(straight_8ths())
    r["grid"] = {**r["grid"], "steps_per_beat": 8}
    with pytest.raises(ValueError, match="16th and triplet"):
        bp.loop_stats(r, SPEC)


def test_load_catalog_keeps_test_split(tmp_path):
    rows = [
        record({"kick": [0]}, clip_id=f"c{i}", split=s) for i, s in enumerate(["test", "train"])
    ]
    (tmp_path / "summary.json").write_text(json.dumps({"spec_version": SPEC.version}))
    (tmp_path / "loops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r["clip_id"] for r in bp.load_catalog(tmp_path, SPEC)] == ["c0"]


def test_build_and_page_without_audio(tmp_path):
    records = [
        record(straight_8ths(), clip_id="a:bar0", pid="same"),
        record(straight_8ths(), clip_id="a:bar2", pid="same"),
        record({"kick": [0], "snare": [SPB]}, clip_id="b:</script>", pid="other"),
    ]
    data = bp.build(records, SPEC, kit_by_id(SPEC, bp.DEFAULT_KIT), tmp_path, audio=False)
    assert data["meta"]["clips"] == 3 and data["meta"]["patterns"] == 2
    assert data["meta"]["spec_version"] == SPEC.version
    assert [r["same_pattern_clips"] for r in data["rows"]] == [2, 2, 1]
    assert [r["audio"] for r in data["rows"]] == [f"audio/{i:04d}.mp3" for i in range(3)]
    page = bp.write_page(data, tmp_path).read_text()
    assert "/*DATA*/" not in page
    assert page.count("</script>") == 1  # the data can't close the script early
    assert "b:<\\/script>" in page

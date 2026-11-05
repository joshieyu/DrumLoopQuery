"""Tests for the kit articulation audition page (tools/listen/kit_articulations.py).

The build test uses a tiny synthetic kit (8 kHz float WAVs in tmp_path), so it runs
without the downloaded kits.
"""

import json
import math

import numpy as np
import pytest
import soundfile as sf

from src.music.spec import Kit, KitNote, load_spec
from tools.listen import kit_articulations as ka
from tools.render import kits

SR = 8000
SPEC = load_spec()


def test_values_round_trip():
    for played in [KitNote(38, {}), KitNote(42, {4: 0, 7: 127}), KitNote(None, {})]:
        assert ka.parse_value(ka.note_value(played)) == played
    assert ka.note_value(KitNote(42, {7: 1, 4: 0})) == "42 cc4=0 cc7=1"
    assert ka.parse_value("  42   cc4=0 ") == KitNote(42, {4: 0})
    for bad in ["", "x", "128", "42 cc4", "42 cc4=200", "42 cc200=1", "silent 4"]:
        with pytest.raises(ValueError):
            ka.parse_value(bad)


def test_project_auditions_load():
    alternatives = ka.load_auditions(ka.AUDITIONS_PATH, SPEC)
    ids = {k.kit_id for k in SPEC.rendering.kits}
    assert set(alternatives) <= ids
    for notes in alternatives.values():
        for note, options in notes.items():
            assert note in SPEC.note_to_group
            assert options and all(o.key is not None for o in options)
    assert alternatives["avl_red_zeppelin"] == alternatives["avl_black_pearl"]  # YAML anchor
    assert KitNote(42, {4: 0}) in alternatives["salamander"][46]


@pytest.mark.parametrize(
    "text, message",
    [
        ("[1, 2]", "mapping of kit_id"),
        ("nope: {38: [38]}", "unknown kit"),
        ("drskit: [38]", "mapping of GMD note"),
        ("drskit: {99: [38]}", "not a drum_map note"),
        ("drskit: {38: 38}", "list of kit sounds"),
        ("drskit: {38: [x]}", "expected a key"),
        ("drskit: {38: [{key: 38}]}", "expected a key"),
        ("drskit: {38: [{key: 38, cc: {4: 200}}]}", "0-127"),
    ],
)
def test_bad_auditions_are_rejected(tmp_path, text, message):
    path = tmp_path / "auditions.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        ka.load_auditions(path, SPEC)


def tiny_kit(kit_id="tiny", program="tiny/kit.sfz"):
    """Kick 36, snare 38 (also its rimshot and cross-stick), hats 42, toms 40, no crashes."""
    note_map = {}
    for group in SPEC.drum_map.values():
        for note in group.notes:
            key = {"kick": 36, "snare": 38, "hats": 42}.get(group.group)
            if key is None:  # dropped: toms on 40, crashes silent
                key = 40 if note in SPEC.drum_map["toms"].notes else None
            note_map[note] = KitNote(key, {})
    return Kit(kit_id, "acoustic", program, None, None, {}, note_map)


def write_tiny_kit(kits_dir):
    folder = kits_dir / "tiny"
    folder.mkdir(parents=True)
    kick = np.zeros(100)
    kick[0] = 0.5
    for name, data in {"kick": kick, "snare": np.full(SR, 0.25), "hat": kick / 2}.items():
        sf.write(folder / f"{name}.wav", data.astype(np.float32), SR, subtype="FLOAT")
    (folder / "kit.sfz").write_text(
        "<global> amp_veltrack=0 pan=-100\n"
        "<region> key=36 sample=kick.wav\n"
        "<region> key=38 sample=snare.wav\n"
        "<region> key=40 hivel=63 sample=snare.wav\n"
        "<region> key=42 sample=hat.wav\n"
    )
    broken = kits_dir / "broken"
    broken.mkdir()
    (broken / "kit.sfz").write_text("<region> key=38 sample=../tiny/snare.wav cutoff=500\n")


def test_cell_options_put_the_draft_first_without_repeats():
    kit = tiny_kit()
    alternatives = {"tiny": {37: (KitNote(40, {}), KitNote(38, {}))}}
    assert ka.cell_options(kit, 37, alternatives) == [KitNote(38, {}), KitNote(40, {})]
    assert ka.cell_options(kit, 49, {"tiny": {49: (KitNote(40, {}),)}}) == [KitNote(40, {})]
    assert ka.cell_options(kit, 49, {}) == []  # a silent draft has nothing to play


def test_kit_gain():
    reference = np.zeros((SR, 2), dtype=np.float32)
    start = round(ka.PRE_ROLL_S * SR)
    reference[start:] = 0.1
    quiet = np.full((10, 2), 0.5, dtype=np.float32)
    match = 10 ** (ka.REF_RMS_DB / 20) / 0.1
    assert ka.kit_gain([reference, quiet], reference, SR) == pytest.approx(match)
    loud = np.full((10, 2), 1.0, dtype=np.float32)
    ceiling = 10 ** (kits.PEAK_CEILING_DB / 20)
    assert ka.kit_gain([reference, loud], reference, SR) == pytest.approx(ceiling)
    assert ka.kit_gain([quiet], None, SR) == pytest.approx(ceiling / 0.5)
    assert ka.kit_gain([np.zeros((0, 2))], None, SR) == 1.0


def test_build_and_page(tmp_path):
    kits_dir, out = tmp_path / "kits", tmp_path / "out"
    write_tiny_kit(kits_dir)
    stale = out / "audio" / "tiny" / "k99_v30.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"")
    kit_list = [tiny_kit(), tiny_kit("broken", "broken/kit.sfz"), tiny_kit("gone", "gone/kit.sfz")]
    alternatives = {"tiny": {37: (KitNote(40, {}),), 38: (KitNote(38, {}),)}}
    data = ka.build(SPEC, kit_list, alternatives, out, (30, 100), 0.5, kits_dir)

    tiny, broken, gone = data["kits"]
    assert tiny["renderable"] and tiny["sample_rate"] == SR
    assert not broken["renderable"] and "cutoff" in broken["reason"]
    assert not gone["renderable"] and gone["reason"] == "not downloaded"
    assert not stale.exists()
    assert [r["note"] for r in data["rows"]] == [n for g in SPEC.drum_map.values() for n in g.notes]

    cross_stick = data["cells"]["tiny|37"]
    assert cross_stick["draft"] == "38"
    assert [o["value"] for o in cross_stick["options"]] == ["38", "40"]
    rim_alt = cross_stick["options"][1]
    assert rim_alt["clips"][0] is not None and rim_alt["clips"][1] is None  # 40 stops at vel 63
    assert [o["value"] for o in data["cells"]["tiny|38"]["options"]] == ["38"]
    assert data["cells"]["tiny|38"]["same_as"] == [40, 37]
    crash = data["cells"]["tiny|49"]
    assert (crash["draft"], crash["options"], crash["same_as"]) == ("silent", [], [])
    assert data["cells"]["broken|38"]["options"][0]["clips"] == [None, None]

    browser = {o["value"]: o for o in data["browser"]["tiny"]}
    assert list(browser) == ["36", "38", "40", "42"]
    assert browser["38"]["used_by"] == [38, 40, 37]
    assert browser["38"]["label"] == "snare"

    snare, rate = sf.read(out / browser["38"]["clips"][1], always_2d=True)
    assert rate == SR and sf.info(out / browser["38"]["clips"][1]).subtype == "PCM_16"
    start = round(ka.PRE_ROLL_S * SR)
    assert len(snare) == round(0.5 * SR)  # cut at max_seconds
    assert int(np.flatnonzero(snare[:, 0])[0]) == start
    assert snare[-1, 0] == 0  # faded out
    window = snare[start : start + round(ka.REF_WINDOW_S * SR)]
    rms_db = 20 * math.log10(np.sqrt(np.mean(window**2)))
    assert rms_db == pytest.approx(ka.REF_RMS_DB, abs=0.1)  # the kit's level reference

    page = ka.write_page(data, out)
    html = page.read_text()
    assert "/*DATA*/" not in html
    embedded = html.split('<script type="application/json" id="data">')[1].split("</script>")[0]
    assert json.loads(embedded.replace("<\\/", "</")) == data

import csv
import zipfile

import pytest

from tools.listen import export_takes as ex

HEADER = [
    "pattern_id", "dualized_pattern_id", "split", "tempo_bpm", "query_id", "source_clip_id",
    "query_type", "source", "participant_id", "mode", "audio_path", "seed", "jitter_ms",
]  # fmt: skip


def take(member, clip, n, usable="yes", mode="click"):
    qid = f"team_{member}_{clip}_{mode}_{n}"
    return {
        "query_id": qid, "pattern_id": f"p{clip}", "dualized_pattern_id": f"d{clip}",
        "split": "test", "tempo_bpm": "96.0", "source_clip_id": f"perf:{clip}",
        "query_type": "beatbox", "source": "team", "participant_id": f"team_{member}",
        "mode": mode, "audio_path": f"data/recordings/team/{qid}.wav", "take": str(n),
        "latency_ms": "88.6", "usable": usable, "notes": "",
    }  # fmt: skip


TAKES = [
    take("A", "x", 1),
    take("A", "x", 2, usable="no"),
    take("A", "x", 3),  # latest usable: kept
    take("A", "y", 1, usable="no"),  # nothing usable on this beat
    take("A", "z", 1, usable=""),  # not judged yet
    take("A", "x", 1, mode="free"),  # free takes wait for after the talk
    take("B", "x", 2),
    take("B", "x", 1),  # earlier than take 2, listed after it
]


def test_keeps_the_latest_usable_click_take_per_member_and_beat():
    kept = ex.select_takes(TAKES)
    assert [r["query_id"] for r in kept] == ["team_A_x_click_3", "team_B_x_click_2"]


def write_repo(tmp_path, takes):
    for row in takes:
        wav = tmp_path / row["audio_path"]
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF fake")
    log = tmp_path / "takes.csv"
    with log.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(takes[0]))
        w.writeheader()
        w.writerows(takes)
    header = tmp_path / "query_manifest.csv"
    header.write_text(",".join(HEADER) + "\n")
    return ["--takes", str(log), "--columns-from", str(header), "--root", str(tmp_path),
            "--out", str(tmp_path / "out.csv"), "--zip", str(tmp_path / "out.zip")]  # fmt: skip


def test_main_writes_the_manifest_and_zip(tmp_path, capsys):
    args = write_repo(tmp_path, TAKES)
    assert ex.main(args) == 0
    with (tmp_path / "out.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == HEADER  # the query manifest's columns, in its order
    assert [r["query_id"] for r in rows] == ["team_A_x_click_3", "team_B_x_click_2"]
    assert rows[0]["seed"] == rows[0]["jitter_ms"] == ""  # generator columns stay empty
    assert rows[0]["participant_id"] == "team_A" and rows[0]["mode"] == "click"
    names = zipfile.ZipFile(tmp_path / "out.zip").namelist()
    assert sorted(names) == [
        "data/recordings/team/team_A_x_click_3.wav",
        "data/recordings/team/team_B_x_click_2.wav",
        "manifests/out.csv",
    ]
    assert "2 queries (team_A 1, team_B 1) over 1 loops" in capsys.readouterr().out


def test_missing_audio_stops_the_export(tmp_path, capsys):
    args = write_repo(tmp_path, TAKES)
    (tmp_path / "data/recordings/team/team_B_x_click_2.wav").unlink()
    assert ex.main(args) == 1
    assert "no audio here" in capsys.readouterr().err
    assert not (tmp_path / "out.csv").exists()


def test_nothing_usable_stops_the_export(tmp_path, capsys):
    args = write_repo(tmp_path, [take("A", "x", 1, usable="no")])
    assert ex.main(args) == 1
    assert "no click take is marked usable" in capsys.readouterr().err


def test_header_must_have_what_the_answer_key_reads(tmp_path):
    with pytest.raises(ex.ExportError, match="participant_id"):
        ex.write_manifest([], [c for c in HEADER if c != "participant_id"], tmp_path / "m.csv")

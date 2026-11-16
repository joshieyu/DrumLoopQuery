"""The recording-session server: beat list, take framing, files, log, and HTTP round trip."""

import csv
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.music.spec import load_spec
from tools.listen import record_session as rs

NORMAL = {"target_rms_dbfs": -20.0, "peak_ceiling_dbfs": -1.0}
BEAT_PICKER_COLUMNS = (
    "member,pattern_id,clip_id,source_performance,start_bar,tempo_bpm,style,feel,"
    "triplet_share,loose_share,density,sounds,sounds_per_s,min_gap_ms"
)


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def write_beats(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([BEAT_PICKER_COLUMNS, *rows]) + "\n")
    return path


@pytest.fixture
def beats(tmp_path, spec):
    path = write_beats(
        tmp_path / "recording_beats.csv",
        [
            "A,87391916ea0e7e12,drummer1/eval_session/1:bar0,drummer1/eval_session/1,0,120,funk-groove1,"
            "straight,0.00,0.00,busy,20,4.1,125",
            "B,850d7aaae0401186,drummer7/session2/114:bar2,drummer7/session2/114,2,96,jazz-fusion,"
            "swung,0.50,0.10,sparse,9,1.6,312",
        ],
    )
    return rs.read_beats(path, spec, {"drummer1/eval_session/1:bar0": "374d35c6327b3c39"})


def test_read_beats(beats, spec):
    a, b = beats
    assert (a.index, a.member, a.pattern_id, a.dualized_pattern_id) == (
        0,
        "A",
        "87391916ea0e7e12",
        "374d35c6327b3c39",
    )
    assert b.dualized_pattern_id == ""
    assert a.bar_beats == 4 and a.beats == spec.loops.bars * 4
    assert a.loop_s == pytest.approx(a.beats * 60 / 120)
    assert (b.style, b.start_bar, b.tempo_bpm) == ("jazz-fusion", 2, 96.0)


@pytest.mark.parametrize(
    "row, message",
    [
        ("E,87391916ea0e7e12,c,p,0,120,s,f,0,0,d,1,1,1", "member"),
        ("A,not-hex,c,p,0,120,s,f,0,0,d,1,1,1", "pattern_id"),
        ("A,87391916ea0e7e12,c,p,0,0,s,f,0,0,d,1,1,1", "tempo"),
    ],
)
def test_read_beats_rejects_bad_rows(tmp_path, spec, row, message):
    with pytest.raises(ValueError, match=message):
        rs.read_beats(write_beats(tmp_path / "b.csv", [row]), spec)


def test_read_beats_needs_beat_picker_columns(tmp_path, spec):
    path = tmp_path / "b.csv"
    path.write_text("member,clip_id\nA,x\n")
    with pytest.raises(ValueError, match="missing columns"):
        rs.read_beats(path, spec)


def clicks_at(times_s: list[float], sample_rate: int, length_s: float) -> np.ndarray:
    x = np.zeros(round(length_s * sample_rate), dtype=np.float32)
    x[[round(t * sample_rate) for t in times_s]] = 0.5
    return x


def test_frame_take_puts_the_downbeat_at_the_lead_in(beats, spec):
    sr, beat = 8000, beats[0]
    downbeat = 2.3  # one bar of count-in at 120 BPM plus 300 ms of latency
    hits = [downbeat + k * 0.5 for k in range(int(beat.beats))]
    framed = rs.frame_take(clicks_at(hits, sr, 8.0), sr, downbeat, beat, spec, NORMAL)
    lead = spec.loops.audio.lead_in_s
    assert len(framed) == round((lead + beat.loop_s) * sr)
    found = np.flatnonzero(np.abs(framed) > 0.01)
    assert found[0] == round(lead * sr)
    assert np.allclose(found / sr - lead, np.arange(len(found)) * 0.5, atol=1 / sr)


def test_frame_take_pads_when_the_recording_starts_late(beats, spec):
    sr, beat = 8000, beats[0]
    framed = rs.frame_take(clicks_at([0.1], sr, 5.0), sr, 0.1, beat, spec, NORMAL)
    assert np.flatnonzero(np.abs(framed) > 0.01)[0] == round(spec.loops.audio.lead_in_s * sr)


def save(out, beat, kind, spec, downbeat=None):
    raw = clicks_at([2.3, 2.8, 3.3], 8000, 7.0)
    return rs.save_take(
        out,
        beat,
        kind,
        raw,
        8000,
        spec=spec,
        normalization=NORMAL,
        downbeat_s=downbeat,
        latency_ms=31.5,
        latency_source="measured",
        count_in_bars=1,
    )


def test_save_take_numbers_takes_and_logs_them(tmp_path, beats, spec):
    first = save(tmp_path, beats[0], "click", spec, downbeat=2.3)
    second = save(tmp_path, beats[0], "click", spec, downbeat=2.3)
    free = save(tmp_path, beats[0], "free", spec)
    assert first["query_id"] == "team_A_87391916ea0e7e12_click_1"
    assert second["query_id"] == "team_A_87391916ea0e7e12_click_2"
    assert free["query_id"] == "team_A_87391916ea0e7e12_free_1"
    for row in (first, second, free):
        name = row["query_id"] + ".wav"
        assert (tmp_path / name).is_file() and (tmp_path / "raw" / name).is_file()
    framed, sr = sf.read(tmp_path / "team_A_87391916ea0e7e12_click_1.wav")
    assert sr == 8000 and len(framed) == round((spec.loops.audio.lead_in_s + beats[0].loop_s) * sr)
    raw, _ = sf.read(tmp_path / "raw" / "team_A_87391916ea0e7e12_click_1.wav")
    assert len(raw) == 7 * 8000
    whole, _ = sf.read(tmp_path / "team_A_87391916ea0e7e12_free_1.wav")
    assert len(whole) == 7 * 8000
    with (tmp_path / "takes.csv").open(newline="") as f:
        log = list(csv.DictReader(f))
    assert tuple(log[0]) == rs.COLUMNS
    assert [r["query_id"] for r in log] == [first["query_id"], second["query_id"], free["query_id"]]
    assert log[0]["participant_id"] == "team_A" and log[0]["mode"] == "click"
    assert log[0]["split"] == "test" and log[0]["dualized_pattern_id"] == "374d35c6327b3c39"
    assert log[0]["downbeat_s"] == "2.3" and log[2]["downbeat_s"] == ""
    assert log[0]["peak_dbfs"] == "-6.0"


def test_save_take_refuses_silence_and_missing_downbeat(tmp_path, beats, spec):
    with pytest.raises(ValueError, match="silent"):
        rs.save_take(
            tmp_path,
            beats[0],
            "free",
            np.zeros(800, dtype=np.float32),
            8000,
            spec=spec,
            normalization=NORMAL,
            downbeat_s=None,
            latency_ms=0,
            latency_source="estimated",
            count_in_bars=1,
        )
    with pytest.raises(ValueError, match="downbeat"):
        save(tmp_path, beats[0], "click", spec, downbeat=None)


def test_set_verdict(tmp_path, beats, spec):
    row = save(tmp_path, beats[1], "click", spec, downbeat=2.3)
    rs.set_verdict(tmp_path, row["query_id"], "yes", "late on bar 2\nbut fine")
    [logged] = rs.read_takes(tmp_path)
    assert (logged["usable"], logged["notes"]) == ("yes", "late on bar 2 but fine")
    with pytest.raises(ValueError):
        rs.set_verdict(tmp_path, row["query_id"], "maybe", "")
    with pytest.raises(KeyError):
        rs.set_verdict(tmp_path, "team_A_00_click_9", "no", "")


class FakeLoops:
    source = "fake"

    def wav(self, beat):
        if beat.member == "B":
            raise FileNotFoundError("no audio for it")
        return rs.wav_bytes(np.zeros(800, dtype=np.float32), 8000)


@pytest.fixture
def server(tmp_path, beats, spec):
    session = rs.Session(beats, spec, tmp_path, FakeLoops(), normalization=NORMAL, member="B")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(session))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url, data=None, content_type="application/octet-stream"):
    request = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


def test_server_round_trip(server, tmp_path):
    status, _, body = call(server + "/")
    assert status == 200 and b"Beatbox recording session" in body
    status, _, body = call(server + "/api/session")
    state = json.loads(body)
    assert state["member"] == "B" and state["members"] == ["A", "B"]
    assert state["beats"][0]["loop_s"] == pytest.approx(4.0)
    assert state["settings"]["plays"] == 2 and state["takes"] == []

    status, headers, body = call(server + "/api/loop/0.wav")
    assert status == 200 and headers["X-Loop-Source"] == "fake"
    assert sf.read(io.BytesIO(body))[1] == 8000
    assert call(server + "/api/loop/1.wav")[0] == 404
    assert call(server + "/api/loop/9.wav")[0] == 404

    raw = clicks_at([2.3, 2.8], 8000, 7.0).astype("<f4").tobytes()
    query = (
        "beat=0&kind=click&sample_rate=8000&latency_ms=31.5&latency_source=measured&downbeat_s=2.3"
    )
    status, _, body = call(f"{server}/api/take?{query}", raw)
    row = json.loads(body)
    assert status == 200 and row["query_id"] == "team_A_87391916ea0e7e12_click_1"
    assert call(f"{server}/takes/{row['query_id']}.wav")[0] == 200
    assert call(f"{server}/takes/raw/{row['query_id']}.wav")[0] == 200
    assert call(f"{server}/takes/takes.csv")[0] == 404
    assert call(f"{server}/takes/..%2Ftakes.csv")[0] == 404

    verdict = json.dumps(
        {"query_id": row["query_id"], "usable": "no", "notes": "lost bar 2"}
    ).encode()
    status, _, body = call(server + "/api/verdict", verdict, "application/json")
    assert status == 200 and json.loads(body)["usable"] == "no"
    assert json.loads(call(server + "/api/session")[2])["takes"][0]["notes"] == "lost bar 2"


@pytest.mark.parametrize(
    "query, body",
    [
        ("beat=5&kind=click&sample_rate=8000&latency_ms=0&latency_source=measured&downbeat_s=1", 4),
        ("beat=0&kind=loud&sample_rate=8000&latency_ms=0&latency_source=measured&downbeat_s=1", 4),
        ("beat=0&kind=click&sample_rate=8000&latency_ms=0&latency_source=guess&downbeat_s=1", 4),
        (
            "beat=0&kind=click&sample_rate=8000&latency_ms=5000&latency_source=measured&downbeat_s=1",
            4,
        ),
        ("beat=0&kind=click&sample_rate=8000&latency_ms=0&latency_source=measured&downbeat_s=1", 3),
        ("beat=0&kind=click&sample_rate=8000&latency_ms=0&latency_source=measured&downbeat_s=", 4),
    ],
)
def test_server_rejects_bad_takes(server, tmp_path, query, body):
    status, _, _ = call(f"{server}/api/take?{query}", np.full(body, 1, dtype=np.uint8).tobytes())
    assert status == 400
    assert not list(tmp_path.glob("*.wav"))

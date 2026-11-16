"""Record the team's real beatbox queries (README §13, spec §8) from a local web page.

    .venv/bin/python -m tools.listen.record_session --beats ~/Downloads/recording_beats.csv
    then open http://127.0.0.1:8770 in Chrome, with wired headphones on

The beat list is the beat picker's export (`tools.listen.beat_picker`, one row per member
and beat). For each beat the page plays the target loop twice, counts in at the loop's
tempo, and records two bars of beatbox with the click in the headphones. A free take
(no click) records after the same two plays until Stop.

Each take is saved under data/recordings/team/ with README §13's name,
`team_<member>_<pattern_id>_<click|free>_<take>.wav`:

- A click take is framed like the library clips and the fake queries: the spec's lead-in
  before the take's downbeat, a cut at the end of the loop with the spec's fade, and the
  fake queries' loudness (config/dataset_v0.yaml). The downbeat is where the click put
  it, shifted by the audio latency (measured on the page, or the browser's estimate).
- A free take keeps the whole recording at that loudness; its timing is the performer's.
- raw/ keeps every take unprocessed, count-in included, so it can be re-cut later.

takes.csv logs every take with the query-manifest columns (README §12), the recording
details, and the music lead's usable/notes verdict (spec §8). The page and the audio stay
on this machine: the server listens on 127.0.0.1 only, and data/ is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import yaml

from src.audio import frame_audio, normalize_audio
from src.music.spec import MusicSpec, load_spec
from tools.listen.beat_picker import CATALOG_DIR, DEFAULT_KIT, GMD_DIR, PLAYS

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "recordings" / "team"
PAGE = Path(__file__).with_suffix(".html")
DATASET_CONFIG = REPO / "config" / "dataset_v0.yaml"  # B's build settings: query loudness
LIBRARY_MANIFEST = REPO / "manifests" / "library_manifest.csv"
DEFAULT_PORT = 8770
COUNT_IN_BARS = 1  # session protocol, not a musical rule; --count-in-bars changes it
MEMBERS = ("A", "B", "C", "D")
KINDS = ("click", "free")
MAX_TAKE_S = 60.0  # longest upload accepted, count-in included
TAKE_NAME = re.compile(r"team_([A-D])_([0-9a-f]+)_(click|free)_([0-9]+)\.wav")
PATTERN_ID = re.compile(r"[0-9a-f]+")
COLUMNS = (
    # README §12 query-manifest columns, so B can merge the usable takes
    "query_id",
    "pattern_id",
    "dualized_pattern_id",
    "split",
    "tempo_bpm",
    "source_performance",
    "start_bar",
    "source_clip_id",
    "query_type",
    "source",
    "participant_id",
    "mode",
    "audio_path",
    "duration_s",
    "sample_rate",
    # recording details
    "take",
    "raw_path",
    "downbeat_s",
    "latency_ms",
    "latency_source",
    "count_in_bars",
    "peak_dbfs",
    "recorded_at",
    # the music lead's verdict (spec §8: keep imperfect takes)
    "usable",
    "notes",
)


@dataclass(frozen=True)
class Beat:
    index: int
    member: str
    clip_id: str
    pattern_id: str
    dualized_pattern_id: str
    tempo_bpm: float
    source_performance: str
    start_bar: int
    style: str
    sounds_per_s: str
    bar_beats: float
    beats: float

    @property
    def loop_s(self) -> float:
        return self.beats * 60 / self.tempo_bpm


def bar_beats(spec: MusicSpec) -> float:
    """Quarter-note beats per bar, from the spec's single time signature."""
    if len(spec.loops.time_signatures) != 1:
        raise ValueError("the recorder assumes the spec keeps one time signature")
    numerator, denominator = map(int, spec.loops.time_signatures[0].split("-"))
    return numerator * 4 / denominator


def dualized_ids(library_manifest: Path) -> dict[str, str]:
    """source_clip_id -> dualized_pattern_id, from B's library manifest when it exists."""
    if not library_manifest.is_file():
        return {}
    with library_manifest.open(newline="") as f:
        return {r["source_clip_id"]: r["dualized_pattern_id"] for r in csv.DictReader(f)}


def read_beats(path: Path, spec: MusicSpec, dualized: dict[str, str] | None = None) -> list[Beat]:
    """The beat picker's recording_beats.csv, one Beat per member and beat."""
    dualized = dualized or {}
    per_bar = bar_beats(spec)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no beats")
    missing = {"member", "clip_id", "pattern_id", "tempo_bpm"} - set(rows[0])
    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}; export it from the beat picker"
        )
    beats = []
    for i, r in enumerate(rows):
        member, pattern = r["member"].strip(), r["pattern_id"].strip()
        tempo = float(r["tempo_bpm"])
        if member not in MEMBERS:
            raise ValueError(f"row {i + 2}: member must be one of {MEMBERS}, got {member!r}")
        if not PATTERN_ID.fullmatch(pattern):
            raise ValueError(f"row {i + 2}: pattern_id {pattern!r} is not a hex pattern id")
        if not (math.isfinite(tempo) and tempo > 0):
            raise ValueError(f"row {i + 2}: tempo_bpm must be positive")
        beats.append(
            Beat(
                index=i,
                member=member,
                clip_id=r["clip_id"].strip(),
                pattern_id=pattern,
                dualized_pattern_id=dualized.get(r["clip_id"].strip(), ""),
                tempo_bpm=tempo,
                source_performance=r.get("source_performance", "").strip(),
                start_bar=int(r.get("start_bar") or 0),
                style=r.get("style", "").strip(),
                sounds_per_s=r.get("sounds_per_s", "").strip(),
                bar_beats=per_bar,
                beats=spec.loops.bars * per_bar,
            )
        )
    return beats


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class LoopAudio:
    """The target loop played PLAYS times, as WAV bytes.

    Rendered through a kit from the GMD catalog (as the beat picker plays it) when the
    kit and GMD are on this machine; otherwise B's rendered library clip of the same
    loop, repeated without its lead-in, when the v0.7 audio is here.
    """

    def __init__(
        self,
        spec: MusicSpec,
        kit_id: str = DEFAULT_KIT,
        catalog_dir: Path = CATALOG_DIR,
        gmd_dir: Path = GMD_DIR,
        library_manifest: Path = LIBRARY_MANIFEST,
    ):
        self.spec, self.kit_id = spec, kit_id
        self.catalog_dir, self.gmd_dir, self.library_manifest = (
            catalog_dir,
            gmd_dir,
            library_manifest,
        )
        self._lock = threading.Lock()
        self._cache: dict[str, bytes] = {}
        self._renderer = None  # (records by clip_id, kit, sampler), or the reason it failed
        self.source = "not loaded yet"

    def _render(self, beat: Beat) -> bytes | None:
        if self._renderer is None:
            try:
                from tools.listen.beat_picker import load_catalog
                from tools.render.kits import kit_by_id, load_program
                from tools.render.sampler import Sampler

                records = {
                    r["clip_id"]: r for r in load_catalog(self.catalog_dir, self.spec, self.gmd_dir)
                }
                kit = kit_by_id(self.spec, self.kit_id)
                self._renderer = (records, kit, Sampler(load_program(kit), seed=0))
            except Exception as exc:  # no GMD, no kit samples: fall back to library audio
                self._renderer = f"{type(exc).__name__}: {exc}"
        if isinstance(self._renderer, str):
            return None
        records, kit, sampler = self._renderer
        record = records.get(beat.clip_id)
        if record is None:
            return None
        from tools.listen.beat_picker import render_loop

        self.source = f"rendered through {kit.kit_id}"
        return wav_bytes(render_loop(record, self.spec, kit, sampler), sampler.sample_rate)

    def _library(self, beat: Beat) -> bytes | None:
        if not self.library_manifest.is_file():
            return None
        with self.library_manifest.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["source_clip_id"] == beat.clip_id]
        for r in rows:
            path = Path(r["audio_path"])
            path = path if path.is_absolute() else REPO / path
            if path.is_file():
                x, sr = sf.read(path, dtype="float32")
                lead = round(self.spec.loops.audio.lead_in_s * sr)
                self.source = f"B's library audio ({r['kit_id']})"
                return wav_bytes(np.concatenate([x] + [x[lead:]] * (PLAYS - 1)), sr)
        return None

    def wav(self, beat: Beat) -> bytes:
        with self._lock:
            if beat.clip_id not in self._cache:
                audio = self._render(beat) or self._library(beat)
                if audio is None:
                    why = (
                        self._renderer if isinstance(self._renderer, str) else "not in the catalog"
                    )
                    raise FileNotFoundError(
                        f"no audio for {beat.clip_id}: rendering failed ({why}) and B's library "
                        "audio for it isn't on this machine"
                    )
                self._cache[beat.clip_id] = audio
            return self._cache[beat.clip_id]


def query_normalization(config: Path = DATASET_CONFIG) -> dict:
    """The fake queries' loudness settings, so real takes match them."""
    return dict(yaml.safe_load(config.read_text())["normalization"])


def take_name(member: str, pattern_id: str, kind: str, take: int) -> str:
    return f"team_{member}_{pattern_id}_{kind}_{take}.wav"


def next_take(out: Path, member: str, pattern_id: str, kind: str) -> int:
    used = [
        int(m.group(4))
        for folder in (out, out / "raw")
        if folder.is_dir()
        for p in folder.glob(f"team_{member}_{pattern_id}_{kind}_*.wav")
        if (m := TAKE_NAME.fullmatch(p.name))
    ]
    return max(used, default=0) + 1


def frame_take(
    raw: np.ndarray,
    sample_rate: int,
    downbeat_s: float,
    beat: Beat,
    spec: MusicSpec,
    normalization: dict,
) -> np.ndarray:
    """Cut a click take like a library clip: the lead-in before its downbeat, the loop, the fade."""
    lead = spec.loops.audio.lead_in_s
    duration = lead + beat.loop_s
    start = round((downbeat_s - lead) * sample_rate)
    frames = round(duration * sample_rate)
    x = np.asarray(raw, dtype=np.float32)
    if start < 0:
        x, start = np.concatenate([np.zeros(-start, dtype=np.float32), x]), 0
    segment = x[start : start + frames]
    return frame_audio(
        normalize_audio(segment, **normalization),
        sample_rate=sample_rate,
        duration_s=duration,
        settings=spec.loops.audio,
    )


def repo_path(path: Path) -> str:
    """Manifest paths are relative to the repo root when they can be."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def read_takes(out: Path) -> list[dict]:
    log = out / "takes.csv"
    if not log.is_file():
        return []
    with log.open(newline="") as f:
        return list(csv.DictReader(f))


def write_takes(out: Path, rows: list[dict]) -> None:
    log = out / "takes.csv"
    tmp = log.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, log)


def save_take(
    out: Path,
    beat: Beat,
    kind: str,
    raw: np.ndarray,
    sample_rate: int,
    *,
    spec: MusicSpec,
    normalization: dict,
    downbeat_s: float | None,
    latency_ms: float,
    latency_source: str,
    count_in_bars: int,
    now: datetime | None = None,
) -> dict:
    """Write the take (framed for click, whole for free) and its raw copy, and log it."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if kind == "click" and downbeat_s is None:
        raise ValueError("a click take needs its downbeat time")
    raw = np.asarray(raw, dtype=np.float32)
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("the recording is empty or not finite")
    peak = float(np.abs(raw).max())
    if peak == 0:
        raise ValueError("the recording is silent: is the mic on?")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    take = next_take(out, beat.member, beat.pattern_id, kind)
    name = take_name(beat.member, beat.pattern_id, kind, take)
    sf.write(out / "raw" / name, raw, sample_rate, subtype="FLOAT")
    if kind == "click":
        audio = frame_take(raw, sample_rate, downbeat_s, beat, spec, normalization)
    else:
        audio = normalize_audio(raw, **normalization)
    sf.write(out / name, audio, sample_rate, subtype="PCM_16")
    row = {
        "query_id": Path(name).stem,
        "pattern_id": beat.pattern_id,
        "dualized_pattern_id": beat.dualized_pattern_id,
        "split": "test",  # README §13: recorded beats come from the test split
        "tempo_bpm": beat.tempo_bpm,
        "source_performance": beat.source_performance,
        "start_bar": beat.start_bar,
        "source_clip_id": beat.clip_id,
        "query_type": "beatbox",
        "source": "team",
        "participant_id": f"team_{beat.member}",
        "mode": kind,
        "audio_path": repo_path(out / name),
        "duration_s": len(audio) / sample_rate,
        "sample_rate": sample_rate,
        "take": take,
        "raw_path": repo_path(out / "raw" / name),
        "downbeat_s": "" if downbeat_s is None else round(downbeat_s, 6),
        "latency_ms": round(latency_ms, 1),
        "latency_source": latency_source,
        "count_in_bars": count_in_bars if kind == "click" else "",
        "peak_dbfs": round(20 * math.log10(peak), 1),
        "recorded_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "usable": "",
        "notes": "",
    }
    write_takes(out, read_takes(out) + [row])
    return row


def set_verdict(out: Path, query_id: str, usable: str, notes: str) -> dict:
    if usable not in ("", "yes", "no"):
        raise ValueError("usable must be yes, no, or empty")
    rows = read_takes(out)
    for row in rows:
        if row["query_id"] == query_id:
            row["usable"], row["notes"] = usable, notes.replace("\n", " ").strip()
            write_takes(out, rows)
            return row
    raise KeyError(f"no take {query_id!r}")


class Session:
    """Everything the page needs, shared by the request threads."""

    def __init__(
        self,
        beats: list[Beat],
        spec: MusicSpec,
        out: Path,
        loops,
        *,
        normalization: dict,
        count_in_bars: int = COUNT_IN_BARS,
        member: str | None = None,
    ):
        self.beats, self.spec, self.out, self.loops = beats, spec, out, loops
        self.normalization, self.count_in_bars = normalization, count_in_bars
        members = sorted({b.member for b in beats})
        self.member = member if member in members else members[0]
        self.lock = threading.Lock()

    def state(self) -> dict:
        return {
            "settings": {
                "lead_in_s": self.spec.loops.audio.lead_in_s,
                "plays": PLAYS,
                "count_in_bars": self.count_in_bars,
                "max_take_s": MAX_TAKE_S,
                "spec_version": self.spec.version,
                "out": repo_path(self.out),
                "loop_source": getattr(self.loops, "source", ""),
            },
            "members": sorted({b.member for b in self.beats}),
            "member": self.member,
            "beats": [{**asdict(b), "loop_s": b.loop_s} for b in self.beats],
            "takes": read_takes(self.out),
        }


def _number(
    params: dict, key: str, low: float, high: float, *, optional: bool = False
) -> float | None:
    value = params.get(key, [""])[0]
    if value == "" and optional:
        return None
    number = float(value)
    if not (math.isfinite(number) and low <= number <= high):
        raise ValueError(f"{key} must be between {low} and {high}")
    return number


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal for errors only
            pass

        def _send(self, status: int, body: bytes, content_type: str, **headers: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/session":
                return self._json(200, session.state())
            if m := re.fullmatch(r"/api/loop/([0-9]+)\.wav", path):
                i = int(m.group(1))
                if i >= len(session.beats):
                    return self._json(404, {"error": "no such beat"})
                try:
                    audio = session.loops.wav(session.beats[i])
                    source = getattr(session.loops, "source", "")
                    return self._send(200, audio, "audio/wav", X_Loop_Source=source)
                except FileNotFoundError as exc:
                    return self._json(404, {"error": str(exc)})
            if m := re.fullmatch(r"/takes/(raw/)?([^/]+)", path):
                if not TAKE_NAME.fullmatch(m.group(2)):
                    return self._json(404, {"error": "not a take"})
                file = session.out / (m.group(1) or "") / m.group(2)
                if file.is_file():
                    return self._send(200, file.read_bytes(), "audio/wav")
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_TAKE_S * 192_000 * 4:
                return self._json(413, {"error": "recording too long"})
            body = self.rfile.read(length)
            try:
                if url.path == "/api/take":
                    params = parse_qs(url.query)
                    i = int(params.get("beat", ["-1"])[0])
                    if not 0 <= i < len(session.beats):
                        raise ValueError("no such beat")
                    kind = params.get("kind", [""])[0]
                    if kind not in KINDS:
                        raise ValueError(f"kind must be one of {KINDS}")
                    source = params.get("latency_source", [""])[0]
                    if source not in ("measured", "estimated"):
                        raise ValueError("latency_source must be measured or estimated")
                    if length % 4:
                        raise ValueError("the body must be 32-bit float samples")
                    with session.lock:
                        row = save_take(
                            session.out,
                            session.beats[i],
                            kind,
                            np.frombuffer(body, dtype="<f4"),
                            int(_number(params, "sample_rate", 8_000, 192_000)),
                            spec=session.spec,
                            normalization=session.normalization,
                            downbeat_s=_number(params, "downbeat_s", 0, MAX_TAKE_S, optional=True),
                            latency_ms=_number(params, "latency_ms", 0, 1_000),
                            latency_source=source,
                            count_in_bars=session.count_in_bars,
                        )
                    return self._json(200, row)
                if url.path == "/api/verdict":
                    data = json.loads(body or b"{}")
                    with session.lock:
                        row = set_verdict(
                            session.out,
                            str(data.get("query_id", "")),
                            str(data.get("usable", "")),
                            str(data.get("notes", "")),
                        )
                    return self._json(200, row)
            except (ValueError, KeyError) as exc:
                return self._json(400, {"error": str(exc)})
            self._json(404, {"error": "not found"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--beats", type=Path, required=True, help="recording_beats.csv from the beat picker"
    )
    ap.add_argument("--member", choices=MEMBERS, help="whose beats the page opens on")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--kit", default=DEFAULT_KIT, help="kit the target loop plays through")
    ap.add_argument("--count-in-bars", type=int, default=COUNT_IN_BARS, choices=range(0, 5))
    ap.add_argument("--catalog", type=Path, default=CATALOG_DIR)
    ap.add_argument("--gmd", type=Path, default=GMD_DIR)
    args = ap.parse_args(argv)

    spec = load_spec()
    beats = read_beats(args.beats.expanduser(), spec, dualized_ids(LIBRARY_MANIFEST))
    args.out.mkdir(parents=True, exist_ok=True)
    session = Session(
        beats,
        spec,
        args.out,
        LoopAudio(spec, args.kit, args.catalog, args.gmd),
        normalization=query_normalization(),
        count_in_bars=args.count_in_bars,
        member=args.member,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    members = ", ".join(
        f"{m} {sum(b.member == m for b in beats)}" for m in sorted({b.member for b in beats})
    )
    print(f"{len(beats)} beats ({members}); takes go to {repo_path(args.out)}")
    print(f"Open http://127.0.0.1:{args.port} in Chrome, with wired headphones on. Ctrl+C stops.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

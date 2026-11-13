"""Practice data for Member C until B's library and query manifests land (week 3 handoff).

THROWAWAY. This is not B's pipeline and not the team's data. It exists so the model,
training, and scoring code can run end to end on real drum audio now.

    .venv/bin/python -m src.library.catalog_gmd --out data/practice/gmd_catalog
    .venv/bin/python -m src.models.practice                  # 400 / 50 / 100 loops
    .venv/bin/python -m src.models.practice --train 20 --validation 5 --test 5

Reads B's MIDI loop catalog and writes, under data/practice/ (gitignored):

- audio/library/*.flac: each loop on the library kit (default avl_black_pearl)
- audio/queries/*.flac: stand-in queries, each loop on another kit (default
  avl_blonde_bop) with sloppiness drawn from the spec's ranges
- manifests/library_manifest.csv, query_manifest.csv (README §12 columns, plus
  start_bar and offset_ms), and relevance.json (practice answer key)
- listen.html: a few test loops next to their stand-in queries, to check by ear

Stand-in queries are drum-kit audio, not beatboxes. Their query_type is "beatbox" only
so they exercise the query side of the model, and their source is "practice".

Practice-only choices (B owns the real ones): a 50 ms lead-in before each clip; a clip
holds every hit whose nearest grid step falls inside the loop and ends at the loop's
end; per-hit timing wobble is drawn uniformly within plus or minus the clip's amount;
the noise is white; no tempo drift or mic EQ. Patterns found in more than one GMD
split are left out. Every musical number (tempo limits, sloppiness ranges, grid) comes
from config/music_spec.yaml.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.models.config import load_model_config
from src.models.features import resample
from src.music.rhythm import Note
from src.music.spec import MusicSpec, load_spec
from tools.render.kits import kit_by_id, kit_hits, load_program, normalize
from tools.render.sampler import Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
PRACTICE_DIR = REPO_ROOT / "data" / "practice"
CATALOG = PRACTICE_DIR / "gmd_catalog" / "loops.jsonl"
DEFAULT_COUNTS = {"train": 400, "validation": 50, "test": 100}


@dataclass(frozen=True)
class Loop:
    """One labelled 2-bar loop from B's catalog (loops.jsonl)."""

    clip_id: str
    pattern_id: str
    dualized_pattern_id: str
    split: str
    tempo_bpm: float
    beats: float  # loop length in beats
    notes: tuple[Note, ...]  # beats from the loop's first downbeat, with context
    midi_path: str
    source_performance: str
    start_bar: int


@dataclass(frozen=True)
class QuerySloppiness:
    """One stand-in query's draw from the spec's sloppiness ranges (spec §6)."""

    offset_ms: float
    jitter_ms: float
    drop_prob: dict[str, float]
    snr_db: float


# --------------------------------------------------------------------------- choosing loops


def read_catalog(path: str | Path = CATALOG) -> list[Loop]:
    """Every loop in B's loops.jsonl."""
    loops = []
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            loops.append(
                Loop(
                    clip_id=r["clip_id"],
                    pattern_id=r["pattern_id"],
                    dualized_pattern_id=r["dualized_pattern_id"],
                    split=r["split"],
                    tempo_bpm=float(r["tempo_bpm"]),
                    beats=float(r["end_beat"]) - float(r["start_beat"]),
                    notes=tuple(
                        Note(int(n["pitch"]), int(n["velocity"]), float(n["beat"]))
                        for n in r["notes"]
                    ),
                    midi_path=r["midi_path"],
                    source_performance=r["source_performance"],
                    start_bar=int(r["start_bar"]),
                )
            )
    return loops


def select_loops(loops: list[Loop], counts: dict[str, int], seed: int) -> list[Loop]:
    """counts[split] loops per split, at random. Patterns in more than one split are skipped."""
    splits_of: dict[str, set[str]] = {}
    for loop in loops:
        splits_of.setdefault(loop.pattern_id, set()).add(loop.split)
    usable = [loop for loop in loops if len(splits_of[loop.pattern_id]) == 1]
    rng = np.random.default_rng(seed)
    chosen = []
    for split, n in counts.items():
        pool = sorted((loop for loop in usable if loop.split == split), key=lambda x: x.clip_id)
        if n > len(pool):
            raise ValueError(f"asked for {n} {split} loops, but only {len(pool)} are usable")
        chosen.extend(pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False)))
    return chosen


# --------------------------------------------------------------------------- timing


def loop_hits(loop: Loop, spec: MusicSpec) -> list[tuple[float, int, int]]:
    """(beat, GMD note, velocity) for every hit whose nearest grid step is inside the loop."""
    half_step = 0.5 / spec.same_rhythm.steps_per_beat
    return [
        (n.beat, n.pitch, n.velocity)
        for n in loop.notes
        if -half_step <= n.beat < loop.beats - half_step
    ]


def render_tempo(recorded_bpm: float, spec: MusicSpec, rng: np.random.Generator) -> float:
    """A tempo within rendering.max_tempo_change_pct of the recording, inside tempo_bpm."""
    lo, hi = spec.rendering.tempo_bpm
    pct = spec.rendering.max_tempo_change_pct or 0.0
    a = max(lo, recorded_bpm * (1 - pct / 100))
    b = min(hi, recorded_bpm * (1 + pct / 100))
    return float(rng.uniform(a, b)) if a < b else float(np.clip(recorded_bpm, lo, hi))


def draw_sloppiness(spec: MusicSpec, rng: np.random.Generator) -> QuerySloppiness:
    """One random amount of each effect, uniform within the spec's [low, high] ranges."""
    s = spec.sloppiness
    return QuerySloppiness(
        offset_ms=float(rng.uniform(*s.offset_ms)),
        jitter_ms=float(rng.uniform(*s.jitter_ms)),
        drop_prob={voice: float(rng.uniform(*r)) for voice, r in s.drop_prob.items()},
        snr_db=float(rng.uniform(*s.snr_db)),
    )


def timed_hits(
    hits: list[tuple[float, int, int]],
    bpm: float,
    spec: MusicSpec,
    rng: np.random.Generator,
    slop: QuerySloppiness | None = None,
) -> list[tuple[float, int, int]]:
    """(time_s, GMD note, velocity) at `bpm`, after the lead-in; with `slop`, made sloppy.

    Sloppiness drops labelled hits with their voice's probability, then shifts every
    remaining hit by the clip's offset plus its own wobble. Dropped-voice notes (toms,
    crashes) are never dropped: they aren't part of the rhythm.
    """
    seconds_per_beat = 60.0 / bpm
    out = []
    for beat, pitch, velocity in hits:
        t = spec.loops.audio.lead_in_s + beat * seconds_per_beat
        if slop is not None:
            voice = spec.voice_of(pitch)
            if voice is not None and rng.random() < slop.drop_prob[voice]:
                continue
            t += (slop.offset_ms + rng.uniform(-1.0, 1.0) * slop.jitter_ms) / 1000.0
        out.append((max(t, 0.0), pitch, velocity))
    return out


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White noise at `snr_db` below the clip's mean power."""
    power = float(np.mean(np.square(audio, dtype=np.float64)))
    sigma = np.sqrt(power / 10 ** (snr_db / 10))
    return (audio + rng.normal(0.0, sigma, audio.shape)).astype(np.float32)


# --------------------------------------------------------------------------- rendering


class KitRenderer:
    """One kit through Joshua's sampler, mixed to mono at the model's sample rate."""

    def __init__(self, spec: MusicSpec, kit_id: str, sample_rate: int, seed: int):
        self.kit_id = kit_id
        self.kit = kit_by_id(spec, kit_id)
        self.sampler = Sampler(load_program(self.kit), seed=seed)
        self.sample_rate = sample_rate

    def render(self, hits_s: list[tuple[float, int, int]], length_s: float) -> np.ndarray:
        stereo = self.sampler.render(kit_hits(self.kit, hits_s), length_s=length_s)
        return resample(stereo.mean(axis=1), self.sampler.sample_rate, self.sample_rate)


def safe_name(clip_id: str) -> str:
    """A file-name-safe version of a catalog clip id ('drummer1/session1/5:bar3')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", clip_id).strip("_")


def repo_relative(path: Path) -> str:
    """The path relative to the repo root when inside it (as manifests store it)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_16")


def relevance(library: pd.DataFrame, queries: pd.DataFrame) -> dict[str, list[str]]:
    """Practice answer key (README §5.4): same pattern_id; for taps, same dualized id.

    Only library clips in the query's own split are listed, so a test set scored
    against its own library never names a clip outside it.
    """
    out = {}
    for q in queries.itertuples(index=False):
        key = "dualized_pattern_id" if q.query_type == "tap" else "pattern_id"
        same = library[(library[key] == getattr(q, key)) & (library.split == q.split)]
        out[q.query_id] = sorted(same.clip_id)
    return out


def build_practice(
    loops: list[Loop],
    spec: MusicSpec,
    sample_rate: int,
    out_dir: Path,
    library_kit: str = "avl_black_pearl",
    query_kit: str = "avl_blonde_bop",
    views_per_query: int = 2,
    seed: int = 0,
    n_listen: int = 10,
    log=print,
) -> dict:
    """Render every loop and its stand-in queries; write manifests and a listening page."""
    lib = KitRenderer(spec, library_kit, sample_rate, seed)
    qry = KitRenderer(spec, query_kit, sample_rate, seed + 1)
    library_rows, query_rows, listen = [], [], []
    started = time.monotonic()
    for i, loop in enumerate(loops):
        hits = loop_hits(loop, spec)
        stem = safe_name(loop.clip_id)
        rng = np.random.default_rng([seed, i])
        bpm = render_tempo(loop.tempo_bpm, spec, rng)
        audio = normalize(lib.render(timed_hits(hits, bpm, spec, rng), _length_s(loop, bpm, spec)))
        lib_path = out_dir / "audio" / "library" / f"{stem}.flac"
        write_flac(lib_path, audio, sample_rate)
        library_rows.append(
            {
                "clip_id": f"{loop.clip_id}@{library_kit}",
                "pattern_id": loop.pattern_id,
                "dualized_pattern_id": loop.dualized_pattern_id,
                "split": loop.split,
                "kit_id": library_kit,
                "tempo_bpm": round(bpm, 3),
                "audio_path": repo_relative(lib_path),
                "midi_path": loop.midi_path,
                "source_performance": loop.source_performance,
                "start_bar": loop.start_bar,
            }
        )
        query_paths = []
        for k in range(views_per_query):
            qrng = np.random.default_rng([seed, i, k + 1])
            qbpm = render_tempo(loop.tempo_bpm, spec, qrng)
            slop = draw_sloppiness(spec, qrng)
            dry = qry.render(timed_hits(hits, qbpm, spec, qrng, slop), _length_s(loop, qbpm, spec))
            audio = normalize(add_noise(dry, slop.snr_db, qrng))
            q_path = out_dir / "audio" / "queries" / f"{stem}_q{k}.flac"
            write_flac(q_path, audio, sample_rate)
            query_paths.append(q_path)
            query_rows.append(
                {
                    "query_id": f"practice:{loop.clip_id}:q{k}",
                    "pattern_id": loop.pattern_id,
                    "dualized_pattern_id": loop.dualized_pattern_id,
                    "query_type": "beatbox",
                    "source": "practice",
                    "participant_id": f"kit:{query_kit}",
                    "split": loop.split,
                    "tempo_bpm": round(qbpm, 3),
                    "jitter_ms": round(slop.jitter_ms, 2),
                    "drift_pct": 0.0,
                    "drop_prob": json.dumps({v: round(p, 3) for v, p in slop.drop_prob.items()}),
                    "snr_db": round(slop.snr_db, 2),
                    "audio_path": repo_relative(q_path),
                    "offset_ms": round(slop.offset_ms, 2),
                }
            )
        if loop.split == "test" and len(listen) < n_listen:
            listen.append((loop, bpm, lib_path, query_paths))
        if (i + 1) % 50 == 0 or i + 1 == len(loops):
            elapsed = time.monotonic() - started
            eta = elapsed / (i + 1) * (len(loops) - i - 1)
            log(f"rendered {i + 1}/{len(loops)} loops, {elapsed:.0f} s so far, ~{eta:.0f} s left")

    library = pd.DataFrame(library_rows)
    queries = pd.DataFrame(query_rows)
    manifests = out_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    library.to_csv(manifests / "library_manifest.csv", index=False)
    queries.to_csv(manifests / "query_manifest.csv", index=False)
    (manifests / "relevance.json").write_text(json.dumps(relevance(library, queries), indent=1))
    _write_listen_page(out_dir / "listen.html", listen, library_kit, query_kit)
    return {
        "loops": len(loops),
        "library_clips": len(library),
        "queries": len(queries),
        "by_split": library.split.value_counts().to_dict(),
        "seconds": round(time.monotonic() - started, 1),
        "out_dir": str(out_dir),
    }


def _length_s(loop: Loop, bpm: float, spec: MusicSpec) -> float:
    return spec.loops.audio.lead_in_s + loop.beats * 60.0 / bpm


def _write_listen_page(path: Path, entries, library_kit: str, query_kit: str) -> None:
    def audio(p: Path) -> str:
        src = html.escape(p.relative_to(path.parent).as_posix())
        return f'<audio controls preload="none" src="{src}"></audio>'

    rows = "\n".join(
        f"<tr><td>{html.escape(loop.clip_id)}<br><small>{loop.tempo_bpm:.0f} BPM recorded, "
        f"{bpm:.0f} BPM rendered</small></td><td>{audio(lib)}</td>"
        f"<td>{'<br>'.join(audio(q) for q in queries)}</td></tr>"
        for loop, bpm, lib, queries in entries
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Practice data: listening check</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px}td{padding:8px;"
        "border-bottom:1px solid #ddd;vertical-align:top}</style>"
        "<h1>Practice data: listening check</h1>"
        f"<p>Each row: a test loop on <b>{library_kit}</b> (library) next to its stand-in "
        f"queries on <b>{query_kit}</b>, made sloppy (timing wobble, dropped hits, noise). "
        "Each pair should play the same rhythm.</p>"
        f"<table><tr><th>Loop</th><th>Library</th><th>Stand-in queries</th></tr>{rows}</table>"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Member C's practice data.")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, default=PRACTICE_DIR)
    for split, n in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{split}", type=int, default=n, help=f"{split} loops")
    parser.add_argument("--library-kit", default="avl_black_pearl")
    parser.add_argument("--query-kit", default="avl_blonde_bop")
    parser.add_argument("--views-per-query", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = load_spec()
    cfg = load_model_config()
    counts = {split: getattr(args, split) for split in DEFAULT_COUNTS}
    loops = select_loops(read_catalog(args.catalog), counts, args.seed)
    summary = build_practice(
        loops,
        spec,
        cfg.audio.sample_rate,
        args.out,
        library_kit=args.library_kit,
        query_kit=args.query_kit,
        views_per_query=args.views_per_query,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    print(f"Listen: open {args.out / 'listen.html'}")


if __name__ == "__main__":
    main()

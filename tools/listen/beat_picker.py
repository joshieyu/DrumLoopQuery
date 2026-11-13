"""Pick the recording-session beats: every GMD test-split loop, with audio and stats.

    .venv/bin/python -m tools.listen.beat_picker
    open results/listening/beat_picker/index.html

README §13 and spec §8: the recorded beats come from the GMD test split, balanced
across straight and swung and across pattern density, skipping beats that are
unreasonable to beatbox. The page lists every test loop with its grid, an MP3 that plays
the 2-bar loop twice through one kit (participants hear the target twice), and:

- `sounds`: grid steps with any hit. A beatboxer voices at most one sound per step.
- `sounds_per_s` and `min_gap_ms`: how fast the sounds come, and the closest pair.
- `straight_off`, `triplet_off`, `other_off`: off-beat sounds on 16th steps, on triplet
  steps, and on neither (loosely played timing). The page turns these into a triplet
  share with a swung cutoff the music lead sets, so no threshold lives here.

Picks are assigned per member (A-D), kept in the browser, and export as
`recording_beats.csv` (one row per member and beat). Audio is at the listening level
(`tools.render.kits.normalize`) with the spec's clip framing; it is for picking only and
never goes into the library.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio import frame_audio
from src.music.spec import Kit, MusicSpec, load_spec
from tools.render.kits import kit_by_id, kit_hits, load_program, normalize
from tools.render.sampler import Sampler

REPO = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPO / "results" / "gmd_catalog"
GMD_DIR = REPO / "data" / "raw" / "gmd" / "groove"
OUT_DIR = REPO / "results" / "listening" / "beat_picker"
PAGE_TEMPLATE = Path(__file__).with_suffix(".html")  # the page; write_page() supplies its data
DEFAULT_KIT = "avl_black_pearl"
SPLIT = "test"  # README §13: recorded beats come from the test split only
PLAYS = 2  # README §13: the participant hears the target loop twice


def style_of(midi_path: str) -> str:
    """GMD style from a MIDI filename, e.g. '1_funk-groove1_138_beat_4-4.mid' -> 'funk-groove1'."""
    parts = Path(midi_path).name.split("_")
    return parts[1] if len(parts) > 1 else ""


def loop_stats(record: dict, spec: MusicSpec) -> dict:
    """Density, speed, and where the off-beat sounds fall, from a catalog record's grid."""
    grid = record["grid"]
    spb = grid["steps_per_beat"]
    if spb % 4 or spb % 3:
        raise ValueError(f"steps_per_beat {spb} has no 16th and triplet steps")
    onsets = np.asarray(grid["onsets"], dtype=bool)
    beats = record["end_beat"] - record["start_beat"]
    seconds = beats * 60 / record["tempo_bpm"]
    occupied = np.flatnonzero(onsets.any(axis=0))
    off = occupied[occupied % spb != 0] % spb
    straight = int(np.sum(off % (spb // 4) == 0))
    triplet = int(np.sum((off % (spb // 3) == 0) & (off % (spb // 4) != 0)))
    gaps = np.diff(occupied)
    return {
        "beats": beats,
        "bar_beats": beats / spec.loops.bars,
        "hits": {v: np.flatnonzero(onsets[k]).tolist() for k, v in enumerate(grid["voices"])},
        "sounds": len(occupied),
        "sounds_per_s": round(len(occupied) / seconds, 2),
        "min_gap_ms": round(float(gaps.min()) * 60_000 / record["tempo_bpm"] / spb)
        if len(gaps)
        else None,
        "straight_off": straight,
        "triplet_off": triplet,
        "other_off": len(off) - straight - triplet,
    }


def render_loop(record: dict, spec: MusicSpec, kit: Kit, sampler: Sampler) -> np.ndarray:
    """The loop PLAYS times back to back, with the spec's lead-in and cut."""
    from src.library.render_preview import audio_events  # B's renderer for catalog loops

    once = audio_events(record, spec)
    loop_s = (record["end_beat"] - record["start_beat"]) * 60 / record["tempo_bpm"]
    events = [(t + k * loop_s, p, v) for k in range(PLAYS) for t, p, v in once]
    duration = spec.loops.audio.lead_in_s + PLAYS * loop_s
    raw = sampler.render(kit_hits(kit, events))
    frames = round(duration * sampler.sample_rate)
    raw = np.pad(raw, ((0, max(0, frames - len(raw))), (0, 0)))
    return frame_audio(
        normalize(raw),
        sample_rate=sampler.sample_rate,
        duration_s=duration,
        settings=spec.loops.audio,
    )


def load_catalog(catalog_dir: Path, spec: MusicSpec, gmd_dir: Path = GMD_DIR) -> list[dict]:
    """Test-split catalog records, rebuilding the catalog first if its spec is stale."""
    summary = catalog_dir / "summary.json"
    if not summary.exists() or json.loads(summary.read_text())["spec_version"] != spec.version:
        from src.library.catalog_gmd import catalog_gmd

        catalog_gmd(gmd_dir, catalog_dir)
    lines = (catalog_dir / "loops.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["split"] == SPLIT]


def build(
    records: list[dict], spec: MusicSpec, kit: Kit, out: Path, *, audio: bool = True, seed: int = 0
) -> dict:
    """Stats for every record and, with audio=True, one MP3 each under out/audio/."""
    repeats = Counter(r["pattern_id"] for r in records)
    sampler = Sampler(load_program(kit), seed=seed) if audio else None
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(records):
        name = f"audio/{i:04d}.mp3"
        if sampler is not None:
            clip = render_loop(r, spec, kit, sampler)
            sf.write(out / name, clip, sampler.sample_rate, format="MP3", subtype="MPEG_LAYER_III")
        rows.append(
            {
                "i": i,
                "clip_id": r["clip_id"],
                "pattern_id": r["pattern_id"],
                "perf": r["source_performance"],
                "drummer": r["source_performance"].split("/")[0],
                "style": style_of(r["midi_path"]),
                "start_bar": r["start_bar"],
                "bpm": r["tempo_bpm"],
                **loop_stats(r, spec),
                "same_pattern_clips": repeats[r["pattern_id"]],
                "audio": name,
            }
        )
    meta = {
        "spec_version": spec.version,
        "kit": kit.kit_id,
        "split": SPLIT,
        "plays": PLAYS,
        "voices": list(spec.voices),
        "steps_per_beat": spec.same_rhythm.steps_per_beat,
        "clips": len(rows),
        "patterns": len(repeats),
    }
    return {"meta": meta, "rows": rows}


def write_page(data: dict, out: Path) -> Path:
    page = out / "index.html"
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page.write_text(PAGE_TEMPLATE.read_text().replace("/*DATA*/{}", payload))
    return page


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--kit", default=DEFAULT_KIT, help="kit_id from rendering.kits")
    ap.add_argument("--catalog", type=Path, default=CATALOG_DIR)
    ap.add_argument("--gmd", type=Path, default=GMD_DIR)
    ap.add_argument("--no-audio", action="store_true", help="stats and page only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    spec = load_spec()
    kit = kit_by_id(spec, args.kit)
    records = load_catalog(args.catalog, spec, args.gmd)
    data = build(records, spec, kit, args.out, audio=not args.no_audio, seed=args.seed)
    page = write_page(data, args.out)
    m = data["meta"]
    print(f"{m['clips']} {SPLIT} loops ({m['patterns']} patterns) through {kit.kit_id}")
    print(f"Open it with: open {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

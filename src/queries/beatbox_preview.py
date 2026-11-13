"""Build clean beatbox/drum listening pairs before full query augmentation.

Run from the IDE or ``python -m src.queries.beatbox_preview``. Inputs are the
labeled GMD catalog, extracted AVP sounds, and rendered drum preview manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.audio import frame_audio
from src.music.rhythm import Note, _merge_flams, _nearest_step, clip_grid, pattern_id
from src.music.spec import MusicSpec, load_spec
from src.queries.mixing import mix_placements
from tools.render.kits import normalize, write_wav

REPO_ROOT = Path(__file__).resolve().parents[2]
# AVP's dataset class codes, not extra musical thresholds.
VOICE_LABEL = {"kick": "kd", "snare": "sd"}


def limit_event_rate(events: list[dict], *, bpm: float, spec: MusicSpec) -> list[dict]:
    """Greedy priority order, using each candidate voice's clean-time gap.

    Compare against every already-kept event, including those later in musical
    time. Return chronological order for rendering. Labels remain untouched.
    """
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("Positive finite BPM required")
    priority = {voice: i for i, voice in enumerate(spec.beatbox.simultaneous_priority)}
    kept = []
    for event in sorted(events, key=lambda e: (priority[e["voice"]], e["beat"])):
        gap = spec.beatbox.min_gap_ms[event["voice"]]
        if all(abs(event["beat"] - other["beat"]) * 60000 / bpm + 1e-9 >= gap for other in kept):
            kept.append(event)
    return sorted(kept, key=lambda e: e["beat"])


def beatbox_events(record: dict, *, spec: MusicSpec, seed: int) -> tuple[list[dict], dict]:
    """Simplify the target grid, retaining provenance for the winning drum hit.

    Use shared rhythm helpers for flam representatives and midpoint rounding;
    do not implement a second label definition. Full straight-sixteenth hat beats
    are optionally reduced to their eighth positions, once per complete beat.
    Simultaneous priority is applied after thinning. Target labels stay unchanged.
    """
    notes = [Note(**n) for n in record["notes"]]
    bpm = record["tempo_bpm"]
    grid = clip_grid(notes, bpm=bpm, spec=spec)
    if pattern_id(grid) != record["pattern_id"]:
        raise ValueError("Source label disagrees with current spec")
    candidates = {}
    for voice in spec.voices:
        source = sorted(
            [
                n
                for n in notes
                if spec.voice_of(n.pitch) == voice and n.velocity >= spec.ghost_threshold(voice)
            ],
            key=lambda n: (n.beat, n.pitch),
        )
        for beat in _merge_flams(source, bpm, spec):
            step = _nearest_step(beat, spec)
            if not 0 <= step < grid.n_steps:
                continue
            representative = max((n for n in source if n.beat == beat), key=lambda n: n.velocity)
            key = (voice, step)
            previous = candidates.get(key)
            if previous is None or (-representative.velocity, representative.beat) < (
                -previous.velocity,
                previous.beat,
            ):
                candidates[key] = representative
    spb = grid.steps_per_beat
    rng = np.random.default_rng(seed)
    removed_hats = set()
    probability = spec.beatbox.thin_16th_hats_to_8ths_prob
    if probability and spb % 4:
        raise ValueError("Hat thinning requires a grid representing straight sixteenths")
    if probability:
        for beat in range(grid.beats):
            positions = [beat * spb + i * (spb // 4) for i in range(4)]
            if (
                all(("hats", step) in candidates for step in positions)
                and rng.random() < probability
            ):
                removed_hats.update(positions[1::2])
    events = []
    suppressed = 0
    for step in range(grid.n_steps):
        eligible = []
        for voice in spec.beatbox.simultaneous_priority:
            note = candidates.get((voice, step))
            if note is None or (voice == "hats" and step in removed_hats):
                continue
            if note.velocity < spec.beatbox.skip_velocity_below[voice]:
                continue
            eligible.append((voice, note))
        if not eligible:
            continue
        suppressed += len(eligible) - 1
        voice, note = eligible[0]
        if voice == "hats":
            label = (
                spec.beatbox.open_hat_sound
                if note.pitch in spec.drum_map["hihat_open"].notes
                else spec.beatbox.hats_sound
            )
        else:
            label = VOICE_LABEL[voice]
        if label not in spec.beatbox.avp_labels:
            raise ValueError(f"AVP label {label} not allowed by spec")
        events.append(
            {
                "step": step,
                "beat": step / spb,
                "voice": voice,
                "label": label,
                "source_pitch": note.pitch,
                "source_velocity": note.velocity,
            }
        )
    before_limit = len(events)
    events = limit_event_rate(events, bpm=bpm, spec=spec)
    if not events:
        raise ValueError("Simplification produced no beatbox events")
    return events, {
        "thinned_hat_hits": len(removed_hats),
        "suppressed_simultaneous": suppressed,
        "speed_limited_hits": before_limit - len(events),
    }


def mix_query(
    events: list[dict],
    samples: pd.DataFrame,
    sample_root: Path,
    *,
    participant: str,
    mode: str,
    bpm: float,
    pre_roll_s: float,
    duration_s: float,
    seed: int,
    normalize_output: bool = True,
) -> tuple[np.ndarray, int, list[dict]]:
    """Place source samples at event times without extra timing/noise augmentation.

    One participant and one mode per query; sample rates must agree. Stereo
    sources are averaged to mono. Choke before framing/normalization, using
    actual clean placement times; augmentation performs its own choke after drops.
    """
    if not events or not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("Need events and a positive finite tempo")
    if pre_roll_s < 0 or duration_s <= 0:
        raise ValueError("Invalid pre-roll or duration")
    pool = samples[(samples.participant_id == participant) & (samples["mode"] == mode)]
    rng = np.random.default_rng(seed)
    placements, trace, rate = [], [], None
    for event in events:
        available = pool[pool.label == event["label"]].sort_values("utterance_id")
        if available.empty:
            raise ValueError(f"No {event['label']} samples for {participant}/{mode}")
        selected = available.iloc[int(rng.integers(len(available)))]
        signal, sr = sf.read(sample_root / selected.sample_path, dtype="float32", always_2d=True)
        if not np.isfinite(signal).all() or not np.any(signal):
            raise ValueError(f"Unusable sample: {selected.utterance_id}")
        if rate is not None and sr != rate:
            raise ValueError("Mixed sample rates require explicit resampling; not implemented here")
        rate = sr
        start = round((pre_roll_s + event["beat"] * 60 / bpm) * sr)
        placements.append((start, signal.mean(axis=1)))
        trace.append(
            {
                **event,
                "utterance_id": selected.utterance_id,
                "sample_path": selected.sample_path,
                "participant_id": participant,
                "mode": mode,
                "frame": start,
                "sample_rate": sr,
            }
        )
    frames = max(round(duration_s * rate), max(start + len(x) for start, x in placements))
    spec = load_spec()
    if spec.loops.audio.end == "cut":
        frames = round(duration_s * rate)
    audio, details = mix_placements(
        placements, frames=frames, sample_rate=rate, settings=spec.beatbox
    )
    for event, detail in zip(trace, details, strict=True):
        event.update(detail)
    audio = frame_audio(audio, sample_rate=rate, duration_s=duration_s, settings=spec.loops.audio)
    return (normalize(audio) if normalize_output else audio), rate, trace


def build_pairs(
    library_dir: Path, catalog_dir: Path, sample_root: Path, out_dir: Path, *, seed: int = 0
) -> dict:
    """Pair each existing drum preview with both approved AVP modes for audition."""
    spec = load_spec()
    for folder in (library_dir, catalog_dir, sample_root):
        summary = json.loads((folder / "summary.json").read_text())
        if summary["spec_version"] != spec.version:
            raise ValueError(f"Stale spec version in {folder}")
    library = pd.read_csv(library_dir / "library_manifest.csv")
    samples = pd.read_csv(sample_root / "sample_catalog.csv")
    category = np.where(samples.kind == "Improvisation", "improvisation", "isolated")
    samples = samples[
        samples["mode"].str.lower().isin(spec.beatbox.avp_modes)
        & pd.Series(category, index=samples.index).isin(spec.beatbox.avp_recordings)
        & samples.label.isin(spec.beatbox.avp_labels)
    ]
    required = set(spec.beatbox.avp_labels)
    modes = [m.capitalize() for m in spec.beatbox.avp_modes]
    people = []
    for person in sorted(samples.participant_id.unique(), key=lambda p: int(p[1:])):
        if all(
            required
            <= set(samples[(samples.participant_id == person) & (samples["mode"] == mode)].label)
            for mode in modes
        ):
            people.append(person)
    splits = sorted(library.split.unique())
    if len(people) < len(splits):
        raise ValueError("Not enough complete participants for disjoint preview allocation")
    allocation = dict(zip(splits, people, strict=False))
    needed = set(zip(library.source_performance, library.start_bar, strict=True))
    records = {}
    with (catalog_dir / "loops.jsonl").open() as source:
        for line in source:
            record = json.loads(line)
            key = (record["source_performance"], record["start_bar"])
            if key in needed:
                records[key] = record
    if set(records) != needed:
        raise ValueError("Some drum previews have no matching catalog window")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries, copied_library, traces, sections = [], [], [], []
    for loop in library.itertuples(index=False):
        record = records[(loop.source_performance, loop.start_bar)]
        if (
            loop.pattern_id != record["pattern_id"]
            or loop.split != record["split"]
            or loop.dualized_pattern_id != record["dualized_pattern_id"]
        ):
            raise ValueError("Library/catalog pattern ID or split mismatch")
        source_audio = Path(loop.audio_path)
        if not source_audio.is_absolute():
            source_audio = REPO_ROOT / source_audio
        drum_relative = Path("drums") / source_audio.name
        (out_dir / "drums").mkdir(exist_ok=True)
        shutil.copyfile(source_audio, out_dir / drum_relative)
        copy = loop._asdict()
        copy["audio_path"] = str(out_dir / drum_relative)
        copied_library.append(copy)
        players = []
        for mode in modes:
            person = allocation[loop.split]
            identity = f"{loop.clip_id}:{person}:{mode}:clean:seed{seed}"
            digest = hashlib.sha256(identity.encode()).hexdigest()
            query_seed = int(digest[:8], 16)
            events, reductions = beatbox_events(record, spec=spec, seed=query_seed)
            audio, sr, trace = mix_query(
                events,
                samples,
                sample_root,
                participant=person,
                mode=mode,
                bpm=loop.tempo_bpm,
                pre_roll_s=loop.pre_roll_s,
                duration_s=loop.pre_roll_s + loop.pattern_duration_s,
                seed=query_seed,
            )
            relative = Path("queries") / f"beatbox_{digest[:16]}.wav"
            write_wav(out_dir / relative, audio, sr)
            query_id = f"beatbox_{digest[:16]}"
            queries.append(
                {
                    "query_id": query_id,
                    "pattern_id": loop.pattern_id,
                    "dualized_pattern_id": loop.dualized_pattern_id,
                    "query_type": "beatbox",
                    "source": "synthetic",
                    "participant_id": person,
                    "split": loop.split,
                    "tempo_bpm": loop.tempo_bpm,
                    "jitter_ms": 0,
                    "drift_pct": 0,
                    "drop_prob": 0,
                    "snr_db": None,
                    "audio_path": str(out_dir / relative),
                    "mode": mode,
                    "offset_ms": 0,
                    "target_clip_id": loop.clip_id,
                    "seed": query_seed,
                    "sample_rate": sr,
                    "duration_s": len(audio) / sr,
                    "pre_roll_s": loop.pre_roll_s,
                    "events": len(events),
                    **reductions,
                    "status": "audition_only_clean",
                }
            )
            traces.append(
                {
                    "query_id": query_id,
                    "target_clip_id": loop.clip_id,
                    "pattern_id": loop.pattern_id,
                    "events": trace,
                    **reductions,
                }
            )
            players.append(
                f"<p>{html.escape(person + ' · ' + mode)}</p>"
                f'<audio controls preload="none" src="{relative.as_posix()}"></audio>'
            )
        sections.append(
            f"<section><h2>{html.escape(loop.source_performance)} · bar {loop.start_bar}</h2>"
            f"<p>{loop.tempo_bpm:g} BPM · source split {html.escape(loop.split)} · "
            f"pattern <code>{loop.pattern_id}</code></p><h3>Drum loop</h3>"
            f'<audio controls preload="none" src="{drum_relative.as_posix()}"></audio>'
            "<h3>Synthetic beatbox</h3>" + "".join(players) + "</section>"
        )
    pd.DataFrame(queries).to_csv(out_dir / "query_manifest.csv", index=False)
    pd.DataFrame(copied_library).to_csv(out_dir / "library_manifest.csv", index=False)
    with (out_dir / "events.jsonl").open("w") as output:
        for trace in traces:
            output.write(json.dumps(trace) + "\n")
    review = pd.DataFrame(queries)[["query_id", "target_clip_id", "participant_id", "mode"]]
    review_path = out_dir / "listening_notes.csv"
    if review_path.exists():
        previous = pd.read_csv(review_path, keep_default_na=False)
        review = review.merge(
            previous[["query_id", "verdict", "notes"]],
            on="query_id",
            how="left",
            validate="one_to_one",
        ).fillna("")
    else:
        review = review.assign(verdict="", notes="")
    review.to_csv(review_path, index=False)
    summary = {
        "spec_version": spec.version,
        "drum_loops": len(library),
        "queries": len(queries),
        "preview_participant_by_source_split": allocation,
        "seed": seed,
        "status": "clean audition pairs; not the v0 training dataset",
        "augmentation": "no offset, jitter, drift, random drops, or added noise",
        "normalization": "existing audition RMS/peak normalization",
        "target_policy": "source pattern ID retained after beatbox simplification",
        "timing": "shared rhythm-grid positions plus drum preview pre-roll",
        "hat_thinning": "one seeded draw per full four-sixteenth hat beat; drop off-eighths",
        "participant_policy": "distinct preview participant per source split; not final 20/3/5",
        "sample_directory": str(sample_root.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "preview.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Beatbox and drum pairs</title><style>"
        "body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5}"
        "section{border-top:1px solid #ddd;padding:20px 0}h2{font-size:19px}h3{font-size:16px}"
        "</style><h1>Beatbox and matching drum loops</h1>"
        "<p>Play the drum loop, then either beatbox version. They share a target rhythm; "
        "the voice performs a simplified version. These are clean, grid-timed previews "
        "with a shared short lead-in, before added performance imperfections.</p>"
        "<p>Listen for recognizable rhythm, sensible sound selection, and natural sample joins. "
        "This page and its audio folders can be shared together for listening.</p>"
        '<p><a href="query_manifest.csv">Queries</a> · '
        '<a href="listening_notes.csv">Listening notes</a> · '
        '<a href="summary.json">Summary</a></p>' + "".join(sections) + "</html>"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO_ROOT / "results/library_preview")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "results/gmd_catalog")
    parser.add_argument("--samples", type=Path, default=REPO_ROOT / "results/avp_samples")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/beatbox_pairs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_pairs(args.library, args.catalog, args.samples, args.out, seed=args.seed),
            indent=2,
        )
    )
    print(f"Listen: {(args.out / 'preview.html').resolve()}")


if __name__ == "__main__":
    main()

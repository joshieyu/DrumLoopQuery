"""One seeded command to regenerate a small, split-valid v0.5 review dataset.

    python -m src.build_checkpoint --seed 0

Uses installed kits only. Default train/validation batch: 10 drum loops and
20 noisy beatbox queries, plus clean references. Not the full dataset freeze.
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
import yaml

from src.audio import frame_audio, normalize_audio
from src.library.catalog_gmd import catalog_gmd
from src.library.render_preview import audio_events
from src.music.spec import load_spec
from src.queries.augment import augment_query
from src.queries.beatbox_preview import beatbox_events, mix_query
from src.queries.extract_avp import extract_samples
from src.splits import assign_participants
from tools.render.kits import check_kit, kit_hits, load_program, write_wav
from tools.render.sampler import Sampler

ROOT = Path(__file__).resolve().parents[1]


def audio_path(path: Path) -> str:
    """Repository-relative for normal builds; absolute for external review folders."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def stable_seed(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:8], 16)


def build_checkpoint(
    *,
    seed: int = 0,
    splits: tuple[str, ...] = ("train", "validation"),
    out_dir: Path = ROOT / "results/v05_checkpoint",
) -> dict:
    spec = load_spec()
    config = yaml.safe_load((ROOT / "config/dataset_v0.yaml").read_text())
    normalization = config["normalization"]
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir = ROOT / "results/gmd_catalog"
    catalog_summary = catalog_gmd(ROOT / "data/raw/gmd/groove", catalog_dir)
    if catalog_summary["failed_performances"]:
        raise ValueError("Catalog has failed sources")
    sample_root = ROOT / "results/avp_samples"
    summary_path = sample_root / "summary.json"
    if (
        not summary_path.exists()
        or json.loads(summary_path.read_text())["spec_version"] != spec.version
    ):
        extract_samples(ROOT / "data/raw/avp/AVP_Dataset", sample_root)
    samples = pd.read_csv(sample_root / "sample_catalog.csv")
    category = np.where(samples.kind == "Improvisation", "improvisation", "isolated")
    samples = samples[
        samples["mode"].str.lower().isin(spec.beatbox.avp_modes)
        & pd.Series(category, index=samples.index).isin(spec.beatbox.avp_recordings)
        & samples.label.isin(spec.beatbox.avp_labels)
    ]
    assignment = assign_participants(
        samples.participant_id.tolist(), counts=config["participant_counts"], seed=seed
    )
    assignment_path = ROOT / "manifests/avp_participant_splits.csv"
    if assignment_path.exists():
        previous = pd.read_csv(assignment_path)
        if not previous.equals(assignment):
            raise ValueError(
                "Saved participant assignment differs; choose a separate dataset before reseeding"
            )
    assignment.to_csv(assignment_path, index=False)
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    assignment.to_csv(manifest_dir / assignment_path.name, index=False)
    samplers = {}
    for split in splits:
        kit_split = "test" if split == "test" else "train"
        available = [
            k
            for k in spec.rendering.kits
            if k.split == kit_split and (ROOT / "data/raw/kits" / k.program).exists()
        ]
        if not available:
            raise ValueError(
                f"No installed {kit_split} kit for {split}; download an approved kit first"
            )
        kit = available[0]
        program = load_program(kit)
        problems = check_kit(kit, program, spec)
        if problems:
            raise ValueError(str(problems))
        samplers[split] = (kit, Sampler(program, seed=seed))
    selected, seen, counts = [], set(), {}
    for line in (catalog_dir / "loops.jsonl").read_text().splitlines():
        r = json.loads(line)
        split = r["split"]
        if split not in splits or r["source_performance"] in seen:
            continue
        if counts.get(split, 0) >= config["review_patterns"][split]:
            continue
        seen.add(r["source_performance"])
        counts[split] = counts.get(split, 0) + 1
        selected.append(r)
    if any(counts.get(s, 0) != config["review_patterns"][s] for s in splits):
        raise ValueError("Not enough distinct performances for review batch")
    library_rows, query_rows, event_rows, sections, active_audio = [], [], [], [], []
    modes = [m.capitalize() for m in spec.beatbox.avp_modes]
    for r in selected:
        identity = hashlib.sha256(r["clip_id"].encode()).hexdigest()[:16]
        kit, sampler = samplers[r["split"]]
        duration = (
            spec.loops.audio.lead_in_s + (r["end_beat"] - r["start_beat"]) * 60 / r["tempo_bpm"]
        )
        raw = sampler.render(kit_hits(kit, audio_events(r, spec)))
        # Cut before level measurement, then apply final fade after gain.
        raw = raw[: round(duration * sampler.sample_rate)]
        raw = normalize_audio(raw, **normalization)
        audio = frame_audio(
            raw, sample_rate=sampler.sample_rate, duration_s=duration, settings=spec.loops.audio
        )
        drum_relative = Path("audio/library") / f"{identity}_{kit.kit_id}.wav"
        write_wav(out_dir / drum_relative, audio, sampler.sample_rate)
        active_audio.append(drum_relative)
        clip_id = f"v05:{r['clip_id']}:{kit.kit_id}:recorded:seed{seed}"
        library_rows.append(
            {
                **{
                    key: r[key]
                    for key in (
                        "pattern_id",
                        "dualized_pattern_id",
                        "split",
                        "tempo_bpm",
                        "midi_path",
                        "source_performance",
                        "start_bar",
                    )
                },
                "clip_id": clip_id,
                "kit_id": kit.kit_id,
                "kit_split": kit.split,
                "audio_path": audio_path(out_dir / drum_relative),
                "duration_s": len(audio) / sampler.sample_rate,
                "lead_in_s": spec.loops.audio.lead_in_s,
                "sample_rate": sampler.sample_rate,
                "seed": seed,
            }
        )
        eligible_people = sorted(assignment[assignment.split == r["split"]].participant_id)
        person = eligible_people[stable_seed(identity + "person", seed) % len(eligible_people)]
        players = []
        for mode in modes:
            query_seed = stable_seed(f"{identity}:{person}:{mode}", seed)
            events, reductions = beatbox_events(r, spec=spec, seed=query_seed)
            clean, sr, trace = mix_query(
                events,
                samples,
                sample_root,
                participant=person,
                mode=mode,
                bpm=r["tempo_bpm"],
                pre_roll_s=spec.loops.audio.lead_in_s,
                duration_s=duration,
                seed=query_seed,
                normalize_output=False,
            )
            clean = normalize_audio(clean, **normalization)
            sloppy, params, actual = augment_query(
                trace,
                sample_root,
                sample_rate=sr,
                duration_s=duration,
                spec=spec,
                seed=query_seed,
                normalization=normalization,
            )
            query_id = f"v05_{identity}_{person}_{mode}_seed{seed}"
            query_relative = Path("audio/queries") / f"{query_id}.wav"
            clean_relative = Path("audio/clean") / f"{query_id}.wav"
            write_wav(out_dir / query_relative, sloppy, sr)
            write_wav(out_dir / clean_relative, clean, sr)
            active_audio.extend([query_relative, clean_relative])
            query_rows.append(
                {
                    "query_id": query_id,
                    "pattern_id": r["pattern_id"],
                    "dualized_pattern_id": r["dualized_pattern_id"],
                    "query_type": "beatbox",
                    "source": "synthetic",
                    "participant_id": person,
                    "mode": mode,
                    "split": r["split"],
                    "tempo_bpm": r["tempo_bpm"],
                    **{k: v for k, v in params.items() if k != "drop_prob_by_voice"},
                    "drop_prob_by_voice": json.dumps(params["drop_prob_by_voice"], sort_keys=True),
                    "audio_path": audio_path(out_dir / query_relative),
                    "target_clip_id": clip_id,
                    "duration_s": len(sloppy) / sr,
                    "sample_rate": sr,
                    "seed": query_seed,
                    **reductions,
                }
            )
            event_rows.append({"query_id": query_id, "parameters": params, "events": actual})
            real = samples[
                (samples.participant_id == person)
                & (samples["mode"] == mode)
                & (samples.kind == "Improvisation")
            ]
            real_player = ""
            if not real.empty:
                reference = Path("references") / f"{person}_{mode}_Improvisation.wav"
                source_root = Path(json.loads(summary_path.read_text())["source_root"])
                (out_dir / "references").mkdir(exist_ok=True)
                if not (out_dir / reference).exists():
                    shutil.copyfile(
                        source_root / real.iloc[0].source_audio_path, out_dir / reference
                    )
                real_player = (
                    f"<p>Real AVP improvisation (same participant/mode, different rhythm)</p>"
                    f'<audio controls preload="none" src="{reference.as_posix()}"></audio>'
                )
            players.append(
                f"<h3>{person} · {mode}</h3><p>Clean reference</p>"
                f'<audio controls preload="none" src="{clean_relative.as_posix()}"></audio>'
                f"<p>Sloppy query: offset {params['offset_ms']:.1f} ms, "
                f"jitter ±{params['jitter_ms']:.1f} ms, {params['dropped_hits']} dropped hits, "
                f"SNR {params['snr_db']:.1f} dB</p>"
                f'<audio controls preload="none" src="{query_relative.as_posix()}"></audio>'
                + real_player
            )
        sections.append(
            f"<section><h2>{html.escape(r['source_performance'])} · "
            f"{r['split']} · {r['tempo_bpm']:g} BPM</h2><p>Drum target</p>"
            f'<audio controls preload="none" src="{drum_relative.as_posix()}"></audio>'
            + "".join(players)
            + "</section>"
        )
    library = pd.DataFrame(library_rows)
    queries = pd.DataFrame(query_rows)
    library.to_csv(manifest_dir / "library_manifest.csv", index=False)
    queries.to_csv(manifest_dir / "query_manifest.csv", index=False)
    review = queries[["query_id", "participant_id", "mode", "split"]].copy()
    review_path = out_dir / "listening_notes.csv"
    if review_path.exists():
        review = review.merge(
            pd.read_csv(review_path, keep_default_na=False)[["query_id", "verdict", "notes"]],
            on="query_id",
            how="left",
        ).fillna("")
    else:
        review = review.assign(verdict="", notes="")
    review.to_csv(review_path, index=False)
    with (out_dir / "events.jsonl").open("w") as output:
        for row in event_rows:
            output.write(json.dumps(row) + "\n")
    summary = {
        "spec_version": spec.version,
        "seed": seed,
        "drum_loops": len(library),
        "sloppy_queries": len(queries),
        "clean_references": len(queries),
        "participant_counts": assignment.split.value_counts().to_dict(),
        "represented_splits": list(splits),
        "kits": library.kit_id.unique().tolist(),
        "audio": {
            "lead_in_s": spec.loops.audio.lead_in_s,
            "end": spec.loops.audio.end,
            "end_fade_ms": spec.loops.audio.end_fade_ms,
        },
        "normalization": normalization,
        "catalog": catalog_summary,
        "status": "review checkpoint; not full dataset or musical sign-off",
        "overlap_policy": "preserve extracted sample tails internally; spec cut at clip end",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "build_settings.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "splits": list(splits),
                "config": config,
                "music_spec_sha256": hashlib.sha256(
                    (ROOT / "config/music_spec.yaml").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    hashes = {
        p.as_posix(): hashlib.sha256((out_dir / p).read_bytes()).hexdigest()
        for p in sorted(active_audio)
    }
    (out_dir / "audio_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    sources = ROOT / "data/raw/SOURCES.md"
    if sources.exists():
        shutil.copyfile(sources, out_dir / "SOURCES.md")
    (out_dir / "preview.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>v0.5 sloppy query review</title><style>"
        "body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5}"
        "section{border-top:1px solid #ddd;padding:20px 0}h2{font-size:19px}h3{font-size:16px}"
        "</style><h1>v0.5 query review</h1><p>Compare each drum target with clean and sloppy "
        "beatbox versions. The source mouth sounds are identical between each clean/sloppy pair. "
        "Offset, jitter, drops, and noise are seeded. No tempo drift. "
        "Every clip uses the approved lead-in and ending fade.</p>"
        "<p>This small train/validation batch uses the installed training kit. "
        "It is not the complete seven-kit dataset. "
        '<a href="listening_notes.csv">Review notes</a> · '
        '<a href="summary.json">Build summary</a></p>' + "".join(sections) + "</html>"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config = yaml.safe_load((ROOT / "config/dataset_v0.yaml").read_text())
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--out", type=Path, default=ROOT / "results/v05_checkpoint")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation"],
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_checkpoint(seed=args.seed, splits=tuple(args.splits), out_dir=args.out), indent=2
        )
    )


if __name__ == "__main__":
    main()

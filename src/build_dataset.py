"""Build all retained GMD loops and two seeded AVP queries per source loop.

    python -m src.build_dataset --seed 0

Audio and resumable receipts stay under data/datasets/v07. Publish manifests
only after the complete build and tools.check_model_loader have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from src.audio import frame_audio, normalize_audio
from src.build_checkpoint import ROOT, audio_path, stable_seed
from src.library.catalog_gmd import catalog_gmd
from src.library.render_preview import audio_events
from src.music.spec import load_spec
from src.queries.augment import augment_query
from src.queries.beatbox_preview import beatbox_events, mix_query
from src.queries.extract_avp import extract_samples
from src.splits import assign_participants
from tools.render.kits import check_kit, kit_hits, load_program, write_wav
from tools.render.sampler import Sampler


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_manifests(library, queries, assignments, spec) -> None:
    """Validate B's handoff contract before publishing any manifests."""
    if library.empty or queries.empty:
        raise ValueError("Empty dataset")
    if library.clip_id.duplicated().any() or queries.query_id.duplicated().any():
        raise ValueError("Duplicate clip/query IDs")
    combined = pd.concat([library, queries])
    if (combined.groupby("pattern_id").split.nunique() > 1).any():
        raise ValueError("Pattern leakage between splits")
    kits = {k.kit_id: k.split for k in spec.rendering.kits}
    for row in library.itertuples(index=False):
        if kits.get(row.kit_id) != ("test" if row.split == "test" else "train"):
            raise ValueError("Kit leakage between splits")
    assigned = dict(zip(assignments.participant_id, assignments.split, strict=True))
    if any(assigned.get(r.participant_id) != r.split for r in queries.itertuples()):
        raise ValueError("Participant leakage between splits")
    targets = set(zip(library.pattern_id, library.split, strict=True))
    if any((r.pattern_id, r.split) not in targets for r in queries.itertuples()):
        raise ValueError("Query without a relevant library target")


class Receipts:
    """Append completed outputs; resume only when recipe and file hashes match."""

    def __init__(self, directory: Path, kind: str):
        self.path = directory / f"{kind}_receipts.jsonl"
        self.rows = {}
        if self.path.exists():
            with self.path.open("r+b") as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        if stream.read():
                            raise
                        stream.truncate(position)  # interrupted final receipt; regenerate it
                        break
                    self.rows[item["id"]] = item

    def find(self, identity: str) -> dict | None:
        item = self.rows.get(identity)
        if item is None:
            return None
        path = ROOT / item["row"]["audio_path"]
        if not path.exists() or digest(path) != item["sha256"]:
            return None
        return item

    def add(self, identity: str, row: dict, **extra) -> dict:
        item = {
            "id": identity,
            "row": row,
            "sha256": digest(ROOT / row["audio_path"]),
            **extra,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
            stream.flush()
        self.rows[identity] = item
        return item


def build_dataset(*, seed=0, out_dir=None, limit_per_split=None) -> dict:
    spec = load_spec()
    tag = "v" + spec.version.replace(".", "")
    out = (out_dir or ROOT / f"data/datasets/{tag}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "config/dataset_v0.yaml").read_text())
    catalog_dir = ROOT / "results/gmd_catalog"
    print("Preparing current-spec MIDI catalog", flush=True)
    catalog = catalog_gmd(ROOT / "data/raw/gmd/groove", catalog_dir)
    if catalog["failed_performances"]:
        raise ValueError("Catalog has failed source performances")
    records = [json.loads(line) for line in (catalog_dir / "loops.jsonl").read_text().splitlines()]
    if limit_per_split is not None:
        if limit_per_split < 1:
            raise ValueError("limit-per-split must be positive")
        counts = Counter()
        selected = []
        for r in records:
            if counts[r["split"]] < limit_per_split:
                counts[r["split"]] += 1
                selected.append(r)
        records = selected
    sample_root = ROOT / "results/avp_samples"
    sample_summary = sample_root / "summary.json"
    if (
        not sample_summary.exists()
        or json.loads(sample_summary.read_text())["spec_version"] != spec.version
    ):
        extract_samples(ROOT / "data/raw/avp/AVP_Dataset", sample_root)
    samples = pd.read_csv(sample_root / "sample_catalog.csv")
    category = samples.kind.map(lambda k: "improvisation" if k == "Improvisation" else "isolated")
    samples = samples[
        samples["mode"].str.lower().isin(spec.beatbox.avp_modes)
        & category.isin(spec.beatbox.avp_recordings)
        & samples.label.isin(spec.beatbox.avp_labels)
    ]
    assignments = assign_participants(
        samples.participant_id.tolist(), counts=config["participant_counts"], seed=seed
    )
    saved_assignment = pd.read_csv(ROOT / "manifests/avp_participant_splits.csv")
    if not saved_assignment.equals(assignments):
        raise ValueError("Seed would change the saved participant partition")
    kits = [k for k in spec.rendering.kits if k.split in ("train", "test")]
    # Fail before rendering if any approved kit is absent/unsupported.
    audio_inputs = {sample_root / p for p in samples.sample_path}
    for kit in kits:
        program = load_program(kit)
        audio_inputs.update(Path(r.sample) for r in program.regions if not r.silent)
        if problems := check_kit(kit, program, spec):
            raise ValueError(f"{kit.kit_id}: {problems}")
        Sampler(program, seed=seed)
    code = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tools/render").glob("*.py")),
    ]
    print(f"Fingerprinting {len(audio_inputs)} source samples", flush=True)
    source_hash = hashlib.sha256()
    for path in sorted(audio_inputs):
        source_hash.update(f"{path.relative_to(ROOT)}:{digest(path)}\n".encode())
    recipe = {
        "source_audio_sha256": source_hash.hexdigest(),
        "seed": seed,
        "spec_version": spec.version,
        "limit_per_split": limit_per_split,
        "query_views": "one participant per source loop, both approved AVP modes",
        "config": config,
        "inputs": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "config/music_spec.yaml",
                catalog_dir / "loops.jsonl",
                sample_root / "sample_catalog.csv",
                ROOT / "manifests/avp_participant_splits.csv",
                *code,
            ]
        },
        "kit_programs": {k.kit_id: digest(ROOT / "data/raw/kits" / k.program) for k in kits},
    }
    recipe_path = out / "build_settings.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError(
            "Build recipe changed. Use a new --out directory; existing audio is preserved."
        )
    atomic_json(recipe_path, recipe)
    normal = config["normalization"]
    library, queries = [], []
    lib_receipts, query_receipts = Receipts(out, "library"), Receipts(out, "query")

    def duration(r):
        return spec.loops.audio.lead_in_s + (r["end_beat"] - r["start_beat"]) * 60 / r["tempo_bpm"]

    total = sum(
        sum(
            r["split"] in (("test",) if k.split == "test" else ("train", "validation"))
            for r in records
        )
        for k in kits
    )
    for kit in kits:
        print(f"Rendering library kit {kit.kit_id}", flush=True)
        sampler = Sampler(load_program(kit), seed=seed)
        for r in records:
            if (r["split"] == "test") != (kit.split == "test"):
                continue
            identity = f"{tag}:{r['clip_id']}:{kit.kit_id}:recorded:seed{seed}"
            previous = lib_receipts.find(identity)
            if previous:
                library.append(previous["row"])
                continue
            name = hashlib.sha256(identity.encode()).hexdigest()[:24]
            path = out / "audio/library" / kit.kit_id / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = sampler.render(kit_hits(kit, audio_events(r, spec)), length_s=duration(r))
            audio = frame_audio(
                normalize_audio(raw, **normal),
                sample_rate=sampler.sample_rate,
                duration_s=duration(r),
                settings=spec.loops.audio,
            )
            write_wav(path, audio, sampler.sample_rate)
            row = {
                key: r[key]
                for key in (
                    "pattern_id",
                    "dualized_pattern_id",
                    "split",
                    "tempo_bpm",
                    "source_performance",
                    "start_bar",
                )
            }
            row.update(
                clip_id=identity,
                source_clip_id=r["clip_id"],
                kit_id=kit.kit_id,
                kit_split=kit.split,
                midi_path=f"data/raw/gmd/groove/{r['midi_path']}",
                audio_path=audio_path(path),
                duration_s=len(audio) / sampler.sample_rate,
                sample_rate=sampler.sample_rate,
                lead_in_s=spec.loops.audio.lead_in_s,
                seed=seed,
            )
            lib_receipts.add(identity, row)
            library.append(row)
            if len(library) % 250 == 0:
                print(f"Library {len(library)}/{total}", flush=True)
        del sampler
    print(f"Library complete: {len(library)} files; generating queries", flush=True)
    modes = [m.capitalize() for m in spec.beatbox.avp_modes]
    for r in records:
        source = hashlib.sha256(r["clip_id"].encode()).hexdigest()[:16]
        people = sorted(assignments[assignments.split == r["split"]].participant_id)
        person = people[stable_seed(source + "person", seed) % len(people)]
        pool = samples[samples.participant_id == person]
        for mode in modes:
            identity = f"{tag}_{source}_{person}_{mode}_seed{seed}"
            previous = query_receipts.find(identity)
            if previous:
                queries.append(previous["row"])
                continue
            query_seed = stable_seed(f"{source}:{person}:{mode}", seed)
            events, reductions = beatbox_events(r, spec=spec, seed=query_seed)
            _, sr, trace = mix_query(
                events,
                pool,
                sample_root,
                participant=person,
                mode=mode,
                bpm=r["tempo_bpm"],
                pre_roll_s=spec.loops.audio.lead_in_s,
                duration_s=duration(r),
                seed=query_seed,
                normalize_output=False,
            )
            audio, params, actual = augment_query(
                trace,
                sample_root,
                sample_rate=sr,
                duration_s=duration(r),
                spec=spec,
                seed=query_seed,
                normalization=normal,
            )
            played = sorted(
                (e["audio_start_frame"], e["audio_end_frame"]) for e in actual if e["played"]
            )
            if any(a[1] > b[0] for a, b in zip(played, played[1:], strict=False)):
                raise ValueError(f"Overlapping vocal events: {identity}")
            path = out / "audio/queries" / r["split"] / f"{identity}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(path, audio, sr)
            row = {
                key: r[key]
                for key in (
                    "pattern_id",
                    "dualized_pattern_id",
                    "split",
                    "tempo_bpm",
                    "source_performance",
                    "start_bar",
                )
            }
            row.update(
                query_id=identity,
                source_clip_id=r["clip_id"],
                query_type="beatbox",
                source="synthetic",
                participant_id=person,
                mode=mode,
                audio_path=audio_path(path),
                duration_s=len(audio) / sr,
                sample_rate=sr,
                seed=query_seed,
                **reductions,
            )
            row.update({k: v for k, v in params.items() if k != "drop_prob_by_voice"})
            row["drop_prob_by_voice"] = json.dumps(params["drop_prob_by_voice"], sort_keys=True)
            query_receipts.add(identity, row, parameters=params, events=actual)
            queries.append(row)
            if len(queries) % 250 == 0:
                print(f"Queries {len(queries)}/{len(records) * len(modes)}", flush=True)
    library, queries = pd.DataFrame(library), pd.DataFrame(queries)
    validate_manifests(library, queries, assignments, spec)
    manifests = out / "manifests"
    manifests.mkdir(exist_ok=True)
    for name, table in (
        ("library_manifest.csv", library),
        ("query_manifest.csv", queries),
        ("avp_participant_splits.csv", assignments),
    ):
        table.to_csv(manifests / name, index=False)
    summary = {
        "spec_version": spec.version,
        "seed": seed,
        "source_loops": len(records),
        "unique_patterns": len({r["pattern_id"] for r in records}),
        "library_clips": len(library),
        "queries": len(queries),
        "library_by_split": library.split.value_counts().to_dict(),
        "queries_by_split": queries.split.value_counts().to_dict(),
        "library_by_kit": library.kit_id.value_counts().to_dict(),
        "participants_by_split": queries.groupby("split").participant_id.nunique().to_dict(),
        "speed_limited_hits": int(queries.speed_limited_hits.sum()),
        "vocal_overlap_check": "passed before noise on every generated query",
        "status": "rendered; full model-loader check and Joshua listening approval are separate",
        "limited_smoke_build": limit_per_split is not None,
    }
    atomic_json(out / "summary.json", summary)
    shutil.copyfile(ROOT / "data/raw/SOURCES.md", out / "SOURCES.md")
    # Small audition page references the full files, without copying audio.
    sections = []
    for split in ("train", "validation", "test"):
        for row in queries[queries.split == split].head(8).itertuples():
            target = library[(library.source_clip_id == row.source_clip_id)].iloc[0]
            drum_url = Path(target.audio_path)
            query_url = Path(row.audio_path)

            sections.append(
                f"<section><h2>{html.escape(row.query_id)}</h2>"
                f"<p>{split} · {html.escape(target.kit_id)}</p><p>Drums</p>"
                "<audio controls preload='none' "
                f"src='{html.escape(os.path.relpath(ROOT / drum_url, out))}'></audio>"
                "<p>Beatbox</p><audio controls preload='none' "
                f"src='{html.escape(os.path.relpath(ROOT / query_url, out))}'></audio></section>"
            )
    (out / "preview.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Full dataset audition</title>"
        "<style>body{font:16px system-ui;max-width:900px;margin:40px auto}"
        "section{border-top:1px solid #ddd;padding:16px}</style>"
        "<h1>Full dataset: listening sample</h1><p>Sloppy queries with matching drum targets. "
        "Labels refer to the source rhythm; voiced sounds follow spec simplification.</p>"
        + "".join(sections)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit-per-split", type=int, help="Smoke test only; use a separate --out")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Run C's full loader check, then copy manifests to the repository's manifests/",
    )
    parser.add_argument("--model-ref", default="origin/c/model-core")
    args = parser.parse_args()
    if args.publish and args.limit_per_split is not None:
        parser.error("A limited smoke build cannot publish production manifests")
    out = (args.out or ROOT / f"data/datasets/v{load_spec().version.replace('.', '')}").resolve()
    build_dataset(seed=args.seed, out_dir=out, limit_per_split=args.limit_per_split)
    if args.publish:
        from tools.check_model_loader import check_loader

        result = check_loader(out, args.model_ref)
        for path in (out / "manifests").glob("*.csv"):
            shutil.copyfile(path, ROOT / "manifests" / path.name)
        atomic_json(
            ROOT / "manifests/dataset_build.json",
            {
                "summary": json.loads((out / "summary.json").read_text()),
                "build_settings": json.loads((out / "build_settings.json").read_text()),
                "model_loader": {k: v for k, v in result.items() if k != "r1_checked_files"},
                "regenerate": (
                    f"python -m src.build_dataset --seed {args.seed} --publish "
                    f"--model-ref {result['model_revision']}"
                ),
            },
        )
        print("Full loader check passed; repository manifests published", flush=True)


if __name__ == "__main__":
    main()

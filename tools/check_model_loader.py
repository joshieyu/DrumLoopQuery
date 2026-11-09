"""Smoke-test manifests and audio using a pinned local snapshot of C's model branch.

    python -m tools.check_model_loader --model-ref origin/c/model-core

Does not merge the branch or alter src/models. Requires the optional handoff
extra (SciPy). Records the exact tested Git revision and feature shapes.
"""

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import src.models
from src.music.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def check_loader(dataset: Path, model_ref: str) -> dict:
    """Execute C's real loader and R1 extraction on every small-batch audio file."""
    revision = subprocess.check_output(["git", "rev-parse", model_ref], cwd=ROOT, text=True).strip()
    files = [
        "src/models/config.py",
        "src/models/data.py",
        "src/models/features.py",
        "config/models/base.yaml",
    ]
    with tempfile.TemporaryDirectory(prefix="drumloop-loader-") as temporary:
        folder = Path(temporary)
        for name in files:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
            )
        original_path = list(src.models.__path__)
        src.models.__path__.insert(0, str(folder / "src/models"))
        try:
            config = importlib.import_module("src.models.config")
            data = importlib.import_module("src.models.data")
            features = importlib.import_module("src.models.features")
            cfg = config.load_model_config(folder / "config/models/base.yaml")
            clips = data.load_clips(
                dataset / "manifests/library_manifest.csv",
                dataset / "manifests/query_manifest.csv",
                audio_root=ROOT,
            )
            data.check_disjoint(clips, ["train", "validation", "test"])
            spec = load_spec()
            library_manifest = pd.read_csv(dataset / "manifests/library_manifest.csv")
            query_manifest = pd.read_csv(dataset / "manifests/query_manifest.csv")
            assignments = pd.read_csv(dataset / "manifests/avp_participant_splits.csv")
            assigned = query_manifest.merge(
                assignments[["participant_id", "split"]],
                on="participant_id",
                validate="many_to_one",
                how="left",
                suffixes=("_query", "_assigned"),
            )
            if not (assigned.split_query == assigned.split_assigned).all():
                raise ValueError("Query participant does not belong to its assigned split")
            kit_splits = {k.kit_id: k.split for k in spec.rendering.kits}
            for row in library_manifest.itertuples(index=False):
                expected = "test" if row.split == "test" else "train"
                if kit_splits.get(row.kit_id) != expected:
                    raise ValueError(f"Kit/pattern split mismatch: {row.clip_id}")
            timing = pd.concat(
                [
                    library_manifest.rename(columns={"clip_id": "id"}),
                    query_manifest.rename(columns={"query_id": "id"}),
                ]
            ).set_index("id")
            numerator, denominator = map(int, spec.loops.time_signatures[0].split("-"))
            beats = spec.loops.bars * numerator * 4 / denominator
            details = []
            for row in clips.itertuples(index=False):
                audio, rate = sf.read(row.audio_path, always_2d=True)
                if not np.isfinite(audio).all() or not np.any(audio):
                    raise ValueError(f"Bad audio: {row.id}")
                if spec.loops.audio.end == "cut" and not np.all(audio[-1] == 0):
                    raise ValueError(f"Missing end fade: {row.id}")
                duration = spec.loops.audio.lead_in_s + beats * 60 / timing.loc[row.id, "tempo_bpm"]
                if spec.loops.audio.end == "cut" and len(audio) != round(duration * rate):
                    raise ValueError(f"Wrong clip framing: {row.id}")
                signal = features.load_audio(row.audio_path, cfg.audio.sample_rate)
                r1 = features.compute_features("r1", signal, cfg)
                if r1.ndim != 2 or r1.shape[1] != 3 or not np.isfinite(r1).all():
                    raise ValueError(f"Bad R1 features: {row.id}")
                details.append(
                    {
                        "id": row.id,
                        "audio_frames": len(audio),
                        "sample_rate": rate,
                        "feature_shape": list(r1.shape),
                    }
                )
            train = clips[clips.split == "train"]
            sampler = data.PatternBatchSampler(train, batch_patterns=2, views_per_pattern=3, seed=0)
            batches = sampler.epoch(0)
            if not batches:
                raise ValueError("No trainable contrastive batches")
            queries = clips[clips.kind == "beatbox"]
            library = clips[clips.kind == "library"]
            matches = data.relevant_library(queries, library, spec)
            if any(not ids for ids in matches.values()):
                raise ValueError("Some queries lack a matching library target")
            result = {
                "passed": True,
                "model_ref": model_ref,
                "model_revision": revision,
                "loaded_clips": len(clips),
                "train_batches": len(batches),
                "matched_queries": len(matches),
                "participant_kit_and_framing_checks": "passed",
                "r1_checked_files": details,
            }
        finally:
            src.models.__path__[:] = original_path
            for name in ("src.models.config", "src.models.data", "src.models.features"):
                sys.modules.pop(name, None)
    (dataset / "loader_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "results/v05_checkpoint")
    parser.add_argument("--model-ref", default="origin/c/model-core")
    args = parser.parse_args()
    result = check_loader(args.dataset, args.model_ref)
    print(json.dumps({k: v for k, v in result.items() if k != "r1_checked_files"}, indent=2))


if __name__ == "__main__":
    main()

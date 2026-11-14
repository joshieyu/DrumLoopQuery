"""Train the rhythm encoder (README §6) from library and query manifests.

    .venv/bin/python -m src.models.train --name r1_practice --front-end r1 \
        --library data/practice/manifests/library_manifest.csv \
        --queries data/practice/manifests/query_manifest.csv

Trains on the train split with SupCon (batches of training.batch_patterns patterns x
training.views_per_pattern versions, at least training.min_query_views of them
queries), AdamW, and a cosine learning-rate decay. After every epoch it checks
retrieval on the validation split (validation queries against the validation library)
with D's metric functions.

Before training it enforces the hard rules (CLAUDE.md): kits the music spec holds out
for testing are never used for training or validation; a query speaker may appear in
only one split; and a pattern found in more than one split is left out of training
and validation (logged), so the test split stays untouched. `--max-patterns N` trains
and validates on N random patterns per split, for a quick check of new data.

Writes results/runs/<name>/:

- best.pt: the epoch with the best validation MRR; last.pt: the final epoch
- log.csv: training loss and validation metrics per epoch
- run.json: the settings, data, and best epoch

Gradients are clipped at training.max_grad_norm: without that, some seeds collapsed to
one embedding for every clip (see src/models/encoder.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.config import ModelConfig, config_to_dict, load_model_config
from src.models.data import REPO_ROOT as DEFAULT_AUDIO_ROOT
from src.models.data import (
    FeatureCache,
    PatternBatchSampler,
    check_kits,
    check_speakers,
    held_out_kits,
    limit_patterns,
    load_clips,
    relevant_library,
    select,
)
from src.models.embed import embed_clips, pick_device, retrieval_metrics
from src.models.encoder import (
    PAD_MULTIPLE,
    RhythmEncoder,
    kind_ids,
    n_trainable,
    pad_features,
)
from src.models.features import feature_dim
from src.models.losses import positive_mask, supcon_loss
from src.music.spec import MusicSpec, load_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"


def validate(
    model: RhythmEncoder,
    clips: pd.DataFrame,
    cache: FeatureCache,
    spec: MusicSpec,
    device: torch.device,
) -> dict[str, float]:
    """Retrieval on one split: its queries against its own library."""
    library, queries = clips[clips.kind == "library"], clips[clips.kind != "library"]
    return retrieval_metrics(
        embed_clips(model, queries, cache, device),
        embed_clips(model, library, cache, device),
        list(queries.id),
        list(library.id),
        relevant_library(queries, library, spec),
    )


def training_sets(
    clips: pd.DataFrame,
    spec: MusicSpec,
    train_split: str = "train",
    val_split: str = "validation",
    max_patterns: int | None = None,
    seed: int = 0,
    log=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The train and validation clips, after the hard-rule checks (see the module docstring)."""
    check_kits(clips, spec)
    check_speakers(clips)
    per_split = clips.groupby("pattern_id").split.nunique()
    shared = set(per_split[per_split > 1].index)
    if shared:
        log(
            f"warning: {len(shared)} patterns appear in more than one split; leaving them out "
            f"of training and validation (e.g. {sorted(shared)[:3]})"
        )
    sets = []
    for split in (train_split, val_split):
        rows = select(clips, split, spec, kits="train")
        rows = rows[~rows.pattern_id.isin(shared)].reset_index(drop=True)
        sets.append(limit_patterns(rows, max_patterns, seed))
    return sets[0], sets[1]


def describe(name: str, clips: pd.DataFrame) -> str:
    """One line: how many clips, patterns, kits, and speakers a set has."""
    library, queries = clips[clips.kind == "library"], clips[clips.kind != "library"]
    return (
        f"{name}: {clips.pattern_id.nunique()} patterns, {len(library)} loops on kits "
        f"{sorted(library.group.unique())}, {len(queries)} queries from "
        f"{queries.group.nunique()} speakers"
    )


def train(
    *,
    cfg: ModelConfig,
    spec: MusicSpec,
    clips: pd.DataFrame,
    front_end: str,
    out_dir: Path,
    device: torch.device,
    train_split: str = "train",
    val_split: str = "validation",
    max_patterns: int | None = None,
    cache_root: Path | None = None,
    log=print,
) -> dict:
    """Train, validate each epoch, save checkpoints and the log. Returns the run summary."""
    t = cfg.training
    torch.manual_seed(t.seed)
    train_clips, val_clips = training_sets(
        clips, spec, train_split, val_split, max_patterns, t.seed, log
    )
    log(describe("train", train_clips))
    log(describe("validation", val_clips))
    log(f"held out for testing (never trained on): kits {held_out_kits(spec) or 'none yet'}")
    cache = FeatureCache.build(
        pd.concat([train_clips, val_clips], ignore_index=True), front_end, cfg, cache_root, log
    )
    sampler = PatternBatchSampler(
        train_clips, t.batch_patterns, t.views_per_pattern, t.seed, t.min_query_views
    )

    n_features = feature_dim(front_end, cfg)
    model = RhythmEncoder(n_features, cfg.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t.epochs * len(sampler))
    out_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"training {front_end} on {device}: {len(train_clips)} clips, {len(sampler.patterns)} "
        f"patterns, {len(sampler)} batches per epoch, {n_trainable(model):,} parameters"
    )

    history, best = [], None
    for epoch in range(1, t.epochs + 1):
        started = time.monotonic()
        model.train()
        losses = []
        for rows in sampler.epoch(epoch):
            batch = train_clips.iloc[rows]
            x, lengths = pad_features([cache.get(i) for i in batch.id], PAD_MULTIPLE)
            pos = positive_mask(
                list(batch.pattern_id), list(batch.dualized_pattern_id), list(batch.kind), spec
            )
            emb = model(x.to(device), lengths.to(device), kind_ids(list(batch.kind)).to(device))
            loss = supcon_loss(emb, pos.to(device), cfg.loss.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.max_grad_norm)
            opt.step()
            sched.step()
            losses.append(loss.item())

        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if len(val_clips):
            val = validate(model, val_clips, cache, spec, device)
            row.update({f"val_{k}": v for k, v in val.items() if k != "n_queries"})
        row["seconds"] = round(time.monotonic() - started, 2)
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "log.csv", index=False)

        checkpoint = {
            "model": model.state_dict(),
            "front_end": front_end,
            "n_features": n_features,
            "config": config_to_dict(cfg),
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        score = row.get("val_mrr", -row["train_loss"])
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "metrics": row}
            torch.save(checkpoint, out_dir / "best.pt")
        val_text = (
            f" | val MRR {row['val_mrr']:.3f} (chance {row['val_chance_mrr']:.3f}), "
            f"hit@1 {row['val_hit@1']:.3f}"
            if "val_mrr" in row
            else ""
        )
        log(f"epoch {epoch}/{t.epochs} loss {row['train_loss']:.3f}{val_text} | {row['seconds']} s")

    summary = {
        "front_end": front_end,
        "device": str(device),
        "train_clips": len(train_clips),
        "train_patterns": len(sampler.patterns),
        "train_kits": sorted(train_clips.loc[train_clips.kind == "library", "group"].unique()),
        "train_speakers": int(train_clips.loc[train_clips.kind != "library", "group"].nunique()),
        "val_clips": len(val_clips),
        "held_out_kits": held_out_kits(spec),
        "max_patterns": max_patterns,
        "parameters": n_trainable(model),
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "config": config_to_dict(cfg),
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the rhythm encoder.")
    parser.add_argument("--name", required=True, help="run name: results/runs/<name>")
    parser.add_argument("--front-end", required=True, choices=["r1", "r3"])
    parser.add_argument("--library", type=Path, required=True, help="library_manifest.csv")
    parser.add_argument("--queries", type=Path, required=True, help="query_manifest.csv")
    parser.add_argument("--config", type=Path, default=None, help="default: base.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--seed", type=int, default=None, help="override training.seed")
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=None,
        help="quick run: at most this many random patterns per split",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="folder relative audio_paths start from (default: the repo root)",
    )
    args = parser.parse_args(argv)

    cfg = load_model_config(args.config)
    overrides = {k: v for k, v in (("epochs", args.epochs), ("seed", args.seed)) if v is not None}
    if overrides:
        cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **overrides))
    summary = train(
        cfg=cfg,
        spec=load_spec(),
        clips=load_clips(args.library, args.queries, args.audio_root),
        front_end=args.front_end,
        out_dir=RUNS_DIR / args.name,
        device=pick_device(args.device),
        train_split=args.train_split,
        val_split=args.val_split,
        max_patterns=args.max_patterns,
    )
    print(f"best epoch {summary['best_epoch']}: {summary['best_metrics']}")
    print(f"run saved to {RUNS_DIR / args.name}")


if __name__ == "__main__":
    main()

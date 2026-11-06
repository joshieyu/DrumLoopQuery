"""Embed a test set with a trained model, write D's .npz file, and score it with D's code.

    .venv/bin/python -m src.models.embed --run results/runs/r1_practice \
        --library data/practice/manifests/library_manifest.csv \
        --queries data/practice/manifests/query_manifest.csv \
        --relevance data/practice/manifests/relevance.json \
        --split test --system r1_practice --testset practice_test

Embeds the split's library loops and queries, writes
results/embeddings/<system>_<testset>.npz (README §12), and scores it with
src.eval.evaluate's evaluate() and summarize(), D's single scoring code. Results go to
results/eval/<system>_<testset>/ (summary.json, per_query.csv), the layout D's command
line uses. (D's command line itself currently crashes on startup: evaluate.py has no
module docstring for argparse. Calling its functions directly avoids that.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.evaluate import (
    chance_rr,
    evaluate,
    load_relevance,
    query_metrics,
    rank_relevant,
    summarize,
)
from src.models.config import ModelConfig, config_from_dict
from src.models.data import FeatureCache, load_clips
from src.models.encoder import PAD_MULTIPLE, RhythmEncoder, kind_ids, pad_features
from src.models.export import embedding_path, write_embeddings

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "results" / "eval"


def pick_device(name: str = "auto") -> torch.device:
    """cuda, then Apple's mps, then cpu; or exactly the device named."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_clips(
    model: RhythmEncoder,
    clips: pd.DataFrame,
    cache: FeatureCache,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """(N, embed_dim) unit rows, in the order of `clips`."""
    model.eval()
    ids, kinds = list(clips.id), list(clips.kind)
    feats = [cache.get(i) for i in ids]
    order = np.argsort([len(f) for f in feats], kind="stable")  # similar lengths together
    out = np.zeros((len(ids), model.cfg.embed_dim), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        x, lengths = pad_features([feats[i] for i in idx], PAD_MULTIPLE)
        emb = model(x.to(device), lengths.to(device), kind_ids([kinds[i] for i in idx]).to(device))
        out[idx] = emb.float().cpu().numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def retrieval_metrics(
    query_emb: np.ndarray,
    library_emb: np.ndarray,
    query_ids: list[str],
    library_ids: list[str],
    relevance: dict[str, list[str]],
) -> dict[str, float]:
    """MRR, Hit@1, and chance MRR with D's per-query functions (for quick validation)."""
    scores = query_emb @ library_emb.T
    column = {c: i for i, c in enumerate(library_ids)}
    rr, hit1, chance = [], [], []
    for row, qid in enumerate(query_ids):
        rel = np.unique([column[c] for c in relevance.get(qid, []) if c in column])
        if rel.size == 0:
            continue
        m = query_metrics(rank_relevant(scores[row], rel))
        rr.append(m["rr"])
        hit1.append(m["hit@1"])
        chance.append(chance_rr(len(library_ids), int(rel.size)))
    if not rr:
        raise ValueError("no query has a relevant clip in this library")
    return {
        "mrr": float(np.mean(rr)),
        "hit@1": float(np.mean(hit1)),
        "chance_mrr": float(np.mean(chance)),
        "n_queries": len(rr),
    }


def load_run(
    run_dir: str | Path, which: str = "best", device: torch.device | None = None
) -> tuple[RhythmEncoder, ModelConfig, str, dict]:
    """A trained model from results/runs/<name>/<which>.pt, with its config and front-end."""
    ckpt = torch.load(Path(run_dir) / f"{which}.pt", map_location="cpu", weights_only=True)
    cfg = config_from_dict(ckpt["config"])
    model = RhythmEncoder(ckpt["n_features"], cfg.encoder)
    model.load_state_dict(ckpt["model"])
    if device is not None:
        model.to(device)
    return model.eval(), cfg, ckpt["front_end"], ckpt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed a test set and score it with D's code.")
    parser.add_argument("--run", type=Path, required=True, help="results/runs/<name>")
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--relevance", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--system", required=True, help="e.g. r1_practice")
    parser.add_argument("--testset", required=True, help="e.g. practice_test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, cfg, front_end, ckpt = load_run(args.run, args.which, device)
    clips = load_clips(args.library, args.queries)
    cache = FeatureCache.build(clips, front_end, cfg)  # the whole manifest, as in training
    test = clips[clips.split == args.split]
    library, queries = test[test.kind == "library"], test[test.kind != "library"]
    if library.empty or queries.empty:
        raise SystemExit(f"split {args.split!r} needs both library clips and queries")

    path = write_embeddings(
        embedding_path(args.system, args.testset),
        list(queries.id),
        embed_clips(model, queries, cache, device),
        list(library.id),
        embed_clips(model, library, cache, device),
    )
    table, counts = evaluate(path, load_relevance(args.relevance))
    summary = summarize(table, n_boot=args.n_boot)
    out = EVAL_DIR / f"{args.system}_{args.testset}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_query.csv", index=False)
    result = {
        "system_file": str(path),
        "run": str(args.run),
        "checkpoint_epoch": ckpt["epoch"],
        **counts,
        "metrics": summary,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"embeddings: {path}")
    for name in ("mrr", "hit@1", "hit@5", "hit@10", "map", "chance_mrr"):
        m = summary[name]
        print(f"  {name:10s} {m['mean']:.3f}  (95% CI {m['ci_low']:.3f} to {m['ci_high']:.3f})")
    print(f"  {counts['n_queries_scored']} queries vs {counts['n_library']} library loops")
    print(f"results: {out}")


if __name__ == "__main__":
    main()

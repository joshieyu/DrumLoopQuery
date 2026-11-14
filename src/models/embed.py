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

The library is the split's loops on the kits the music spec holds out for testing
(--library-kits test, the default), so scoring never uses a kit the model trained on.
If the spec holds no kit out yet (as with the practice data), every kit is used and a
warning says so. --library-kits all, or a comma-separated list, overrides that.

Cross-kit fallback (Joshua's presentation plan, if the beatbox generator is late):
--query-kit K uses the split's loops on kit K as the queries and its loops on the
other library kits as the library. The answer key then comes from the same-rhythm rule
(same pattern_id) and is saved next to the results; --relevance isn't needed.
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
from src.models.data import REPO_ROOT as DEFAULT_AUDIO_ROOT
from src.models.data import (
    FeatureCache,
    held_out_kits,
    load_clips,
    relevant_library,
    select,
)
from src.models.encoder import PAD_MULTIPLE, RhythmEncoder, kind_ids, pad_features
from src.models.export import embedding_path, write_embeddings
from src.music.spec import MusicSpec, load_spec

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


def library_kits(spec: MusicSpec, choice: str, log=print) -> str | list[str]:
    """Resolve --library-kits: "test" (held-out kits), "all", or "kit_a,kit_b"."""
    if choice == "test" and not held_out_kits(spec):
        log("warning: the music spec holds no kit out for testing yet; scoring on every kit")
        return "all"
    if choice in ("test", "all"):
        return choice
    return [k.strip() for k in choice.split(",") if k.strip()]


def scoring_sets(
    clips: pd.DataFrame,
    spec: MusicSpec,
    split: str,
    kits: str | list[str],
    query_kit: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]] | None]:
    """(queries, library, answer key or None) for one split.

    Normal mode: the split's queries against its loops on `kits`; the answer key is D's
    relevance.json (None here). Cross-kit mode (`query_kit`): the split's loops on that
    kit are the queries, its loops on the other `kits` the library, and the answer key
    comes from the same-rhythm rule.
    """
    rows = select(clips, split, spec, kits)
    library = rows[rows.kind == "library"]
    if query_kit is None:
        queries = rows[rows.kind != "library"]
        answer_key = None
    else:
        everything = select(clips, split, spec, "all")
        queries = everything[(everything.kind == "library") & (everything.group == query_kit)]
        library = library[library.group != query_kit]
        if queries.empty:
            raise ValueError(f"no {split} loops on query kit {query_kit!r}")
        answer_key = relevant_library(queries, library, spec)
    if library.empty or queries.empty:
        raise ValueError(f"split {split!r} needs both library loops and queries here")
    return queries.reset_index(drop=True), library.reset_index(drop=True), answer_key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed a test set and score it with D's code.")
    parser.add_argument("--run", type=Path, required=True, help="results/runs/<name>")
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--relevance", type=Path, help="D's answer key (not with --query-kit)")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--library-kits", default="test", help="test (held-out kits), all, or kit_a,kit_b"
    )
    parser.add_argument("--query-kit", help="cross-kit fallback: this kit's loops are the queries")
    parser.add_argument("--system", required=True, help="e.g. r1_practice")
    parser.add_argument("--testset", required=True, help="e.g. practice_test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    args = parser.parse_args(argv)
    if (args.relevance is None) == (args.query_kit is None):
        parser.error("give --relevance (D's answer key), or --query-kit for the cross-kit mode")

    spec = load_spec()
    device = pick_device(args.device)
    model, cfg, front_end, ckpt = load_run(args.run, args.which, device)
    clips = load_clips(args.library, args.queries, args.audio_root)
    kits = library_kits(spec, args.library_kits)
    queries, library, answer_key = scoring_sets(clips, spec, args.split, kits, args.query_kit)
    print(
        f"scoring {len(queries)} queries against {len(library)} loops on kits "
        f"{sorted(library.group.unique())}"
    )
    cache = FeatureCache.build(pd.concat([queries, library], ignore_index=True), front_end, cfg)

    path = write_embeddings(
        embedding_path(args.system, args.testset),
        list(queries.id),
        embed_clips(model, queries, cache, device),
        list(library.id),
        embed_clips(model, library, cache, device),
    )
    out = EVAL_DIR / f"{args.system}_{args.testset}"
    out.mkdir(parents=True, exist_ok=True)
    if answer_key is None:
        answer_key = load_relevance(args.relevance)
    else:
        (out / "relevance.json").write_text(json.dumps(answer_key, indent=1) + "\n")
    table, counts = evaluate(path, answer_key)
    summary = summarize(table, n_boot=args.n_boot)
    table.to_csv(out / "per_query.csv", index=False)
    result = {
        "system_file": str(path),
        "run": str(args.run),
        "checkpoint_epoch": ckpt["epoch"],
        "split": args.split,
        "library_kits": sorted(library.group.unique()),
        "query_kit": args.query_kit,
        "answer_key": str(args.relevance) if args.relevance else "same-rhythm rule (cross-kit)",
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

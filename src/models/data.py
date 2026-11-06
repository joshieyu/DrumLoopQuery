"""Manifests in, training batches out: Member C's data loading (README §12 contracts).

    from src.models.data import FeatureCache, PatternBatchSampler, load_clips

    clips = load_clips("manifests/library_manifest.csv", "manifests/query_manifest.csv")
    cache = FeatureCache.build(clips, "r1", cfg)    # computes each clip once, then reuses
    sampler = PatternBatchSampler(clips[clips.split == "train"], 32, 3, seed=0)
    for rows in sampler.epoch(1): ...               # row positions, P patterns x K views

`load_clips` puts library loops and queries in one table with the columns the model
needs: id, pattern_id, dualized_pattern_id, split, kind ("library", "beatbox", or
"tap"), audio_path (absolute), and group (kit_id for loops, participant_id for queries).
Relative audio paths are read from the repo root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.config import ModelConfig
from src.models.features import compute_features, load_audio
from src.music.spec import MusicSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data" / "features"
LIBRARY_COLUMNS = [
    "clip_id",
    "pattern_id",
    "dualized_pattern_id",
    "split",
    "kit_id",
    "tempo_bpm",
    "audio_path",
    "midi_path",
    "source_performance",
]
QUERY_COLUMNS = [
    "query_id",
    "pattern_id",
    "dualized_pattern_id",
    "query_type",
    "source",
    "participant_id",
    "split",
    "tempo_bpm",
    "jitter_ms",
    "drift_pct",
    "drop_prob",
    "snr_db",
    "audio_path",
]
QUERY_TYPES = ("beatbox", "tap")
CLIP_COLUMNS = ["id", "pattern_id", "dualized_pattern_id", "split", "kind", "audio_path", "group"]


# --------------------------------------------------------------------------- manifests


def load_clips(
    library_csv: str | Path,
    query_csv: str | Path | None = None,
    audio_root: str | Path = REPO_ROOT,
) -> pd.DataFrame:
    """Library loops and queries as one table (see the module docstring)."""
    library = _read_manifest(library_csv, LIBRARY_COLUMNS)
    tables = [
        pd.DataFrame(
            {
                "id": library.clip_id,
                "pattern_id": library.pattern_id,
                "dualized_pattern_id": library.dualized_pattern_id,
                "split": library.split,
                "kind": "library",
                "audio_path": library.audio_path,
                "group": library.kit_id,
            }
        )
    ]
    if query_csv is not None:
        queries = _read_manifest(query_csv, QUERY_COLUMNS)
        bad = sorted(set(queries.query_type) - set(QUERY_TYPES))
        if bad:
            raise ValueError(f"{query_csv}: unknown query_type {bad}; expected {QUERY_TYPES}")
        tables.append(
            pd.DataFrame(
                {
                    "id": queries.query_id,
                    "pattern_id": queries.pattern_id,
                    "dualized_pattern_id": queries.dualized_pattern_id,
                    "split": queries.split,
                    "kind": queries.query_type,
                    "audio_path": queries.audio_path,
                    "group": queries.participant_id,
                }
            )
        )
    clips = pd.concat(tables, ignore_index=True)
    duplicated = clips.id[clips.id.duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"ids appear more than once across the manifests, e.g. {duplicated[:3]}")
    root = Path(audio_root)
    clips["audio_path"] = [str(p if Path(p).is_absolute() else root / p) for p in clips.audio_path]
    return clips[CLIP_COLUMNS]


def _read_manifest(path: str | Path, columns: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (README §12)")
    empty = [c for c in ("pattern_id", "split", "audio_path") if (table[c] == "").any()]
    if empty:
        raise ValueError(f"{path}: empty values in {empty}")
    return table


def check_disjoint(clips: pd.DataFrame, splits: list[str]) -> None:
    """Raise if a pattern appears in more than one of `splits` (CLAUDE.md: keep splits disjoint)."""
    subset = clips[clips.split.isin(splits)]
    shared = subset.groupby("pattern_id").split.nunique()
    shared = shared[shared > 1]
    if len(shared):
        raise ValueError(
            f"{len(shared)} patterns appear in more than one of {splits}, e.g. "
            f"{list(shared.index[:3])}"
        )


def relevant_library(
    queries: pd.DataFrame, library: pd.DataFrame, spec: MusicSpec
) -> dict[str, list[str]]:
    """For each query, the library ids that match it (README §5.4, as positive_mask does).

    Used for validation during training; the official answer key is D's relevance.json.
    """
    if spec.same_rhythm.match != "exact":
        raise ValueError(f"relevant_library implements 'exact', not {spec.same_rhythm.match!r}")
    by_pattern = library.groupby("pattern_id").id.apply(list).to_dict()
    by_dualized = library.groupby("dualized_pattern_id").id.apply(list).to_dict()
    out = {}
    for q in queries.itertuples(index=False):
        if q.kind == "tap":
            out[q.id] = by_dualized.get(q.dualized_pattern_id, [])
        else:
            out[q.id] = by_pattern.get(q.pattern_id, [])
    return out


# --------------------------------------------------------------------------- feature cache


class FeatureCache:
    """Every clip's features for one front-end, computed once and saved as one .npz."""

    def __init__(self, ids: np.ndarray, offsets: np.ndarray, frames: np.ndarray):
        self.index = {str(i): k for k, i in enumerate(ids)}
        self.offsets = offsets
        self.frames = frames

    def get(self, clip_id: str) -> np.ndarray:
        k = self.index[clip_id]
        return self.frames[self.offsets[k] : self.offsets[k + 1]].astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def build(
        cls,
        clips: pd.DataFrame,
        front_end: str,
        cfg: ModelConfig,
        path: str | Path | None = None,
        log=print,
    ) -> FeatureCache:
        """Load the cache at `path` if it matches these clips and settings; else compute it.

        The match covers the front-end, the audio and feature settings, and every clip's
        id, audio path, size, and modification time, so re-rendered audio or a changed
        config never reuses stale features. Default path: data/features/<front_end>_<key>.npz.
        """
        key = _fingerprint(clips, front_end, cfg)
        path = Path(path) if path is not None else FEATURES_DIR / f"{front_end}_{key[:12]}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                if str(f["fingerprint"]) == key:
                    log(f"features: reusing {path}")
                    return cls(f["ids"], f["offsets"], f["frames"])
        log(f"features: computing {front_end} for {len(clips)} clips -> {path}")
        arrays = []
        for i, row in enumerate(clips.itertuples(index=False)):
            audio = load_audio(row.audio_path, cfg.audio.sample_rate)
            arrays.append(compute_features(front_end, audio, cfg).astype(np.float16))
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(clips)}")
        offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrays])]).astype(np.int64)
        frames = np.concatenate(arrays, axis=0)
        ids = np.array(list(clips.id), dtype=str)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=ids, offsets=offsets, frames=frames, fingerprint=np.array(key))
        return cls(ids, offsets, frames)


def _fingerprint(clips: pd.DataFrame, front_end: str, cfg: ModelConfig) -> str:
    settings = {"front_end": front_end, "audio": asdict(cfg.audio), "r1": asdict(cfg.r1)}
    settings["r3"] = asdict(cfg.r3)
    h = hashlib.sha1(json.dumps(settings, sort_keys=True).encode())
    for row in clips.itertuples(index=False):
        stat = Path(row.audio_path).stat()
        h.update(f"{row.id}|{row.audio_path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- batches


class PatternBatchSampler:
    """Batches of `batch_patterns` patterns with up to `views_per_pattern` clips each.

    Only patterns with at least two clips are used (a lone clip has no positive). Each
    epoch shuffles the patterns and, within a pattern, picks views without replacement.
    The order depends only on (seed, epoch), so runs are reproducible.
    """

    def __init__(
        self, clips: pd.DataFrame, batch_patterns: int, views_per_pattern: int, seed: int = 0
    ):
        if batch_patterns < 2:
            raise ValueError("batch_patterns must be at least 2, so every batch has negatives")
        groups = clips.reset_index(drop=True).groupby("pattern_id").indices
        self.patterns = sorted(p for p, rows in groups.items() if len(rows) >= 2)
        self.rows = {p: np.asarray(groups[p]) for p in self.patterns}
        self.batch_patterns = batch_patterns
        self.views = views_per_pattern
        self.seed = seed
        if len(self.patterns) < 2:
            raise ValueError("need at least two patterns with two or more clips each")

    def __len__(self) -> int:
        n = len(self.patterns)
        return n // self.batch_patterns + (1 if n % self.batch_patterns >= 2 else 0)

    def epoch(self, epoch: int) -> list[list[int]]:
        """Row positions (into the clips table given at construction) for each batch."""
        rng = np.random.default_rng([self.seed, epoch])
        order = [self.patterns[i] for i in rng.permutation(len(self.patterns))]
        batches = []
        for start in range(0, len(order), self.batch_patterns):
            chunk = order[start : start + self.batch_patterns]
            if len(chunk) < 2:
                continue
            batch = []
            for p in chunk:
                rows = self.rows[p]
                take = min(self.views, len(rows))
                batch.extend(int(r) for r in rng.choice(rows, size=take, replace=False))
            batches.append(batch)
        return batches

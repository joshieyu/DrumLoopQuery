"""Manifests in, training batches out: Member C's data loading (README §12 contracts).

    from src.models.data import FeatureCache, PatternBatchSampler, load_clips, select

    clips = load_clips("manifests/library_manifest.csv", "manifests/query_manifest.csv")
    check_kits(clips, spec); check_speakers(clips)
    train = select(clips, "train", spec, kits="train")   # library: kits not held out
    test = select(clips, "test", spec, kits="test")      # library: held-out kits only
    cache = FeatureCache.build(train, "r1", cfg)         # computes each clip once, ever
    sampler = PatternBatchSampler(train, 32, 3, seed=0, min_query_views=1)
    for rows in sampler.epoch(1): ...                    # row positions, P patterns x K views

`load_clips` puts library loops and queries in one table with the columns the model
needs: id, pattern_id, dualized_pattern_id, split, kind ("library", "beatbox", or
"tap"), source ("library" for loops; the query manifest's source otherwise),
audio_path (absolute), and group (kit_id for loops, participant_id for queries).
Relative audio paths are read from `audio_root` (default: the repo root).

Held-out kits come from the music spec: a kit whose rendering.kits split is "test"
is never used for training or validation, only for scoring (README §5.2).
"""

from __future__ import annotations

import hashlib
import json
import os
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
PRACTICE_SOURCE = "practice"  # src/models/practice.py's stand-in queries (their "speaker" is a kit)
CLIP_COLUMNS = [
    "id",
    "pattern_id",
    "dualized_pattern_id",
    "split",
    "kind",
    "source",
    "audio_path",
    "group",
]


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
                "source": "library",
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
                    "source": queries.source,
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


# --------------------------------------------------------------------------- splits and kits


def kit_splits(spec: MusicSpec) -> dict[str, str | None]:
    """Each kit's split from the music spec: "train", "test", or None (not used)."""
    return {kit.kit_id: kit.split for kit in spec.rendering.kits}


def held_out_kits(spec: MusicSpec) -> list[str]:
    """Kits the music lead held out for testing (rendering.kits split "test")."""
    return sorted(k for k, s in kit_splits(spec).items() if s == "test")


def check_kits(clips: pd.DataFrame, spec: MusicSpec) -> None:
    """Raise if a library loop uses a kit that isn't in the music spec."""
    used = set(clips.loc[clips.kind == "library", "group"])
    unknown = sorted(used - set(kit_splits(spec)))
    if unknown:
        raise ValueError(
            f"library kits not in config/music_spec.yaml rendering.kits: {unknown}; "
            "their train/test split is unknown"
        )


def select(
    clips: pd.DataFrame, split: str, spec: MusicSpec, kits: str | list[str] = "train"
) -> pd.DataFrame:
    """One split's clips, with its library loops limited to some kits.

    kits: "train" (the kits marked split: train; for training and validation), "test"
    (the held-out kits; raises if the spec holds none out), "all", or a list of kit ids.
    Kits whose split is unset aren't used, so "train" and "test" both leave them out.
    Queries are never filtered by kit.
    """
    rows = clips[clips.split == split]
    library = rows.kind == "library"
    splits = kit_splits(spec)
    if kits == "train":
        keep = {k for k, s in splits.items() if s == "train"}
    elif kits == "test":
        keep = set(held_out_kits(spec))
        if not keep:
            raise ValueError("no kit is marked split: test in config/music_spec.yaml yet")
    elif kits == "all":
        keep = set(rows.loc[library, "group"])
    else:
        keep = set(kits)
        unknown = sorted(keep - set(splits))
        if unknown:
            raise ValueError(f"kits not in the music spec: {unknown}")
    return rows[~library | rows.group.isin(keep)].reset_index(drop=True)


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


def check_speakers(clips: pd.DataFrame) -> None:
    """Raise if a query speaker (participant_id) appears in more than one split.

    CLAUDE.md: test AVP participants never appear in training. Practice stand-in
    queries are skipped (their "speaker" is a drum kit), and so are empty ids.
    """
    queries = clips[
        (clips.kind != "library") & (clips.source != PRACTICE_SOURCE) & (clips.group != "")
    ]
    shared = queries.groupby("group").split.unique()
    shared = shared[shared.map(len) > 1]
    if len(shared):
        example = {k: sorted(v) for k, v in shared.head(3).items()}
        raise ValueError(f"{len(shared)} speakers appear in more than one split, e.g. {example}")


def limit_patterns(clips: pd.DataFrame, max_patterns: int | None, seed: int) -> pd.DataFrame:
    """At most `max_patterns` patterns' clips, chosen at random (for quick runs)."""
    patterns = sorted(clips.pattern_id.unique())
    if max_patterns is None or len(patterns) <= max_patterns:
        return clips
    keep = set(np.random.default_rng(seed).choice(patterns, size=max_patterns, replace=False))
    return clips[clips.pattern_id.isin(keep)].reset_index(drop=True)


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
    """Features for one front-end, one small .npy file per clip, read from disk on demand.

    Files live in data/features/<front_end>_<settings key>/ and are named by a hash of
    the clip's id, audio path, size, and modification time. So any subset of clips
    reuses what is already computed, re-rendered audio gets fresh features, and a
    changed config uses a different folder. Nothing is held in memory: v0's ~40,000
    clips would be ~2.4 GB of R3 features.
    """

    def __init__(self, root: Path, files: dict[str, Path]):
        self.root = root
        self.files = files

    def get(self, clip_id: str) -> np.ndarray:
        return np.load(self.files[clip_id]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.files)

    @classmethod
    def build(
        cls,
        clips: pd.DataFrame,
        front_end: str,
        cfg: ModelConfig,
        root: str | Path | None = None,
        log=print,
    ) -> FeatureCache:
        """Compute features for every clip that isn't cached yet; return the cache."""
        settings = _settings(front_end, cfg)
        key = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        root = Path(root) if root is not None else FEATURES_DIR / f"{front_end}_{key[:12]}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

        files, missing = {}, []
        for row in clips.itertuples(index=False):
            path = root / f"{_clip_key(row.id, row.audio_path)}.npy"
            files[row.id] = path
            if not path.exists():
                missing.append((row, path))
        if not missing:
            log(f"features: all {len(clips)} clips already cached in {root}")
        else:
            log(f"features: computing {front_end} for {len(missing)} of {len(clips)} clips")
        for i, (row, path) in enumerate(missing):
            audio = load_audio(row.audio_path, cfg.audio.sample_rate)
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, compute_features(front_end, audio, cfg).astype(np.float16))
            os.replace(tmp, path)  # a crash never leaves a half-written file under the real name
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(missing)}")
        return cls(root, files)


def _settings(front_end: str, cfg: ModelConfig) -> dict:
    return {
        "front_end": front_end,
        "audio": asdict(cfg.audio),
        "r1": asdict(cfg.r1),
        "r3": asdict(cfg.r3),
    }


def _clip_key(clip_id: str, audio_path: str) -> str:
    stat = Path(audio_path).stat()
    text = f"{clip_id}|{audio_path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(text.encode()).hexdigest()[:24]


# --------------------------------------------------------------------------- batches


class PatternBatchSampler:
    """Batches of `batch_patterns` patterns with up to `views_per_pattern` clips each.

    Only patterns with at least two clips are used (a lone clip has no positive). With
    `min_query_views`, each pattern that has queries contributes at least that many of
    them plus at least one library loop when it has one, so batches hold query-to-loop
    pairs (the retrieval task) rather than only loop-to-loop ones. The other views are
    picked at random, without replacement. The order depends only on (seed, epoch).
    """

    def __init__(
        self,
        clips: pd.DataFrame,
        batch_patterns: int,
        views_per_pattern: int,
        seed: int = 0,
        min_query_views: int = 0,
    ):
        if batch_patterns < 2:
            raise ValueError("batch_patterns must be at least 2, so every batch has negatives")
        if not 0 <= min_query_views <= views_per_pattern:
            raise ValueError("min_query_views must be between 0 and views_per_pattern")
        table = clips.reset_index(drop=True)
        is_query = (table.kind != "library").to_numpy()
        groups = table.groupby("pattern_id").indices
        self.patterns = sorted(p for p, rows in groups.items() if len(rows) >= 2)
        self.queries = {p: groups[p][is_query[groups[p]]] for p in self.patterns}
        self.loops = {p: groups[p][~is_query[groups[p]]] for p in self.patterns}
        self.batch_patterns = batch_patterns
        self.views = views_per_pattern
        self.min_query_views = min_query_views
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
                batch.extend(self._views(p, rng))
            batches.append(batch)
        return batches

    def _views(self, pattern: str, rng: np.random.Generator) -> list[int]:
        queries, loops = self.queries[pattern], self.loops[pattern]
        chosen: list[int] = []
        if self.min_query_views and len(queries):
            n = min(self.min_query_views, len(queries))
            chosen += [int(r) for r in rng.choice(queries, size=n, replace=False)]
            if len(loops) and len(chosen) < self.views:
                chosen.append(int(rng.choice(loops)))
        rest = np.setdiff1d(np.concatenate([queries, loops]), chosen)
        take = min(self.views - len(chosen), len(rest))
        if take > 0:
            chosen += [int(r) for r in rng.choice(rest, size=take, replace=False)]
        return chosen

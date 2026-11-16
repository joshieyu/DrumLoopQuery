from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.music.spec import MusicSpec, load_spec

from src.eval.audio_io import load_mono, missing_files
from src.eval.evaluate import EvalError, load_relevance
from src.eval.hit_classifier import (
    LABEL_TO_VOICE,
    NEXT_HIT_GUARD_S,
    HitClassifier,
    hit_features,
    train_classifier,
)
from src.eval.library_grids import GRID_COLUMNS, source_grids
from src.eval.onset_baseline import ANALYSIS_RATE, HOP, lead_in_seconds, loop_beats, onset_bands
from src.eval.pool import Pool, load_pool
from src.eval.relevance import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
B3_DIR = REPO_ROOT / "results/b3"

## the flux curve peaks about this long before a hit actually starts; tune() recalibrates it
DEFAULT_DELAY_MS = 16.0
DEFAULT_PEAK_FRACTION = 0.2
DEFAULT_MIN_GAP_MS = 60.0
## sounds this close to the loop's edges are still kept, as an early or late hit
EDGE_S = 0.05
## a query onset and a library onset on the same voice match within this many grid steps
TOLERANCE_STEPS = 1

TUNE_FRACTIONS = (0.1, 0.15, 0.2, 0.3, 0.4)
TUNE_GAPS_MS = (40.0, 60.0, 90.0)
TUNE_TOLERANCE_S = 0.05
TIGHT_TOLERANCE_S = 0.03


@dataclass(frozen=True)
class Settings:
    peak_fraction: float = DEFAULT_PEAK_FRACTION
    min_gap_ms: float = DEFAULT_MIN_GAP_MS
    delay_ms: float = DEFAULT_DELAY_MS


def load_settings(path: Path) -> Settings:
    if not path.exists():
        print(f"no settings at {path}: using defaults (run the tune command first)")
        return Settings()
    raw = json.loads(path.read_text())
    return Settings(raw["peak_fraction"], raw["min_gap_ms"], raw["delay_ms"])


def detect_onsets(wave: np.ndarray, settings: Settings) -> np.ndarray:
    ## the strongest change in any of the three bands, so kicks (low), snares and hats (high)
    ## are all found whatever their width
    times, flux = onset_bands(wave, ANALYSIS_RATE, 3, per_bin_mean=True)
    curve = flux.max(axis=0)
    top = curve.max() if len(curve) else 0.0
    if top <= 0:
        return np.zeros(0)
    inner = np.arange(1, len(curve) - 1)
    is_peak = (
        (curve[inner] > curve[inner - 1])
        & (curve[inner] >= curve[inner + 1])
        & (curve[inner] >= settings.peak_fraction * top)
    )
    candidates = inner[is_peak]
    gap = max(1, round(settings.min_gap_ms / 1000.0 * ANALYSIS_RATE / HOP))
    kept: list[int] = []
    ## strongest peaks first, so a weaker one too close to a stronger one is dropped
    for i in candidates[np.argsort(-curve[candidates], kind="stable")]:
        if all(abs(i - j) >= gap for j in kept):
            kept.append(int(i))
    return np.sort(times[kept]) + settings.delay_ms / 1000.0


def label_hits(wave: np.ndarray, times: np.ndarray, classifier: HitClassifier) -> np.ndarray:
    if len(times) == 0:
        return np.zeros(0, dtype=int)
    limits = np.append(times[1:] - NEXT_HIT_GUARD_S, np.inf)
    features = np.stack(
        [
            hit_features(wave, ANALYSIS_RATE, t, None if np.isinf(limit) else limit)
            for t, limit in zip(times, limits, strict=True)
        ]
    )
    return classifier.predict(features)


def quantize(
    times: np.ndarray,
    voice_rows: np.ndarray,
    lead_in_s: float,
    tempo_bpm: float,
    n_voices: int,
    n_steps: int,
    steps_per_beat: int,
) -> np.ndarray:
    grid = np.zeros((n_voices, n_steps), dtype=bool)
    step_s = 60.0 / tempo_bpm / steps_per_beat
    steps = np.rint((times - lead_in_s) / step_s).astype(int)
    ## a downbeat played a hair early is still the downbeat; anything further out is dropped
    steps[steps == -1] = 0
    inside = (steps >= 0) & (steps < n_steps)
    grid[voice_rows[inside], steps[inside]] = True
    return grid


def transcribe_wave(
    wave: np.ndarray,
    tempo_bpm: float,
    duration_s: float,
    settings: Settings,
    classifier: HitClassifier,
    spec: MusicSpec,
) -> tuple[np.ndarray, int]:
    beats = loop_beats(spec)
    lead_in_s = lead_in_seconds(duration_s, tempo_bpm, beats)
    loop_s = beats * 60.0 / tempo_bpm
    times = detect_onsets(wave, settings)
    times = times[(times >= lead_in_s - EDGE_S) & (times < lead_in_s + loop_s + EDGE_S)]
    steps_per_beat = spec.same_rhythm.steps_per_beat
    n_steps = round(beats * steps_per_beat)
    rows = label_hits(wave, times, classifier)
    grid = quantize(times, rows, lead_in_s, tempo_bpm, len(spec.voices), n_steps, steps_per_beat)
    return grid, len(times)


def transcribe_queries(
    queries: pd.DataFrame,
    settings: Settings,
    classifier: HitClassifier,
    spec: MusicSpec,
) -> tuple[np.ndarray, np.ndarray]:
    grids, counts = [], []
    for i, row in enumerate(queries.itertuples()):
        wave = load_mono(row.audio_path, ANALYSIS_RATE)
        grid, n_hits = transcribe_wave(
            wave, row.tempo_bpm, row.duration_s, settings, classifier, spec
        )
        grids.append(grid)
        counts.append(n_hits)
        if i % 200 == 0 or i == len(queries) - 1:
            print(f"  queries {i + 1}/{len(queries)}", flush=True)
    return np.stack(grids), np.array(counts)


def dilate(grids: np.ndarray, tolerance: int) -> np.ndarray:
    out = grids.copy()
    for shift in range(1, tolerance + 1):
        out[..., shift:] |= grids[..., :-shift]
        out[..., :-shift] |= grids[..., shift:]
    return out


def tolerant_f1(
    query_grids: np.ndarray, library_grids: np.ndarray, tolerance: int = TOLERANCE_STEPS
) -> np.ndarray:
    ## precision: query hits with a library hit of the same voice within the tolerance.
    ## recall: library hits with a query hit of the same voice within the tolerance
    q = query_grids.reshape(len(query_grids), -1).astype(np.float32)
    lib = library_grids.reshape(len(library_grids), -1).astype(np.float32)
    q_wide = dilate(query_grids, tolerance).reshape(len(query_grids), -1).astype(np.float32)
    lib_wide = dilate(library_grids, tolerance).reshape(len(library_grids), -1).astype(np.float32)
    precision = (q @ lib_wide.T) / np.maximum(q.sum(axis=1), 1.0)[:, None]
    recall = (q_wide @ lib.T) / np.maximum(lib.sum(axis=1), 1.0)[None, :]
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)


def match_onsets(
    detected: np.ndarray, annotated: np.ndarray, tolerance_s: float
) -> list[tuple[int, int]]:
    ## one-to-one pairs, closest first: (index into annotated, index into detected)
    if len(detected) == 0 or len(annotated) == 0:
        return []
    distance = np.abs(annotated[:, None] - detected[None, :])
    pairs = []
    while True:
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] > tolerance_s:
            return pairs
        pairs.append((int(i), int(j)))
        distance[i, :] = np.inf
        distance[:, j] = np.inf


def detection_scores(recordings: list, settings: Settings, tolerance_s: float) -> dict:
    tp = fp = fn = 0
    errors: list[float] = []
    for wave, annotated, _ in recordings:
        detected = detect_onsets(wave, settings)
        pairs = match_onsets(detected, annotated, tolerance_s)
        tp += len(pairs)
        fp += len(detected) - len(pairs)
        fn += len(annotated) - len(pairs)
        errors += [annotated[i] - detected[j] for i, j in pairs]
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "median_error_ms": 1000.0 * float(np.median(errors)) if errors else 0.0,
    }


def label_accuracy(
    recordings: list, settings: Settings, classifier: HitClassifier, tolerance_s: float
) -> float:
    right = total = 0
    for wave, annotated, voices in recordings:
        detected = detect_onsets(wave, settings)
        guesses = label_hits(wave, detected, classifier)
        for i, j in match_onsets(detected, annotated, tolerance_s):
            right += int(guesses[j] == voices[i])
            total += 1
    return right / max(total, 1)


def validation_recordings(
    index: pd.DataFrame, avp_root: Path, participants: set[str], voices: tuple[str, ...]
) -> list:
    ## real beatbox improvisations with hand-annotated hits, from validation participants only
    voice_row = {voice: i for i, voice in enumerate(voices)}
    rows = index[
        index["participant_id"].isin(participants)
        & (index["kind"].str.lower() == "improvisation")
        & index["label"].isin(LABEL_TO_VOICE)
    ]
    recordings = []
    for audio_path, group in rows.groupby("audio_path", sort=True):
        group = group.sort_values("onset_s")
        recordings.append(
            (
                load_mono(avp_root / audio_path, ANALYSIS_RATE),
                group["onset_s"].to_numpy(dtype=float),
                np.array([voice_row[LABEL_TO_VOICE[label]] for label in group["label"]]),
            )
        )
    if not recordings:
        raise EvalError("no validation improvisation recordings found")
    return recordings


def tune(recordings: list, classifier: HitClassifier | None = None) -> tuple[Settings, dict]:
    rows = []
    for fraction, gap in itertools.product(TUNE_FRACTIONS, TUNE_GAPS_MS):
        score = detection_scores(recordings, Settings(fraction, gap), TUNE_TOLERANCE_S)
        rows.append({"peak_fraction": fraction, "min_gap_ms": gap, **score})
        print(
            f"  fraction {fraction:.2f}  gap {gap:3.0f} ms  F1 {score['f1']:.3f}  "
            f"P {score['precision']:.3f}  R {score['recall']:.3f}"
        )
    best = max(rows, key=lambda r: r["f1"])
    ## move the delay by the typical remaining error, so hits are placed where they start
    delay_ms = DEFAULT_DELAY_MS + best["median_error_ms"]
    settings = Settings(best["peak_fraction"], best["min_gap_ms"], delay_ms)
    final = detection_scores(recordings, settings, TIGHT_TOLERANCE_S)
    report = {
        "validation_recordings": len(recordings),
        "validation_annotated_hits": int(sum(len(a) for _, a, _ in recordings)),
        "f1_within_50ms_before_delay_fix": best["f1"],
        "median_error_ms_before_delay_fix": best["median_error_ms"],
        "f1_within_30ms_after_delay_fix": final["f1"],
        "median_error_ms_after_delay_fix": final["median_error_ms"],
    }
    if classifier is not None:
        report["label_accuracy_on_matched_hits"] = label_accuracy(
            recordings, settings, classifier, TIGHT_TOLERANCE_S
        )
    return settings, report


def score_queries(
    pool: Pool,
    library_extra: pd.DataFrame,
    spec: MusicSpec,
    classifier: HitClassifier,
    settings: Settings,
    allow_mismatch: bool = False,
) -> tuple[np.ndarray, dict, np.ndarray]:
    clips = library_extra.set_index("clip_id").loc[pool.library["clip_id"]].reset_index()
    grids = source_grids(clips, spec)
    if grids.mismatched and not allow_mismatch:
        raise EvalError(
            f"{len(grids.mismatched)} library clips have a MIDI grid whose pattern_id differs from "
            f"the manifest's, e.g. {grids.mismatched[:3]}; the loop cutting may not match B's"
        )
    clip_rows = grids.rows_for(clips["source_clip_id"])
    query_grids, counts = transcribe_queries(pool.queries, settings, classifier, spec)
    scores = tolerant_f1(query_grids, grids.onsets)[:, clip_rows]
    stats = {
        "queries": len(counts),
        "median_hits_detected": float(np.median(counts)),
        "median_hits_in_library_loops": float(np.median(grids.onsets.sum(axis=(1, 2)))),
        "queries_with_no_hits": int((counts == 0).sum()),
        "library_grids_not_matching_manifest": len(grids.mismatched),
    }
    return scores.astype(np.float32), stats, query_grids


## the generator records how many hits it took out of each beatbox, in these manifest columns
REMOVAL_COLUMNS = (
    "thinned_hat_hits",
    "suppressed_simultaneous",
    "speed_limited_hits",
    "dropped_hits",
)


def diagnose_queries(
    query_ids: np.ndarray,
    query_grids: np.ndarray,
    true_grids: np.ndarray,
    removed: pd.DataFrame,
) -> pd.DataFrame:
    ## query_grids: what B3 transcribed. true_grids: the library loop each query was made from.
    ## removed: per query (indexed by query_id), the hits the generator took out of that loop
    n = len(query_ids)
    transcribed = query_grids.reshape(n, -1).sum(axis=1)
    in_loop = true_grids.reshape(n, -1).sum(axis=1)
    matching = (query_grids & dilate(true_grids, TOLERANCE_STEPS)).reshape(n, -1).sum(axis=1)
    taken_out = removed.reindex(query_ids)[list(REMOVAL_COLUMNS)].sum(axis=1).to_numpy()
    return pd.DataFrame(
        {
            "query_id": query_ids,
            "hits_in_true_loop": in_loop,
            "hits_generator_removed": taken_out,
            "hits_expected_voiced": np.maximum(in_loop - taken_out, 0),
            "hits_transcribed": transcribed,
            "transcribed_hits_matching_true_loop": matching,
        }
    )


def summarize_diagnostics(table: pd.DataFrame) -> dict:
    expected = table["hits_expected_voiced"]
    usable = expected > 0
    return {
        "queries": len(table),
        "median_hits_in_true_loop": float(table["hits_in_true_loop"].median()),
        "median_hits_expected_voiced": float(expected.median()),
        "median_hits_transcribed": float(table["hits_transcribed"].median()),
        "transcribed_vs_expected_median_ratio": float(
            (table.loc[usable, "hits_transcribed"] / expected[usable]).median()
        ),
        "share_of_transcribed_hits_matching_true_loop": float(
            table["transcribed_hits_matching_true_loop"].sum()
            / max(table["hits_transcribed"].sum(), 1)
        ),
        ## if the four removal counts overlapped, this would be above zero
        "queries_where_removed_exceeds_loop": int(
            (table["hits_generator_removed"] > table["hits_in_true_loop"]).sum()
        ),
    }


def _add_avp_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--avp-index", type=Path, default=REPO_ROOT / "results/avp_index/utterance_index.csv"
    )
    parser.add_argument("--avp-root", type=Path, default=REPO_ROOT / "data/raw/avp/AVP_Dataset")
    parser.add_argument(
        "--splits", type=Path, default=REPO_ROOT / "manifests/avp_participant_splits.csv"
    )
    parser.add_argument("--classifier", type=Path, default=B3_DIR / "classifier.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe-then-match baseline (B3).")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train the hit classifier on AVP train participants")
    _add_avp_arguments(train)
    tune_parser = sub.add_parser("tune", help="tune the onset detector on AVP validation")
    _add_avp_arguments(tune_parser)
    tune_parser.add_argument("--settings", type=Path, default=B3_DIR / "settings.json")
    run = sub.add_parser("run", help="transcribe the test queries and score them")
    run.add_argument("--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv")
    run.add_argument("--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv")
    run.add_argument("--relevance", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    run.add_argument("--pool-kits", nargs="+", metavar="KIT_ID")
    run.add_argument("--classifier", type=Path, default=B3_DIR / "classifier.npz")
    run.add_argument("--settings", type=Path, default=B3_DIR / "settings.json")
    run.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results/embeddings/transcribe_syn_beatbox.npz"
    )
    run.add_argument("--allow-mismatch", action="store_true")
    run.add_argument("--grids", type=Path, default=B3_DIR / "query_grids.npz")
    diagnose = sub.add_parser("diagnose", help="compare transcribed grids with the true loops")
    diagnose.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    diagnose.add_argument(
        "--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv"
    )
    diagnose.add_argument("--relevance", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    diagnose.add_argument("--grids", type=Path, default=B3_DIR / "query_grids.npz")
    diagnose.add_argument("--out", type=Path, default=B3_DIR / "diagnostics.csv")
    args = parser.parse_args()
    try:
        spec = load_spec()
        ## say what is missing and what to run, instead of a bare file-not-found
        if args.command in ("train", "tune"):
            if not args.avp_root.is_dir():
                raise EvalError(
                    f"AVP audio not found at {args.avp_root}. "
                    "Download it with: python -m tools.download_datasets"
                )
            if not args.avp_index.exists():
                raise EvalError(
                    f"AVP index not found at {args.avp_index}. "
                    "Build it with: python -m src.queries.index_avp"
                )
        if args.command == "run" and not args.classifier.exists():
            raise EvalError(
                f"no classifier at {args.classifier}. Run the train command first "
                "(and tune, to set the detector)"
            )
        if args.command == "train":
            metrics = train_classifier(
                args.avp_index, args.avp_root, args.splits, spec.voices, args.classifier
            )
            print(json.dumps(metrics, indent=2))
            print(f"Wrote {args.classifier.resolve()}")
        elif args.command == "tune":
            index = pd.read_csv(args.avp_index)
            splits = read_manifest(args.splits, ("participant_id", "split"))
            validation = set(splits.loc[splits["split"] == "validation", "participant_id"])
            recordings = validation_recordings(index, args.avp_root, validation, spec.voices)
            classifier = HitClassifier.load(args.classifier) if args.classifier.exists() else None
            settings, report = tune(recordings, classifier)
            args.settings.parent.mkdir(parents=True, exist_ok=True)
            args.settings.write_text(
                json.dumps({**asdict(settings), "tuning": report}, indent=2) + "\n"
            )
            print(json.dumps({**asdict(settings), **report}, indent=2))
            print(f"Wrote {args.settings.resolve()}")
        elif args.command == "diagnose":
            pool = load_pool(args.library, args.queries, args.relevance, pool_kits=None)
            relevance = load_relevance(args.relevance)
            saved = np.load(args.grids, allow_pickle=False)
            query_ids = saved["query_ids"].astype(str)
            if set(query_ids) != set(pool.queries["query_id"]):
                raise EvalError("the saved grids are for different queries; rerun the run command")
            library_extra = read_manifest(args.library, ("clip_id", *GRID_COLUMNS))
            first_clips = [relevance[q][0] for q in query_ids]
            needed = library_extra.set_index("clip_id").loc[sorted(set(first_clips))].reset_index()
            truth = source_grids(needed, spec)
            row_of = dict(
                zip(needed["clip_id"], truth.rows_for(needed["source_clip_id"]), strict=True)
            )
            true_grids = np.stack([truth.onsets[row_of[c]] for c in first_clips])
            removed = pd.read_csv(args.queries).set_index("query_id")
            missing = [c for c in REMOVAL_COLUMNS if c not in removed.columns]
            if missing:
                raise EvalError(f"query manifest is missing {missing}")
            table = diagnose_queries(query_ids, saved["query_grids"], true_grids, removed)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(json.dumps(summarize_diagnostics(table), indent=2))
            print(f"Wrote {args.out.resolve()}")
        else:
            pool = load_pool(
                args.library, args.queries, args.relevance, pool_kits=args.pool_kits, timing=True
            )
            missing = missing_files(pool.queries["audio_path"].tolist())
            if missing:
                raise EvalError(f"{len(missing)} query audio files are missing, e.g. {missing[:3]}")
            classifier = HitClassifier.load(args.classifier)
            if classifier.classes != tuple(spec.voices):
                raise EvalError(f"classifier voices {classifier.classes} differ from the spec's")
            library_extra = read_manifest(args.library, ("clip_id", *GRID_COLUMNS))
            settings = load_settings(args.settings)
            print(f"{len(pool.library)} library clips, {len(pool.queries)} queries, {settings}")
            scores, stats, query_grids = score_queries(
                pool, library_extra, spec, classifier, settings, args.allow_mismatch
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.out,
                query_ids=pool.queries["query_id"].to_numpy(dtype=str),
                library_ids=pool.library["clip_id"].to_numpy(dtype=str),
                scores=scores,
            )
            ## kept so the diagnose command can compare them with the true loops without audio
            args.grids.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.grids,
                query_ids=pool.queries["query_id"].to_numpy(dtype=str),
                query_grids=query_grids,
            )
            print(json.dumps(stats, indent=2))
            print(f"Wrote {args.out.resolve()}")
    except ImportError as error:
        parser.exit(1, f"Missing library ({error}). Install: pip install soundfile scipy\n")
    except (OSError, EvalError) as error:
        parser.exit(1, f"Transcribe baseline failed: {error}\n")


if __name__ == "__main__":
    main()

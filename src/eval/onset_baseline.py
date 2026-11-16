from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from src.music.spec import MusicSpec, load_spec

from src.eval.audio_io import load_mono, missing_files
from src.eval.evaluate import EvalError
from src.eval.pool import Pool, load_pool

REPO_ROOT = Path(__file__).resolve().parents[2]

## signal-processing settings, not musical rules, so they live here and not in the spec
ANALYSIS_RATE = 22_050
N_FFT = 1024
HOP = 128
## bands for the 3-band version: low (kick), mid (snare), high (hats)
BAND_EDGES_HZ = (250.0, 3000.0)
## every loop is resampled to this many time bins, whatever its tempo
N_BINS = 384
SMOOTH_BINS = 1.5
## how far a query may be shifted against a library clip, as a share of the loop
MAX_LAG_FRACTION = 0.03
DB_FLOOR = -60.0
## a clip may be a little shorter than its nominal loop (tempo drift); more than this is an error
LEAD_IN_TOLERANCE_S = 0.1
MAX_LEAD_IN_S = 1.0


def loop_beats(spec: MusicSpec) -> float:
    if len(spec.loops.time_signatures) != 1:
        raise EvalError("onset baseline needs exactly one time signature in the spec")
    numerator, denominator = map(int, spec.loops.time_signatures[0].split("-"))
    return spec.loops.bars * numerator * 4 / denominator


def lead_in_seconds(duration_s: float, tempo_bpm: float, beats: float) -> float:
    ## each clip is a short lead-in followed by exactly one loop
    lead = duration_s - beats * 60.0 / tempo_bpm
    if lead < -LEAD_IN_TOLERANCE_S or lead > MAX_LEAD_IN_S:
        raise EvalError(
            f"a {duration_s:.2f} s clip at {tempo_bpm:g} BPM does not fit a {beats:g}-beat loop "
            f"(lead-in would be {lead:.2f} s)"
        )
    return max(lead, 0.0)


def band_edges(n_bands: int, sample_rate: int) -> list[float]:
    top = sample_rate / 2 + 1.0
    if n_bands == 1:
        return [0.0, top]
    if n_bands == 3:
        return [0.0, *BAND_EDGES_HZ, top]
    raise EvalError(f"bands must be 1 or 3, got {n_bands}")


def onset_bands(
    wave: np.ndarray, sample_rate: int, n_bands: int, per_bin_mean: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if len(wave) < N_FFT:
        raise EvalError("clip is shorter than one analysis window")
    frames = np.lib.stride_tricks.sliding_window_view(wave, N_FFT)[::HOP]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1))
    ## decibels relative to the clip's loudest bin, floored, so loudness does not matter
    db = np.maximum(20 * np.log10(spectrum / (spectrum.max() + 1e-12) + 1e-6), DB_FLOOR)
    rise = np.maximum(np.diff(db, axis=0), 0.0)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    edges = band_edges(n_bands, sample_rate)
    ## a sum favours wide bands; an average per bin lets a narrow low band (a kick) count as
    ## much as a wide high one (hats), which is what hit detection needs
    reduce = np.mean if per_bin_mean else np.sum
    flux = np.stack(
        [
            reduce(rise[:, (freqs >= lo) & (freqs < hi)], axis=1)
            for lo, hi in itertools.pairwise(edges)
        ]
    )
    ## keep only what rises above each band's typical level, which removes the noise floor
    flux = np.maximum(flux - np.median(flux, axis=1, keepdims=True), 0.0)
    times = (np.arange(1, len(frames)) * HOP + N_FFT / 2) / sample_rate
    return times, flux


def loop_envelope(
    times: np.ndarray, flux: np.ndarray, lead_in_s: float, loop_s: float
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    relative = times - lead_in_s
    inside = (relative >= 0) & (relative < loop_s)
    bins = np.minimum((relative[inside] / loop_s * N_BINS).astype(int), N_BINS - 1)
    envelope = np.zeros((flux.shape[0], N_BINS))
    for band in range(flux.shape[0]):
        np.maximum.at(envelope[band], bins, flux[band][inside])
    envelope = gaussian_filter1d(envelope, SMOOTH_BINS, axis=1, mode="wrap")
    envelope -= envelope.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(envelope, axis=1, keepdims=True)
    return np.divide(envelope, norms, out=np.zeros_like(envelope), where=norms > 1e-9)


def envelopes_for(
    paths: list[str],
    tempos: np.ndarray,
    durations: np.ndarray,
    beats: float,
    n_bands: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(paths), n_bands, N_BINS), dtype=np.float32)
    leads = np.zeros(len(paths))
    for i, path in enumerate(paths):
        leads[i] = lead_in_seconds(durations[i], tempos[i], beats)
        wave = load_mono(path, ANALYSIS_RATE)
        times, flux = onset_bands(wave, ANALYSIS_RATE, n_bands)
        out[i] = loop_envelope(times, flux, leads[i], beats * 60.0 / tempos[i])
        if i % 200 == 0 or i == len(paths) - 1:
            print(f"  {label} {i + 1}/{len(paths)}", flush=True)
    return out, leads


def lagged_scores(queries: np.ndarray, library: np.ndarray) -> np.ndarray:
    ## best correlation over small shifts, averaged over bands; both inputs (n, bands, bins)
    n_bands = queries.shape[1]
    max_lag = round(MAX_LAG_FRACTION * queries.shape[2])
    flat_queries = queries.reshape(len(queries), -1)
    best = np.full((len(queries), len(library)), -np.inf, dtype=np.float32)
    for lag in range(-max_lag, max_lag + 1):
        shifted = np.roll(library, lag, axis=2).reshape(len(library), -1)
        best = np.maximum(best, flat_queries @ shifted.T / n_bands)
    return best


def build_scores_file(pool: Pool, spec: MusicSpec, n_bands: int, out_path: Path) -> Path:
    paths = pool.library["audio_path"].tolist() + pool.queries["audio_path"].tolist()
    missing = missing_files(paths)
    if missing:
        raise EvalError(
            f"{len(missing)} of {len(paths)} audio files are missing, e.g. {missing[:3]}"
        )
    beats = loop_beats(spec)
    library_env, library_leads = envelopes_for(
        pool.library["audio_path"].tolist(),
        pool.library["tempo_bpm"].to_numpy(),
        pool.library["duration_s"].to_numpy(),
        beats,
        n_bands,
        "library",
    )
    query_env, query_leads = envelopes_for(
        pool.queries["audio_path"].tolist(),
        pool.queries["tempo_bpm"].to_numpy(),
        pool.queries["duration_s"].to_numpy(),
        beats,
        n_bands,
        "queries",
    )
    print(
        f"lead-in before the loop: library median {np.median(library_leads):.3f} s, "
        f"queries median {np.median(query_leads):.3f} s"
    )
    silent = int((np.abs(query_env).sum(axis=(1, 2)) == 0).sum())
    if silent:
        print(f"warning: {silent} queries have no onsets at all and will score 0 everywhere")
    scores = lagged_scores(query_env, library_env)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        query_ids=pool.queries["query_id"].to_numpy(dtype=str),
        library_ids=pool.library["clip_id"].to_numpy(dtype=str),
        scores=scores,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Onset-correlation baseline (B2).")
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "manifests/query_manifest.csv")
    parser.add_argument("--relevance", type=Path, default=REPO_ROOT / "manifests/relevance.json")
    parser.add_argument("--pool-kits", nargs="+", metavar="KIT_ID")
    parser.add_argument("--bands", type=int, choices=(1, 3), default=1)
    parser.add_argument("--out", type=Path, help="default depends on --bands")
    args = parser.parse_args()
    out = args.out or REPO_ROOT / (
        "results/embeddings/" + ("onset" if args.bands == 1 else "onset3band") + "_syn_beatbox.npz"
    )
    try:
        spec = load_spec()
        pool = load_pool(
            args.library, args.queries, args.relevance, pool_kits=args.pool_kits, timing=True
        )
        print(
            f"{len(pool.library)} library clips, {len(pool.queries)} queries, {args.bands} band(s)"
        )
        path = build_scores_file(pool, spec, args.bands, out)
    except (OSError, EvalError) as error:
        parser.exit(1, f"Onset baseline failed: {error}\n")
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()

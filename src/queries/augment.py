"""Seeded v0 query imperfections, with every realized choice recorded."""

from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio import frame_audio, normalize_audio
from src.music.spec import MusicSpec


def augment_query(
    trace: list[dict],
    sample_root: Path,
    *,
    sample_rate: int,
    duration_s: float,
    spec: MusicSpec,
    seed: int,
    normalization: dict,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Uniform offset/jitter amplitudes, Bernoulli drops, and Gaussian noise.

    The spec supplies all effect ranges. jitter_ms is a per-query bound, with
    each retained hit displaced uniformly within +/- that bound. Drift remains
    off. Noise is scaled using measured whole-clip RMS before normalization/fade.
    """
    if any(spec.sloppiness.tempo_drift_pct):
        raise ValueError("v0 does not implement tempo drift")
    if not trace:
        raise ValueError("No source events")
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(*spec.sloppiness.offset_ms))
    jitter = float(rng.uniform(*spec.sloppiness.jitter_ms))
    probabilities = {
        v: float(rng.uniform(*bounds)) for v, bounds in spec.sloppiness.drop_prob.items()
    }
    snr = float(rng.uniform(*spec.sloppiness.snr_db))
    audio = np.zeros(round(duration_s * sample_rate), dtype=np.float32)
    events = []
    for event in trace:
        dropped = bool(rng.random() < probabilities[event["voice"]])
        shift = float(rng.uniform(-jitter, jitter))
        start = event["frame"] + round((offset + shift) * sample_rate / 1000)
        entry = {
            **event,
            "clean_frame": event["frame"],
            "frame": start,
            "dropped": dropped,
            "jitter_ms": shift,
        }
        events.append(entry)
        if dropped:
            continue
        signal, sr = sf.read(sample_root / event["sample_path"], dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise ValueError("Sample rate mismatch")
        signal = signal.mean(axis=1)
        left, right = max(0, start), min(len(audio), start + len(signal))
        if right > left:
            audio[left:right] += signal[left - start : right - start]
    signal_rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
    if signal_rms == 0 or not np.isfinite(audio).all():
        raise ValueError("All events dropped/outside clip, or invalid samples")
    noise = rng.normal(size=len(audio))
    noise *= signal_rms / (10 ** (snr / 20) * np.sqrt(np.mean(noise**2)))
    measured_snr = float(20 * np.log10(signal_rms / np.sqrt(np.mean(noise**2))))
    mixed = normalize_audio(audio + noise, **normalization)
    mixed = frame_audio(
        mixed, sample_rate=sample_rate, duration_s=duration_s, settings=spec.loops.audio
    )
    parameters = {
        "offset_ms": offset,
        "jitter_ms": jitter,
        "drift_pct": 0,
        "drop_prob": float(np.mean([probabilities[e["voice"]] for e in trace])),
        "drop_prob_by_voice": probabilities,
        "snr_db": snr,
        "measured_snr_db_before_fade": measured_snr,
        "dropped_hits": sum(e["dropped"] for e in events),
    }
    return mixed, parameters, events

"""Audio front-ends (README §6): one clip's audio in, a (frames, features) array out.

    from src.models.config import load_model_config
    from src.models.features import compute_features, load_audio

    cfg = load_model_config()
    audio = load_audio("clip.wav", cfg.audio.sample_rate)  # mono float32 at 22,050 Hz
    r1 = compute_features("r1", audio, cfg)                # (frames, 3)
    r3 = compute_features("r3", audio, cfg)                # (frames, 64)

Every front-end uses the same frames: one every `audio.hop_length` samples (~10 ms),
centred, so frame t is centred on time t * hop_length / sample_rate, and a clip of n
samples has 1 + n // hop_length frames. Audio before the clip counts as silence.

- R1, 3-band onset envelopes: the energy in a low, mid, and high band (edges in
  `features.r1.band_edges_hz`), in dB, and how much it rises from one frame to the
  next (rises only, scaled so `onset_db_scale` dB reads as 1.0). The bands roughly
  follow kick, snare, and hats, but nothing relies on that.
- R3, log-mel: a mel spectrogram in dB relative to the clip's loudest bin, floored
  at `dynamic_range_db` below it and scaled to [0, 1].

Both are measured against the clip's own level, so the same clip played louder or
softer gives the same features. R2 (dualized) and R4 (MERT) come in a later step.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window, resample_poly

from src.models.config import ModelConfig

FRONT_ENDS = ("r1", "r3")


# --------------------------------------------------------------------------- audio


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Read an audio file as mono float32 at `sample_rate` (channels are averaged)."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), rate, sample_rate)


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Polyphase resampling between integer rates."""
    audio = np.asarray(audio, dtype=np.float32)
    if from_rate == to_rate:
        return audio
    g = math.gcd(int(from_rate), int(to_rate))
    return resample_poly(audio, to_rate // g, from_rate // g).astype(np.float32)


def n_frames(n_samples: int, hop_length: int) -> int:
    """Number of feature frames for a clip of n_samples."""
    return 1 + n_samples // hop_length


def power_spectrogram(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """|STFT|^2 with a periodic Hann window and centred frames: (frames, n_fft // 2 + 1)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio (1-D), got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("audio is empty")
    padded = np.pad(audio, n_fft // 2)  # zeros: before the clip is silence
    frames = sliding_window_view(padded, n_fft)[::hop_length]
    spectrum = np.fft.rfft(frames * get_window("hann", n_fft).astype(np.float32), axis=1)
    return (spectrum.real**2 + spectrum.imag**2).astype(np.float32)


def _relative_db(power: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """10 log10(power) relative to the array's maximum, floored at -dynamic_range_db.

    A silent clip gives all -dynamic_range_db.
    """
    ref = float(power.max())
    if ref <= 0:
        return np.full(power.shape, -dynamic_range_db, dtype=np.float32)
    floor = ref * 10.0 ** (-dynamic_range_db / 10.0)
    return (10.0 * np.log10(np.maximum(power, floor) / ref)).astype(np.float32)


# --------------------------------------------------------------------------- R1


def band_energies(power: np.ndarray, sample_rate: int, n_fft: int, edges_hz) -> np.ndarray:
    """Sum of power per band: (frames, len(edges_hz) + 1), bands from 0 Hz to Nyquist."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bounds = [0.0, *edges_hz, math.inf]
    cols = [power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in pairwise(bounds)]
    return np.stack(cols, axis=1)


def r1_onset_bands(audio: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """R1: per-band onset strength, (frames, n_bands), >= 0."""
    a, c = cfg.audio, cfg.r1
    bands = band_energies(
        power_spectrogram(audio, a.n_fft, a.hop_length), a.sample_rate, a.n_fft, c.band_edges_hz
    )
    db = _relative_db(bands, c.dynamic_range_db)
    # Before the clip is silence, so a hit right on the first frame still reads as a rise.
    before = np.full((1, db.shape[1]), -c.dynamic_range_db, dtype=np.float32)
    rise = np.diff(db, axis=0, prepend=before)
    return (np.maximum(rise, 0.0) / c.onset_db_scale).astype(np.float32)


# --------------------------------------------------------------------------- R3


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin_hz: float, fmax_hz: float
) -> np.ndarray:
    """Triangular filters on the HTK mel scale, peak 1: (n_mels, n_fft // 2 + 1)."""

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    points = mel_to_hz(np.linspace(hz_to_mel(fmin_hz), hz_to_mel(fmax_hz), n_mels + 2))
    lower, centre, upper = points[:-2, None], points[1:-1, None], points[2:, None]
    rising = (freqs[None, :] - lower) / (centre - lower)
    falling = (upper - freqs[None, :]) / (upper - centre)
    bank = np.maximum(0.0, np.minimum(rising, falling)).astype(np.float32)
    empty = np.flatnonzero(bank.sum(axis=1) == 0)
    if empty.size:
        raise ValueError(
            f"mel filters {empty.tolist()} cover no STFT bin; use fewer mels or a larger n_fft"
        )
    return bank


def r3_logmel(audio: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """R3: log-mel spectrogram scaled to [0, 1], (frames, n_mels)."""
    a, c = cfg.audio, cfg.r3
    bank = mel_filterbank(a.sample_rate, a.n_fft, c.n_mels, c.fmin_hz, c.fmax_hz)
    mel = power_spectrogram(audio, a.n_fft, a.hop_length) @ bank.T
    return (_relative_db(mel, c.dynamic_range_db) / c.dynamic_range_db + 1.0).astype(np.float32)


# --------------------------------------------------------------------------- dispatch


def compute_features(name: str, audio: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """Features for one clip (mono, at cfg.audio.sample_rate) from front-end `name`."""
    if name == "r1":
        return r1_onset_bands(audio, cfg)
    if name == "r3":
        return r3_logmel(audio, cfg)
    raise ValueError(f"unknown front-end {name!r}; built so far: {FRONT_ENDS}")


def feature_dim(name: str, cfg: ModelConfig) -> int:
    """Number of features per frame for front-end `name`."""
    if name == "r1":
        return len(cfg.r1.band_edges_hz) + 1
    if name == "r3":
        return cfg.r3.n_mels
    raise ValueError(f"unknown front-end {name!r}; built so far: {FRONT_ENDS}")

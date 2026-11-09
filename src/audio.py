"""Shared audio framing and level control for Member B's dataset pipeline."""

import numpy as np

from src.music.spec import ClipAudio


def frame_audio(
    audio: np.ndarray, *, sample_rate: int, duration_s: float, settings: ClipAudio
) -> np.ndarray:
    """Crop/pad to lead-in + pattern duration and fade inside the ending.

    duration_s already includes the spec's lead-in. No new audio is appended by
    the fade. The caller must place events relative to that same lead-in.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError("Audio must be finite mono or multi-channel data")
    if sample_rate <= 0 or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Invalid rate or duration")
    frames = round(duration_s * sample_rate)
    if settings.end == "ring":
        frames = max(frames, len(x))
    result = np.zeros((frames,) + x.shape[1:], dtype=np.float32)
    count = min(frames, len(x))
    result[:count] = x[:count]
    if settings.end == "cut":
        fade = min(frames, round(settings.end_fade_ms * sample_rate / 1000))
        if fade:
            curve = np.linspace(1, 0, fade) if fade > 1 else np.zeros(1)
            result[-fade:] *= curve if x.ndim == 1 else curve[:, None]
    return result


def normalize_audio(
    audio: np.ndarray, *, target_rms_dbfs: float, peak_ceiling_dbfs: float
) -> np.ndarray:
    """One gain for the entire clip; peak safety takes priority over RMS target."""
    x = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0
    if not np.isfinite(x).all() or rms == 0:
        raise ValueError("Cannot normalize silent or nonfinite audio")
    gain = min(
        10 ** (target_rms_dbfs / 20) / rms, 10 ** (peak_ceiling_dbfs / 20) / float(np.abs(x).max())
    )
    return (x * gain).astype(np.float32)

"""Mix one-shot vocal samples with the music spec's single-mouth choke rule."""

import numpy as np

from src.music.spec import Beatbox


def mix_placements(
    placements: list[tuple[int, np.ndarray]],
    *,
    frames: int,
    sample_rate: int,
    settings: Beatbox,
) -> tuple[np.ndarray, list[dict]]:
    """Stop each sample at the next audible placement, after timing and drops.

    Placements contain retained hits only. Sort by actual frame, with input order
    breaking ties. Empty/out-of-window hits cannot choke an audible hit. Fade
    inside the interrupted tail; never stretch a naturally shorter sample.
    Metadata is returned in input order, including non-playing placements.
    """
    if frames < 1 or sample_rate < 1:
        raise ValueError("Positive frame count and sample rate required")
    audio = np.zeros(frames, dtype=np.float32)
    details = [
        dict(
            played=False,
            audio_start_frame=None,
            audio_end_frame=None,
            next_played_frame=None,
            choked_at_next_sound=False,
            choke_fade_frames=0,
        )
        for _ in placements
    ]
    audible = []
    for i, (start, source) in enumerate(placements):
        signal = np.asarray(source, dtype=np.float32)
        if signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError("Vocal samples must be finite mono arrays")
        if len(signal) and start < frames and start + len(signal) > 0:
            audible.append((start, i, signal))
    audible.sort(key=lambda item: (item[0], item[1]))
    fade_length = round(settings.sound_end.fade_ms * sample_rate / 1000)
    for position, (start, i, signal) in enumerate(audible):
        next_frame = audible[position + 1][0] if position + 1 < len(audible) else None
        natural_end = start + len(signal)
        end = min(frames, natural_end)
        choked = bool(settings.choke_at_next_sound and next_frame is not None and next_frame < end)
        if choked:
            end = next_frame
        left = max(0, start)
        if end <= left:
            continue  # Same-time earlier event is fully superseded.
        part = signal[left - start : end - start].copy()
        fade = min(len(part), fade_length) if choked else 0
        if fade:
            part[-fade:] *= np.linspace(1, 0, fade) if fade > 1 else np.zeros(1)
        audio[left:end] += part
        details[i] = dict(
            played=True,
            audio_start_frame=left,
            audio_end_frame=end,
            next_played_frame=next_frame,
            choked_at_next_sound=choked,
            choke_fade_frames=fade,
        )
    return audio, details

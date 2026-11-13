"""Extract AVP one-shot sounds and a browser audition page using the music spec.

Run from the IDE or with ``python -m src.queries.extract_avp``.
Generated audio and catalogs stay under gitignored results/avp_samples/.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.music.spec import SoundEnd, load_spec
from src.queries.index_avp import build_index

REPO_ROOT = Path(__file__).resolve().parents[2]
# Measurement resolution, following tools.inventory.avp's decay measurement.
# This is an analysis window, not an extra sample-duration/music threshold.
ENVELOPE_FRAME_S = 0.010


@dataclass(frozen=True)
class SoundCut:
    start_frame: int
    end_frame: int  # exclusive, in the original recording
    stop_reason: str
    fade_frames: int


def cut_sound(
    audio: np.ndarray,
    *,
    sample_rate: int,
    onset_s: float,
    next_onset_s: float | None,
    settings: SoundEnd,
) -> tuple[np.ndarray, SoundCut]:
    """Cut without crossing the next annotation, duration cap, or recording end.

    Measure a channel-pooled RMS envelope in 10 ms frames, including a partial
    final frame. After the peak envelope frame, the first frame at/below the
    configured relative level marks decay completion. Fade the retained tail
    inside the chosen boundary; never extend into a neighbouring sound.
    """
    x = np.asarray(audio)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Audio must be nonempty, finite, and mono or multi-channel")
    if sample_rate <= 0 or not math.isfinite(onset_s) or onset_s < 0:
        raise ValueError("Invalid sample rate or onset")
    # Ceil gives the first sample at/after an annotation and never crosses a cap.
    start = math.ceil(onset_s * sample_rate)
    if start >= len(x):
        raise ValueError("Onset is outside the recording")
    max_frames = max(1, math.floor(settings.max_ms * sample_rate / 1000))
    bounds = [(len(x), "recording_end"), (start + max_frames, "max_duration")]
    if settings.stop_at_next_onset and next_onset_s is not None:
        if not math.isfinite(next_onset_s) or next_onset_s <= onset_s:
            raise ValueError("Next onset must be finite and later than this onset")
        bounds.append((math.ceil(next_onset_s * sample_rate), "next_onset"))
    end, reason = min(bounds, key=lambda item: item[0])
    segment = x[start:end].astype(np.float64, copy=True)
    if not len(segment) or not np.any(segment):
        raise ValueError("Silent or empty sound")
    width = max(1, round(ENVELOPE_FRAME_S * sample_rate))
    energy = np.mean(segment**2, axis=1)
    offsets = np.arange(0, len(segment), width)
    sizes = np.minimum(width, len(segment) - offsets)
    rms = np.sqrt(np.add.reduceat(energy, offsets) / sizes)
    peak = int(np.argmax(rms))
    threshold = rms[peak] * 10 ** (-settings.below_peak_db / 20)
    crossings = np.flatnonzero(rms[peak + 1 :] <= threshold)
    if len(crossings):
        decay_end = (peak + 1 + int(crossings[0])) * width
        if decay_end < len(segment):
            segment = segment[:decay_end]
            end = start + decay_end
            reason = "decay"
    fade = min(len(segment), round(settings.fade_ms * sample_rate / 1000))
    if fade:
        # A one-frame sound is faded to zero and will be rejected as unusable.
        curve = np.linspace(1.0, 0.0, fade) if fade > 1 else np.zeros(1)
        segment[-fade:] *= curve[:, None]
    if not np.any(segment):
        raise ValueError("Sound is silent after fading")
    return segment.astype(np.float32), SoundCut(start, end, reason, fade)


def waveform_svg(audio: np.ndarray) -> str:
    """A compact min/max waveform that retains brief transients in the preview."""
    x = np.max(np.abs(audio), axis=1)
    peak = float(x.max()) or 1.0
    buckets = np.array_split(x, min(240, len(x)))
    lines = []
    for i, bucket in enumerate(buckets):
        xpos = (i + 0.5) * 480 / len(buckets)
        height = float(bucket.max()) / peak * 32
        lines.append(f'<path d="M{xpos:.1f},{36 - height:.1f}V{36 + height:.1f}"/>')
    return (
        '<svg viewBox="0 0 480 72" role="img" aria-label="Extracted sound waveform" '
        'style="width:100%;max-width:480px"><g stroke="#176b91">' + "".join(lines) + "</g></svg>"
    )


def extract_samples(root: Path, out_dir: Path) -> dict:
    """Refresh the onset index, apply approved source selection, write samples.

    Next-onset limits use all finite source annotations, including invalid-label
    rows excluded from the index. Output paths are deterministic and relative
    to out_dir; participant identities are retained without assigning splits.
    """
    root, out_dir = root.resolve(), out_dir.resolve()
    spec = load_spec()
    index_summary = build_index(root, out_dir / "index")
    index = pd.read_csv(out_dir / "index/utterance_index.csv")
    recording_type = np.where(index.kind.eq("Improvisation"), "improvisation", "isolated")
    chosen = index[
        index["mode"].str.lower().isin(spec.beatbox.avp_modes)
        & pd.Series(recording_type, index=index.index).isin(spec.beatbox.avp_recordings)
        & index.label.isin(spec.beatbox.avp_labels)
    ]
    rows, issues, previews = [], [], []
    preview_groups = set()
    for relative_path, group in chosen.groupby("audio_path", sort=True):
        source = root / relative_path
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        # Preserve unlabeled onsets as boundaries instead of skipping over them.
        raw = pd.read_csv(source.with_suffix(".csv"), header=None)
        onsets = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy()
        onsets = np.unique(onsets[np.isfinite(onsets) & (onsets >= 0)])
        for row in group.itertuples(index=False):
            following = onsets[onsets > row.onset_s]
            next_onset = float(following[0]) if len(following) else None
            try:
                sample, cut = cut_sound(
                    audio,
                    sample_rate=sr,
                    onset_s=row.onset_s,
                    next_onset_s=next_onset,
                    settings=spec.beatbox.sound_end,
                )
            except ValueError as error:
                issues.append({"utterance_id": row.utterance_id, "reason": str(error)})
                continue
            relative_sample = (
                Path("samples")
                / Path(relative_path).parent
                / (f"{source.stem}_row{row.annotation_row:04d}.wav")
            )
            target = out_dir / relative_sample
            target.parent.mkdir(parents=True, exist_ok=True)
            # Float WAV preserves source amplitude without PCM quantization/clipping.
            sf.write(target, sample, sr, subtype="FLOAT")
            record = {
                "utterance_id": row.utterance_id,
                "participant_id": row.participant_id,
                "mode": row.mode,
                "kind": row.kind,
                "label": row.label,
                "sound_class": row.sound_class,
                "source_audio_path": relative_path,
                "annotation_row": row.annotation_row,
                "onset_s": row.onset_s,
                "next_onset_s": next_onset,
                "sample_path": relative_sample.as_posix(),
                "sample_rate": sr,
                "channels": sample.shape[1],
                "frames": len(sample),
                "duration_s": len(sample) / sr,
                **asdict(cut),
            }
            rows.append(record)
            # One preview per mode, recording category, and source class (up to 16).
            category = "improvisation" if row.kind == "Improvisation" else "isolated"
            preview_key = (row.mode, category, row.label)
            if preview_key not in preview_groups:
                preview_groups.add(preview_key)
                title = html.escape(
                    f"{row.participant_id} · {row.mode} · {category} · {row.sound_class}"
                )
                previews.append(
                    f"<section><h2>{title}</h2><p>{len(sample) / sr:.3f} seconds; "
                    f"stop: {cut.stop_reason}; source row {row.annotation_row}</p>"
                    + waveform_svg(sample)
                    + '<br><audio controls preload="none" '
                    + f'src="{relative_sample.as_posix()}"></audio>'
                    + "</section>"
                )
    columns = [
        "utterance_id",
        "participant_id",
        "mode",
        "kind",
        "label",
        "sound_class",
        "source_audio_path",
        "annotation_row",
        "onset_s",
        "next_onset_s",
        "sample_path",
        "sample_rate",
        "channels",
        "frames",
        "duration_s",
        "start_frame",
        "end_frame",
        "stop_reason",
        "fade_frames",
    ]
    catalog = pd.DataFrame(rows, columns=columns)
    summary = {
        "spec_version": spec.version,
        "source_root": str(root),
        "indexed_annotations": index_summary["indexed_sounds"],
        "selected_annotations": len(chosen),
        "extracted_samples": len(rows),
        "skipped_samples": len(issues),
        "participants": int(catalog.participant_id.nunique()),
        "sound_end": asdict(spec.beatbox.sound_end),
        "envelope_frame_ms": ENVELOPE_FRAME_S * 1000,
        "stop_counts": catalog.stop_reason.value_counts().to_dict(),
        "split_status": "unassigned; listening samples only, not approved training data",
        "fade_policy": "fade within retained tail; do not extend beyond endpoint",
        "preview_count": len(previews),
    }
    catalog.to_csv(out_dir / "sample_catalog.csv", index=False)
    pd.DataFrame(issues, columns=["utterance_id", "reason"]).to_csv(
        out_dir / "extraction_issues.csv", index=False
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "preview.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>AVP extracted sounds</title><style>"
        "body{font:16px system-ui;max-width:850px;margin:40px auto;padding:0 20px;line-height:1.5}"
        "section{border-top:1px solid #ddd;padding:16px 0}h2{font-size:18px}"
        "</style><h1>Extracted AVP sounds</h1><p>Press play to audition individual mouth sounds. "
        "Waveforms are scaled per sound for visibility; audio keeps its original level. "
        "Check for cut-off attacks, missing decay, or a neighbouring sound. "
        "Joshua’s listening approval and participant splits are still pending.</p>"
        f"<p>{len(rows):,} samples extracted; {len(issues)} skipped. "
        '<a href="sample_catalog.csv">Sample catalog</a> · '
        '<a href="summary.json">Extraction summary</a></p>' + "".join(previews) + "</html>"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avp-dir", type=Path, default=REPO_ROOT / "data/raw/avp/AVP_Dataset")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/avp_samples")
    args = parser.parse_args()
    summary = extract_samples(args.avp_dir, args.out)
    print(json.dumps(summary, indent=2))
    print(f"Listen: {(args.out / 'preview.html').resolve()}")


if __name__ == "__main__":
    main()

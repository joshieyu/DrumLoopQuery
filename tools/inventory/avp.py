"""AVP inventory: file and utterance tables plus simple per-utterance audio measures.

Layout (checked on v4): AVP_Dataset/{Personal,Fixed}/Participant_<n>/P<n>_<Kind>_<Mode>.wav
with a header-less CSV next to each wav: onset_sec, label[, onset_phoneme, coda_phoneme].
Phoneme columns exist only in the Personal subset.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

LABELS = ["kd", "sd", "hhc", "hho"]
LABEL_NAMES = {"kd": "kick", "sd": "snare", "hhc": "closed hat", "hho": "open hat"}
KINDS = ["Kick", "Snare", "HHclosed", "HHopened", "Improvisation"]

_NAME = re.compile(r"^P(?P<participant>\d+)_(?P<kind>[A-Za-z]+)_(?P<mode>Personal|Fixed)\.wav$")


def load_avp(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (files, utterances). `root` is the AVP_Dataset folder."""
    files, utts = [], []
    for wav in sorted(root.glob("*/Participant_*/*.wav")):
        m = _NAME.match(wav.name)
        if not m:
            continue
        info = sf.info(wav)
        rec = {
            "participant": int(m["participant"]),
            "mode": m["mode"],
            "kind": m["kind"],
            "path": wav,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "duration_s": info.duration,
        }
        files.append(rec)
        csv = wav.with_suffix(".csv")
        if not csv.exists():
            continue
        d = pd.read_csv(csv, header=None, dtype={1: str})
        for row in d.itertuples(index=False):
            utts.append(
                {
                    "participant": rec["participant"],
                    "mode": rec["mode"],
                    "kind": rec["kind"],
                    "onset_s": float(row[0]),
                    "label_raw": row[1],
                    "label": str(row[1]).strip(),
                    "onset_phoneme": row[2] if len(row) > 2 else np.nan,
                    "coda_phoneme": row[3] if len(row) > 3 else np.nan,
                }
            )
    return pd.DataFrame(files), pd.DataFrame(utts)


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(x, 1e-10))


def measure_utterances(
    files: pd.DataFrame, utts: pd.DataFrame, max_len_s: float = 1.0, decay_db: float = 30.0
) -> pd.DataFrame:
    """Add per-utterance level, decay length, and spectral centroid columns.

    Each utterance runs from its onset to the next onset (capped at `max_len_s`).
    decay_ms: time from onset until the 10 ms RMS envelope last sits within
    `decay_db` of its peak. centroid_hz: spectral centroid of the first 50 ms.
    """
    out = []
    for f in files.itertuples(index=False):
        sel = utts[
            (utts["participant"] == f.participant)
            & (utts["mode"] == f.mode)
            & (utts["kind"] == f.kind)
        ].sort_values("onset_s")
        if sel.empty:
            continue
        audio, sr = sf.read(f.path, always_2d=True)
        x = audio.mean(axis=1)
        onsets = sel["onset_s"].to_numpy()
        ends = np.append(onsets[1:], len(x) / sr)
        frame = int(0.010 * sr)
        n50 = int(0.050 * sr)
        window = np.hanning(n50)
        freqs = np.fft.rfftfreq(n50, 1 / sr)
        for (idx, _), t0, t1 in zip(sel.iterrows(), onsets, ends, strict=True):
            a = int(t0 * sr)
            b = min(int(min(t1, t0 + max_len_s) * sr), len(x))
            seg = x[a:b]
            if len(seg) < frame:
                out.append({"index": idx})
                continue
            n_frames = len(seg) // frame
            rms = np.sqrt((seg[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1))
            env_db = _db(rms)
            above = np.flatnonzero(env_db >= env_db.max() - decay_db)
            head = seg[:n50] if len(seg) >= n50 else np.pad(seg, (0, n50 - len(seg)))
            spec = np.abs(np.fft.rfft(head * window))
            out.append(
                {
                    "index": idx,
                    "peak_dbfs": _db(np.abs(seg).max()),
                    "rms100_dbfs": _db(np.sqrt((seg[: int(0.1 * sr)] ** 2).mean())),
                    "decay_ms": (above.max() + 1) * 10.0,
                    "segment_ms": 1000 * len(seg) / sr,
                    "centroid_hz": float((freqs * spec).sum() / max(spec.sum(), 1e-12)),
                }
            )
    m = pd.DataFrame(out).set_index("index")
    return utts.join(m)


def file_noise_floor(files: pd.DataFrame, quantile: float = 0.1) -> pd.DataFrame:
    """Per file: 10 ms RMS frames at the given quantile (a noise-floor estimate) and the
    loudest frame, so their difference approximates the recording's SNR. Frames of exact
    digital silence are left out of the floor and counted separately."""
    rows = []
    for f in files.itertuples(index=False):
        audio, sr = sf.read(f.path, always_2d=True)
        x = audio.mean(axis=1)
        frame = int(0.010 * sr)
        n = len(x) // frame
        rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        live = rms[rms > 0]
        rows.append(
            {
                "participant": f.participant,
                "mode": f.mode,
                "kind": f.kind,
                "floor_dbfs": _db(np.quantile(live, quantile)) if len(live) else np.nan,
                "loudest_dbfs": _db(rms.max()),
                "zero_sample_pct": 100 * float((x == 0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["peak_to_floor_db"] = out["loudest_dbfs"] - out["floor_dbfs"]
    return out


def noise_sample_stats(path: Path) -> dict:
    audio, sr = sf.read(path, always_2d=True)
    return {
        "file": path.name,
        "samplerate": sr,
        "channels": audio.shape[1],
        "duration_s": len(audio) / sr,
        "rms_dbfs": float(_db(np.sqrt((audio**2).mean()))),
        "peak_dbfs": float(_db(np.abs(audio).max())),
    }


def improvisation_mix(utts: pd.DataFrame) -> pd.DataFrame:
    """Share of each label in each improvisation, per participant and mode."""
    imp = utts[utts["kind"] == "Improvisation"]
    counts = imp.groupby(["mode", "participant", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=LABELS, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    return counts.join(share.add_prefix("share_"))


def repetition_gaps(utts: pd.DataFrame) -> pd.DataFrame:
    """Gaps between consecutive onsets within each file, in ms."""
    u = utts.sort_values(["participant", "mode", "kind", "onset_s"])
    same = (
        (u["participant"] == u["participant"].shift())
        & (u["mode"] == u["mode"].shift())
        & (u["kind"] == u["kind"].shift())
    )
    gaps = (u["onset_s"] - u["onset_s"].shift()) * 1000
    return pd.DataFrame({"kind": u["kind"][same], "mode": u["mode"][same], "gap_ms": gaps[same]})

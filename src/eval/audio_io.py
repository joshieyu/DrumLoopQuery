from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src.eval.evaluate import EvalError

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    ## manifest paths are relative to the repo root
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def missing_files(paths: Iterable[str | Path]) -> list[str]:
    return [str(p) for p in paths if not resolve(p).is_file()]


def load_mono(path: str | Path, target_sr: int) -> np.ndarray:
    ## imported here so the rest of the package works without these libraries
    import soundfile as sf
    from scipy.signal import resample_poly

    resolved = resolve(path)
    if not resolved.is_file():
        raise EvalError(f"audio file not found: {path}")
    data, sr = sf.read(resolved, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // divisor, sr // divisor).astype(np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise EvalError(f"audio file is empty or has NaN values: {path}")
    return mono

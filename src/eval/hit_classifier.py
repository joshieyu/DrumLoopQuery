from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.audio_io import load_mono
from src.eval.evaluate import EvalError
from src.eval.onset_baseline import ANALYSIS_RATE
from src.eval.relevance import read_manifest

N_FFT = 512
FRAME_HOP = 128
N_MELS = 24
FMIN_HZ = 30.0
FMAX_HZ = 10_000.0
## the window around a hit, in seconds relative to its onset: a short attack and a tail
PRE_S = -0.01
ATTACK_END_S = 0.06
POST_S = 0.15
## training sounds are also used a little early and late, because the detector is not exact
SHIFTS_MS = (-15, 0, 15)
## the AVP labels as voices; this is spec section 1 (open and closed hats are one voice)
LABEL_TO_VOICE = {"kd": "kick", "sd": "snare", "hhc": "hats", "hho": "hats"}
## a hit's features stop this long before the next detected hit starts
NEXT_HIT_GUARD_S = 0.005
## bands further than this below the loudest one are not told apart, so a noise floor in an
## empty band cannot dominate the features
DYNAMIC_RANGE_DB = 60.0
## training sounds are also used with white noise added, as the query generator adds it
NOISE_SNRS_DB = (None, 30.0, 20.0, 10.0)


@lru_cache(maxsize=4)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float):
    def to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def to_hz(mel):
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bank = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (freqs - lo) / (mid - lo)
        falling = (hi - freqs) / (hi - mid)
        bank[m] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def hit_features(
    wave: np.ndarray, sample_rate: int, onset_s: float, end_limit_s: float | None = None
) -> np.ndarray:
    ## 2 x N_MELS numbers: the spectrum of the attack and of the tail, in decibels, with the
    ## overall loudness of the attack removed so soft and loud hits look alike
    start = round((onset_s + PRE_S) * sample_rate)
    length = round((POST_S - PRE_S) * sample_rate)
    segment = np.zeros(length)
    lo, hi = max(start, 0), min(start + length, len(wave))
    if hi > lo:
        segment[lo - start : hi - start] = wave[lo:hi]
    if end_limit_s is not None:
        cut = round(end_limit_s * sample_rate) - start
        if cut < length:
            segment[max(cut, 0) :] = 0.0
    frames = np.lib.stride_tricks.sliding_window_view(segment, N_FFT)[::FRAME_HOP]
    power = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1)) ** 2
    bank = mel_filterbank(sample_rate, N_FFT, N_MELS, FMIN_HZ, FMAX_HZ)
    decibels = 10.0 * np.log10(power @ bank.T + 1e-10)
    decibels = np.maximum(decibels, decibels.max() - DYNAMIC_RANGE_DB)
    centers = (np.arange(len(frames)) * FRAME_HOP + N_FFT / 2) / sample_rate
    in_attack = centers < (ATTACK_END_S - PRE_S)
    attack = decibels[in_attack].mean(axis=0)
    tail = decibels[~in_attack].mean(axis=0)
    level = attack.mean()
    return np.concatenate([attack - level, tail - level])


def add_white_noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(wave, dtype=np.float64)))
    return wave + rng.normal(0.0, rms * 10.0 ** (-snr_db / 20.0), len(wave)).astype(wave.dtype)


class HitClassifier:
    def __init__(self, classes, weights, bias, mean, std):
        self.classes = tuple(classes)
        self.weights = weights
        self.bias = bias
        self.mean = mean
        self.std = std

    @classmethod
    def fit(
        cls, features: np.ndarray, labels: np.ndarray, classes: Sequence[str], l2: float = 1e-2
    ):
        from scipy.optimize import minimize

        n, d = features.shape
        k = len(classes)
        counts = np.bincount(labels, minlength=k)
        if (counts == 0).any():
            missing = [c for c, n_c in zip(classes, counts, strict=True) if n_c == 0]
            raise EvalError(f"no training examples for {missing}")
        mean, std = features.mean(axis=0), features.std(axis=0) + 1e-8
        z = (features - mean) / std
        ## every class counts equally, however many examples it has
        weight = (n / (k * counts))[labels]
        onehot = np.eye(k)[labels]

        def objective(theta):
            w = theta[: k * d].reshape(k, d)
            b = theta[k * d :]
            logits = z @ w.T + b
            logits -= logits.max(axis=1, keepdims=True)
            log_p = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            loss = -(weight * (onehot * log_p).sum(axis=1)).sum() / n + l2 * (w**2).sum()
            grad = (np.exp(log_p) - onehot) * weight[:, None] / n
            return loss, np.concatenate([(grad.T @ z + 2 * l2 * w).ravel(), grad.sum(axis=0)])

        result = minimize(
            objective, np.zeros(k * d + k), jac=True, method="L-BFGS-B", options={"maxiter": 500}
        )
        theta = result.x
        return cls(classes, theta[: k * d].reshape(k, d), theta[k * d :], mean, std)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = ((features - self.mean) / self.std) @ self.weights.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            classes=np.array(self.classes),
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            meta=np.array(json.dumps(meta or {})),
        )

    @classmethod
    def load(cls, path: Path) -> HitClassifier:
        with np.load(path, allow_pickle=False) as f:
            return cls([str(c) for c in f["classes"]], f["weights"], f["bias"], f["mean"], f["std"])


def training_examples(
    index: pd.DataFrame,
    avp_root: Path,
    participants: set[str],
    voices: Sequence[str],
    shifts_ms: Sequence[int] = SHIFTS_MS,
    noise_snrs_db: Sequence[float | None] = NOISE_SNRS_DB,
) -> tuple[np.ndarray, np.ndarray]:
    ## only the listed participants' recordings are opened, so a test participant's audio is
    ## never read here
    voice_row = {voice: i for i, voice in enumerate(voices)}
    rows = index[index["participant_id"].isin(participants) & index["label"].isin(LABEL_TO_VOICE)]
    features, labels = [], []
    for audio_path, group in rows.groupby("audio_path", sort=True):
        clean = load_mono(avp_root / audio_path, ANALYSIS_RATE)
        group = group.sort_values("onset_s")
        onsets = group["onset_s"].to_numpy(dtype=float)
        limits = np.append(onsets[1:] - NEXT_HIT_GUARD_S, np.inf)
        rng = np.random.default_rng(0)
        for snr in noise_snrs_db:
            wave = clean if snr is None else add_white_noise(clean, snr, rng)
            for onset, limit, label in zip(onsets, limits, group["label"], strict=True):
                for shift in shifts_ms:
                    features.append(
                        hit_features(
                            wave,
                            ANALYSIS_RATE,
                            onset + shift / 1000.0,
                            None if np.isinf(limit) else limit,
                        )
                    )
                    labels.append(voice_row[LABEL_TO_VOICE[label]])
    if not features:
        raise EvalError("no training sounds found for those participants")
    return np.stack(features), np.array(labels, dtype=int)


def train_classifier(
    index_path: Path,
    avp_root: Path,
    splits_path: Path,
    voices: Sequence[str],
    out_path: Path,
) -> dict:
    index = pd.read_csv(index_path)
    needed = {"participant_id", "label", "audio_path", "onset_s"}
    if not needed <= set(index.columns):
        raise EvalError(f"{index_path}: needs columns {sorted(needed)}")
    splits = read_manifest(splits_path, ("participant_id", "split"))
    train = set(splits.loc[splits["split"] == "train", "participant_id"])
    validation = set(splits.loc[splits["split"] == "validation", "participant_id"])
    if not train or not validation:
        raise EvalError("the participant split file needs train and validation participants")
    x_train, y_train = training_examples(index, avp_root, train, voices)
    x_val, y_val = training_examples(
        index, avp_root, validation, voices, shifts_ms=(0,), noise_snrs_db=(None,)
    )
    x_noisy, y_noisy = training_examples(
        index, avp_root, validation, voices, shifts_ms=(0,), noise_snrs_db=(20.0,)
    )
    classifier = HitClassifier.fit(x_train, y_train, voices)
    predicted = classifier.predict(x_val)
    k = len(voices)
    confusion = np.zeros((k, k), dtype=int)
    for truth, guess in zip(y_val, predicted, strict=True):
        confusion[truth, guess] += 1
    metrics = {
        "n_train_examples": len(y_train),
        "n_validation_sounds": len(y_val),
        "validation_accuracy": float((predicted == y_val).mean()),
        "validation_accuracy_with_20db_noise": float(
            (classifier.predict(x_noisy) == y_noisy).mean()
        ),
        "validation_confusion_rows_true_cols_predicted": confusion.tolist(),
        "classes": list(voices),
        "train_participants": len(train),
        "validation_participants": len(validation),
    }
    classifier.save(out_path, metrics)
    return metrics

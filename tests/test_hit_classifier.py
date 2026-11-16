from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.evaluate import EvalError
from src.eval.hit_classifier import (
    N_MELS,
    NOISE_SNRS_DB,
    HitClassifier,
    add_white_noise,
    hit_features,
    mel_filterbank,
    train_classifier,
    training_examples,
)

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

RATE = 22_050
VOICES = ("kick", "snare", "hats")


def _raw_sound(kind, rng, rate=RATE):
    n = int(0.15 * rate)
    t = np.arange(n) / rate
    if kind == "kick":
        freq = rng.uniform(50, 90)
        return np.sin(2 * np.pi * freq * t * (1 + 1.5 * np.exp(-t / 0.02))) * np.exp(
            -t / rng.uniform(0.05, 0.09)
        )
    if kind == "snare":
        noise = rng.normal(0, 1, n) * np.exp(-t / rng.uniform(0.04, 0.07))
        return noise + 0.6 * np.sin(2 * np.pi * rng.uniform(180, 260) * t) * np.exp(-t / 0.05)
    return np.diff(rng.normal(0, 1, n + 1)) * np.exp(-t / rng.uniform(0.015, 0.03))


def make_sound(kind, rng, rate=RATE):
    ## every sound ends with a 10 ms fade, as the query generator's sounds do
    sound = _raw_sound(kind, rng, rate)
    fade = int(0.01 * rate)
    sound[-fade:] *= np.linspace(1.0, 0.0, fade)
    return sound


def place(wave, sound, onset_s, rate=RATE, amplitude=1.0):
    start = round(onset_s * rate)
    wave[start : start + len(sound)] += sound[: len(wave) - start] * amplitude


def _dataset(seed, per_class=120):
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for row, kind in enumerate(VOICES):
        for _ in range(per_class):
            wave = np.zeros(RATE)
            place(wave, make_sound(kind, rng), 0.3, amplitude=rng.uniform(0.3, 1.0))
            features.append(hit_features(wave, RATE, 0.3))
            labels.append(row)
    return np.stack(features), np.array(labels)


def test_mel_filterbank_covers_the_spectrum():
    bank = mel_filterbank(RATE, 512, N_MELS, 30.0, 10_000.0)
    assert bank.shape == (N_MELS, 257)
    assert (bank >= 0).all()
    assert (bank.sum(axis=1) > 0).all()


def test_features_have_a_fixed_size_and_ignore_loudness():
    rng = np.random.default_rng(0)
    wave = np.zeros(RATE)
    place(wave, make_sound("snare", rng), 0.3)
    base = hit_features(wave, RATE, 0.3)
    assert base.shape == (2 * N_MELS,)
    assert np.allclose(base, hit_features(wave * 10, RATE, 0.3), atol=1e-3)
    assert np.array_equal(base, hit_features(wave, RATE, 0.3))


def test_silence_gives_zero_features_not_nan():
    features = hit_features(np.zeros(RATE), RATE, 0.3)
    assert np.isfinite(features).all()
    assert not features.any()


def test_a_following_hit_is_cut_off_at_the_limit():
    rng = np.random.default_rng(1)
    alone = np.zeros(RATE)
    place(alone, make_sound("kick", rng), 0.3)
    with_next = alone.copy()
    place(with_next, make_sound("hats", rng), 0.38)
    uncut = hit_features(with_next, RATE, 0.3)
    cut = hit_features(with_next, RATE, 0.3, end_limit_s=0.375)
    ## the next hit starts at 0.38, so cutting at 0.375 must give what the kick alone gives
    assert not np.allclose(hit_features(alone, RATE, 0.3), uncut, atol=1e-3)
    assert np.allclose(hit_features(alone, RATE, 0.3, end_limit_s=0.375), cut, atol=1e-6)


def test_classifier_separates_kick_snare_and_hats(tmp_path):
    x_train, y_train = _dataset(0)
    x_test, y_test = _dataset(1, per_class=60)
    classifier = HitClassifier.fit(x_train, y_train, VOICES)
    assert (classifier.predict(x_test) == y_test).mean() > 0.95
    assert np.allclose(classifier.predict_proba(x_test).sum(axis=1), 1.0)
    classifier.save(tmp_path / "c.npz", {"note": "test"})
    loaded = HitClassifier.load(tmp_path / "c.npz")
    assert loaded.classes == VOICES
    assert np.array_equal(loaded.predict(x_test), classifier.predict(x_test))


def test_a_class_with_no_examples_is_an_error():
    x, y = _dataset(0, per_class=10)
    with pytest.raises(EvalError, match="no training examples"):
        HitClassifier.fit(x[y != 2], y[y != 2], VOICES)


def _write_avp(tmp_path):
    ## P1, P2 train; P3 validation; P9 test, whose audio file does not exist at all
    rng = np.random.default_rng(5)
    (tmp_path / "avp").mkdir(exist_ok=True)
    rows = []
    labels = ["kd", "sd", "hhc", "hho"]
    kind_of = {"kd": "kick", "sd": "snare", "hhc": "hats", "hho": "hats"}
    for participant in ("P1", "P2", "P3", "P9"):
        onsets = [0.5 + 0.5 * i for i in range(8)]
        name = f"{participant}_improvisation.wav"
        if participant != "P9":
            wave = np.zeros(RATE * 5)
            for i, onset in enumerate(onsets):
                place(wave, make_sound(kind_of[labels[i % 4]], rng), onset)
            sf.write(tmp_path / "avp" / name, wave.astype(np.float32), RATE)
        for i, onset in enumerate(onsets):
            rows.append((participant, labels[i % 4], name, onset, "Improvisation"))
    index = pd.DataFrame(rows, columns=["participant_id", "label", "audio_path", "onset_s", "kind"])
    index.to_csv(tmp_path / "index.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": ["P1", "P2", "P3", "P9"],
            "split": ["train", "train", "validation", "test"],
            "seed": 0,
        }
    ).to_csv(tmp_path / "splits.csv", index=False)
    return index


def test_training_reads_only_the_requested_participants(tmp_path):
    index = _write_avp(tmp_path)
    _x, y = training_examples(index, tmp_path / "avp", {"P1", "P2"}, VOICES)
    ## two participants, eight sounds each, three shifts, and each noise variant
    assert len(y) == 2 * 8 * 3 * len(NOISE_SNRS_DB)
    assert set(y) == {0, 1, 2}
    ## asking for the test participant, whose audio is missing, fails: so it was never opened above
    with pytest.raises(EvalError, match="not found"):
        training_examples(index, tmp_path / "avp", {"P9"}, VOICES)


def test_train_classifier_uses_train_and_validation_and_never_the_test_participant(tmp_path):
    _write_avp(tmp_path)
    metrics = train_classifier(
        tmp_path / "index.csv",
        tmp_path / "avp",
        tmp_path / "splits.csv",
        VOICES,
        tmp_path / "out" / "classifier.npz",
    )
    assert metrics["n_train_examples"] == 2 * 8 * 3 * len(NOISE_SNRS_DB)
    assert metrics["n_validation_sounds"] == 8
    assert metrics["train_participants"] == 2
    assert metrics["validation_participants"] == 1
    assert metrics["validation_accuracy"] > 0.8
    assert metrics["validation_accuracy_with_20db_noise"] > 0.7
    assert (tmp_path / "out" / "classifier.npz").exists()


def test_added_noise_has_the_requested_signal_to_noise_ratio():
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.3, RATE * 4).astype(np.float32)
    for snr in (10.0, 30.0):
        noise = add_white_noise(wave, snr, rng) - wave
        measured = 10 * np.log10(np.mean(wave**2) / np.mean(noise**2))
        assert measured == pytest.approx(snr, abs=0.5)


def test_noise_in_an_empty_band_does_not_change_a_kick_into_a_broadband_sound(tmp_path):
    rng = np.random.default_rng(3)
    wave = np.zeros(RATE)
    place(wave, make_sound("kick", rng), 0.3)
    wave = wave.astype(np.float32)
    clean = hit_features(wave, RATE, 0.3)
    noisy = hit_features(add_white_noise(wave, 40.0, rng), RATE, 0.3)
    ## the floor keeps the noise from rewriting the high bands, where the kick has no energy
    assert np.abs(noisy - clean).max() < 25.0

"""Model config and audio front-ends (src/models/config.py, src/models/features.py)."""

import copy

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")  # part of the optional "model" extra

import soundfile as sf  # noqa: E402

from src.models.config import (  # noqa: E402
    DEFAULT_PATH,
    ModelConfigError,
    load_model_config,
    parse_model_config,
)
from src.models.features import (  # noqa: E402
    compute_features,
    feature_dim,
    load_audio,
    mel_filterbank,
    n_frames,
    power_spectrogram,
    resample,
)

CFG = load_model_config()
SR = CFG.audio.sample_rate


def burst(freq_hz: float, start_s: float, total_s: float = 2.0) -> np.ndarray:
    """Silence with one decaying sine hit starting at start_s."""
    audio = np.zeros(int(total_s * SR), dtype=np.float32)
    t = np.arange(int(0.2 * SR)) / SR
    hit = np.sin(2 * np.pi * freq_hz * t) * np.exp(-t / 0.03)
    start = int(start_s * SR)
    audio[start : start + hit.size] = hit[: audio.size - start]
    return audio


# --------------------------------------------------------------------------- config


def test_default_config_loads():
    assert CFG.audio.sample_rate == 22050
    assert 95 < CFG.audio.frame_rate < 105  # about one frame every 10 ms
    assert CFG.encoder.d_model % CFG.encoder.n_heads == 0
    assert CFG.r3.fmax_hz == CFG.audio.nyquist  # null in the YAML


def _raw():
    return yaml.safe_load(DEFAULT_PATH.read_text())


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda r: r["audio"].update(typo=1), "unknown keys"),
        (lambda r: r["encoder"].pop("dropout"), "missing keys"),
        (lambda r: r["encoder"].update(n_heads=3), "divisible"),
        (lambda r: r["features"]["r1"].update(band_edges_hz=[5000, 200]), "increasing"),
        (lambda r: r["features"]["r1"].update(band_edges_hz=[200, 20000]), "Nyquist"),
        (lambda r: r["loss"].update(name="hinge"), "loss.name"),
        (lambda r: r["audio"].update(sample_rate="fast"), "integer"),
    ],
)
def test_bad_configs_are_rejected(edit, message):
    raw = copy.deepcopy(_raw())
    edit(raw)
    with pytest.raises(ModelConfigError, match=message):
        parse_model_config(raw)


# --------------------------------------------------------------------------- audio


def test_resample_halves_the_length():
    audio = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    assert resample(audio, 44100, 22050).shape == (22050,)
    np.testing.assert_array_equal(resample(audio, 22050, 22050), audio)


def test_load_audio_mixes_to_mono_and_resamples(tmp_path):
    stereo = np.random.default_rng(0).uniform(-0.5, 0.5, (44100, 2)).astype(np.float32)
    path = tmp_path / "clip.wav"
    sf.write(path, stereo, 44100, subtype="FLOAT")
    audio = load_audio(path, SR)
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert abs(audio.size - SR) <= 1


def test_frame_count_matches_formula():
    for n in (1, 219, 220, 22050, 100_000):
        spec = power_spectrogram(np.zeros(n, np.float32), CFG.audio.n_fft, CFG.audio.hop_length)
        assert spec.shape == (n_frames(n, CFG.audio.hop_length), CFG.audio.n_fft // 2 + 1)


# --------------------------------------------------------------------------- R1 and R3


@pytest.mark.parametrize("name", ["r1", "r3"])
def test_shapes_and_dims(name):
    audio = burst(100, 0.5)
    out = compute_features(name, audio, CFG)
    assert out.shape == (n_frames(audio.size, CFG.audio.hop_length), feature_dim(name, CFG))
    assert out.dtype == np.float32 and np.isfinite(out).all()


def test_r1_low_hit_lands_in_the_low_band_at_the_right_time():
    r1 = compute_features("r1", burst(60, 1.0), CFG)
    peak = int(r1[:, 0].argmax())
    assert abs(peak - 1.0 * CFG.audio.frame_rate) <= 3  # within ~30 ms
    assert r1[peak, 0] > r1[peak, 2]


def test_r1_high_hit_lands_in_the_high_band():
    r1 = compute_features("r1", burst(8000, 1.0), CFG)
    peak = int(r1[:, 2].argmax())
    assert abs(peak - 1.0 * CFG.audio.frame_rate) <= 3
    assert r1[peak, 2] > r1[peak, 0]


def test_r1_hit_on_the_first_frame_counts():
    r1 = compute_features("r1", burst(60, 0.0), CFG)
    assert r1[0, 0] > 1.0  # a downbeat right at the clip start still reads as an onset


@pytest.mark.parametrize("name", ["r1", "r3"])
def test_features_ignore_overall_level(name):
    audio = burst(60, 0.3) + burst(8000, 1.1)
    loud = compute_features(name, audio, CFG)
    soft = compute_features(name, 0.01 * audio, CFG)
    np.testing.assert_allclose(loud, soft, atol=1e-3)


@pytest.mark.parametrize("name", ["r1", "r3"])
def test_silence_gives_finite_flat_features(name):
    out = compute_features(name, np.zeros(SR, np.float32), CFG)
    assert np.isfinite(out).all()
    assert np.ptp(out) == 0


def test_r3_is_scaled_to_unit_range():
    r3 = compute_features("r3", burst(300, 0.5), CFG)
    assert r3.min() >= 0.0 and r3.max() == pytest.approx(1.0)


def test_mel_filterbank_has_no_empty_filters():
    bank = mel_filterbank(SR, CFG.audio.n_fft, CFG.r3.n_mels, CFG.r3.fmin_hz, CFG.r3.fmax_hz)
    assert bank.shape == (CFG.r3.n_mels, CFG.audio.n_fft // 2 + 1)
    assert (bank.sum(axis=1) > 0).all()
    with pytest.raises(ValueError, match="cover no STFT bin"):
        mel_filterbank(SR, 64, 128, 0, SR / 2)


def test_unknown_front_end_is_rejected():
    with pytest.raises(ValueError, match="unknown front-end"):
        compute_features("r9", np.zeros(100, np.float32), CFG)
    with pytest.raises(ValueError, match="unknown front-end"):
        feature_dim("r9", CFG)

"""Load and check config/models/*.yaml, Member C's model settings.

    from src.models.config import load_model_config

    cfg = load_model_config()     # config/models/base.yaml
    cfg.audio.sample_rate         # 22050
    cfg.audio.frame_rate          # ~100.2 frames per second
    cfg.encoder.d_model           # 128

These are technical settings (audio rate, frame size, model size). Musical constants
(grid, tempos, voices, what counts as the same rhythm) stay in config/music_spec.yaml
and are read through src.music.spec.load_spec().

Loading fails with ModelConfigError on missing or unknown keys and out-of-range values,
so a typo in the YAML can't silently change an experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "models" / "base.yaml"
LOSSES = ("supcon",)


class ModelConfigError(ValueError):
    """A model config file is malformed or inconsistent."""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    n_fft: int
    hop_length: int

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2

    @property
    def frame_rate(self) -> float:
        """Feature frames per second."""
        return self.sample_rate / self.hop_length


@dataclass(frozen=True)
class R1Config:
    band_edges_hz: tuple[float, ...]  # inner edges; the bands run from 0 Hz to Nyquist
    dynamic_range_db: float
    onset_db_scale: float


@dataclass(frozen=True)
class R3Config:
    n_mels: int
    fmin_hz: float
    fmax_hz: float  # null in the YAML means Nyquist
    dynamic_range_db: float


@dataclass(frozen=True)
class EncoderConfig:
    frame_stride: int  # feature frames merged into one token by the input convolution
    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int
    dropout: float
    embed_dim: int


@dataclass(frozen=True)
class LossConfig:
    name: str
    temperature: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_patterns: int
    views_per_pattern: int
    learning_rate: float
    weight_decay: float
    epochs: int
    max_grad_norm: float
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    version: str
    audio: AudioConfig
    r1: R1Config
    r3: R3Config
    encoder: EncoderConfig
    loss: LossConfig
    training: TrainingConfig
    path: Path


def config_to_dict(cfg: ModelConfig) -> dict:
    """A JSON-friendly copy of a config, as saved inside checkpoints."""
    d = asdict(cfg)
    d["path"] = str(cfg.path)
    return d


def config_from_dict(d: dict) -> ModelConfig:
    """Rebuild a config saved by config_to_dict (e.g. from a checkpoint)."""
    r1 = dict(d["r1"], band_edges_hz=tuple(d["r1"]["band_edges_hz"]))
    return ModelConfig(
        version=d["version"],
        audio=AudioConfig(**d["audio"]),
        r1=R1Config(**r1),
        r3=R3Config(**d["r3"]),
        encoder=EncoderConfig(**d["encoder"]),
        loss=LossConfig(**d["loss"]),
        training=TrainingConfig(**d["training"]),
        path=Path(d["path"]),
    )


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    """Read and check a model config (default: config/models/base.yaml)."""
    path = Path(path) if path is not None else DEFAULT_PATH
    with path.open() as f:
        raw = yaml.safe_load(f)
    return parse_model_config(raw, path)


def parse_model_config(raw: Any, path: Path = DEFAULT_PATH) -> ModelConfig:
    """Build a ModelConfig from the parsed YAML, checking every value."""
    top = _keys(raw, ["version", "audio", "features", "encoder", "loss", "training"], str(path))

    a = _keys(top["audio"], ["sample_rate", "n_fft", "hop_length"], "audio")
    audio = AudioConfig(
        sample_rate=_int(a["sample_rate"], "audio.sample_rate", lo=1),
        n_fft=_int(a["n_fft"], "audio.n_fft", lo=2),
        hop_length=_int(a["hop_length"], "audio.hop_length", lo=1),
    )
    if audio.n_fft % 2:
        raise ModelConfigError("audio.n_fft must be even")
    if audio.hop_length > audio.n_fft:
        raise ModelConfigError("audio.hop_length must not exceed audio.n_fft")

    feats = _keys(top["features"], ["r1", "r3"], "features")
    f1 = _keys(feats["r1"], ["band_edges_hz", "dynamic_range_db", "onset_db_scale"], "r1")
    edges = f1["band_edges_hz"]
    if not isinstance(edges, list) or not edges:
        raise ModelConfigError("features.r1.band_edges_hz: expected a non-empty list")
    edges = tuple(_num(e, "features.r1.band_edges_hz", lo=0) for e in edges)
    if any(lo >= hi for lo, hi in pairwise(edges)) or edges[0] <= 0:
        raise ModelConfigError("features.r1.band_edges_hz must be positive and increasing")
    if edges[-1] >= audio.nyquist:
        raise ModelConfigError("features.r1.band_edges_hz must lie below Nyquist")
    r1 = R1Config(
        band_edges_hz=edges,
        dynamic_range_db=_num(f1["dynamic_range_db"], "features.r1.dynamic_range_db", lo=1),
        onset_db_scale=_num(f1["onset_db_scale"], "features.r1.onset_db_scale", lo=1e-6),
    )

    f3 = _keys(feats["r3"], ["n_mels", "fmin_hz", "fmax_hz", "dynamic_range_db"], "r3")
    fmax = audio.nyquist if f3["fmax_hz"] is None else _num(f3["fmax_hz"], "features.r3.fmax_hz")
    r3 = R3Config(
        n_mels=_int(f3["n_mels"], "features.r3.n_mels", lo=1),
        fmin_hz=_num(f3["fmin_hz"], "features.r3.fmin_hz", lo=0),
        fmax_hz=fmax,
        dynamic_range_db=_num(f3["dynamic_range_db"], "features.r3.dynamic_range_db", lo=1),
    )
    if not r3.fmin_hz < r3.fmax_hz <= audio.nyquist:
        raise ModelConfigError("features.r3 needs fmin_hz < fmax_hz <= Nyquist")

    e = _keys(
        top["encoder"],
        ["frame_stride", "d_model", "n_layers", "n_heads", "ff_dim", "dropout", "embed_dim"],
        "encoder",
    )
    encoder = EncoderConfig(
        frame_stride=_int(e["frame_stride"], "encoder.frame_stride", lo=1),
        d_model=_int(e["d_model"], "encoder.d_model", lo=1),
        n_layers=_int(e["n_layers"], "encoder.n_layers", lo=1),
        n_heads=_int(e["n_heads"], "encoder.n_heads", lo=1),
        ff_dim=_int(e["ff_dim"], "encoder.ff_dim", lo=1),
        dropout=_num(e["dropout"], "encoder.dropout", lo=0),
        embed_dim=_int(e["embed_dim"], "encoder.embed_dim", lo=1),
    )
    if encoder.d_model % encoder.n_heads:
        raise ModelConfigError("encoder.d_model must be divisible by encoder.n_heads")
    if encoder.dropout >= 1:
        raise ModelConfigError("encoder.dropout must be below 1")

    lo_ = _keys(top["loss"], ["name", "temperature"], "loss")
    if lo_["name"] not in LOSSES:
        raise ModelConfigError(f"loss.name must be one of {LOSSES}, got {lo_['name']!r}")
    loss = LossConfig(
        name=lo_["name"], temperature=_num(lo_["temperature"], "loss.temperature", lo=1e-6)
    )

    t = _keys(
        top["training"],
        [
            "batch_patterns",
            "views_per_pattern",
            "learning_rate",
            "weight_decay",
            "epochs",
            "max_grad_norm",
            "seed",
        ],
        "training",
    )
    training = TrainingConfig(
        batch_patterns=_int(t["batch_patterns"], "training.batch_patterns", lo=1),
        views_per_pattern=_int(t["views_per_pattern"], "training.views_per_pattern", lo=2),
        learning_rate=_num(t["learning_rate"], "training.learning_rate", lo=1e-12),
        weight_decay=_num(t["weight_decay"], "training.weight_decay", lo=0),
        epochs=_int(t["epochs"], "training.epochs", lo=1),
        max_grad_norm=_num(t["max_grad_norm"], "training.max_grad_norm", lo=1e-12),
        seed=_int(t["seed"], "training.seed", lo=0),
    )

    return ModelConfig(str(top["version"]), audio, r1, r3, encoder, loss, training, Path(path))


# --------------------------------------------------------------------------- helpers


def _keys(d: Any, expected: list[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise ModelConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    missing = [k for k in expected if k not in d]
    unknown = [k for k in d if k not in expected]
    if missing or unknown:
        raise ModelConfigError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return d


def _int(v: Any, where: str, lo: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ModelConfigError(f"{where}: expected an integer, got {v!r}")
    if lo is not None and v < lo:
        raise ModelConfigError(f"{where}: must be at least {lo}, got {v}")
    return v


def _num(v: Any, where: str, lo: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ModelConfigError(f"{where}: expected a number, got {v!r}")
    if lo is not None and v < lo:
        raise ModelConfigError(f"{where}: must be at least {lo}, got {v}")
    return float(v)

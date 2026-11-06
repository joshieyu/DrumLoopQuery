"""The rhythm encoder (src/models/encoder.py)."""

import pytest

torch = pytest.importorskip("torch")  # part of the optional "model" extra

from src.models.config import load_model_config  # noqa: E402
from src.models.encoder import (  # noqa: E402
    RhythmEncoder,
    kind_ids,
    n_trainable,
    pad_features,
)

CFG = load_model_config()


def make(n_features: int = 3) -> RhythmEncoder:
    torch.manual_seed(0)
    return RhythmEncoder(n_features, CFG.encoder).eval()


def clips(lengths, n_features=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(n, n_features, generator=g) for n in lengths]


def test_output_is_one_unit_vector_per_clip():
    x, lengths = pad_features(clips([50, 80, 120]))
    emb = make()(x, lengths, kind_ids(["library", "beatbox", "tap"]))
    assert emb.shape == (3, CFG.encoder.embed_dim)
    torch.testing.assert_close(emb.norm(dim=1), torch.ones(3))


def test_padding_does_not_change_a_clips_embedding():
    model = make()
    short, long_ = clips([40, 300])
    x1, l1 = pad_features([short])
    x2, l2 = pad_features([short, long_])
    with torch.no_grad():
        alone = model(x1, l1, kind_ids(["beatbox"]))[0]
        batched = model(x2, l2, kind_ids(["beatbox", "library"]))[0]
    torch.testing.assert_close(alone, batched, atol=1e-5, rtol=1e-5)


def test_kind_token_changes_the_embedding():
    model = make()
    x, lengths = pad_features(clips([60, 60]))
    x[1] = x[0]
    with torch.no_grad():
        emb = model(x, lengths, kind_ids(["library", "beatbox"]))
    assert not torch.allclose(emb[0], emb[1])


@pytest.mark.parametrize("n_features", [3, 64, 768])  # R1, R3, and MERT-sized inputs
def test_parameter_budget(n_features):
    assert n_trainable(make(n_features)) < 5_000_000  # README §6


def test_gradients_reach_every_parameter():
    model = make().train()
    x, lengths = pad_features(clips([30, 45]))
    model(x, lengths, kind_ids(["library", "tap"])).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_bad_inputs_are_rejected():
    model = make()
    x, lengths = pad_features(clips([10, 20]))
    with pytest.raises(ValueError, match="unknown input kinds"):
        kind_ids(["library", "humming"])
    with pytest.raises(ValueError, match="features must be"):
        model(x[..., :2], lengths, kind_ids(["library", "tap"]))
    with pytest.raises(ValueError, match="between 1 and"):
        model(x, torch.tensor([10, 21]), kind_ids(["library", "tap"]))
    with pytest.raises(ValueError, match="same width"):
        pad_features([torch.rand(5, 3), torch.rand(5, 4)])
    with pytest.raises(ValueError, match="no frames"):
        pad_features([torch.rand(0, 3)])

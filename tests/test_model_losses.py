"""Positives and the SupCon loss (src/models/losses.py)."""

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")  # part of the optional "model" extra

from src.models.config import load_model_config  # noqa: E402
from src.models.encoder import RhythmEncoder, kind_ids  # noqa: E402
from src.models.losses import positive_mask, supcon_loss  # noqa: E402
from src.music.spec import load_spec  # noqa: E402

SPEC = load_spec()
CFG = load_model_config()


# --------------------------------------------------------------------------- positives


def test_positive_rules_follow_the_answer_key():
    # 0, 1: same pattern on two kits. 2: other pattern, same tap reduction as 0 and 1.
    # 3: beatbox of pattern A. 4: tap of reduction X. 5: unrelated library loop.
    patterns = ["A", "A", "B", "A", "T", "C"]
    dualized = ["X", "X", "X", "X", "X", "Y"]
    kinds = ["library", "library", "library", "beatbox", "tap", "library"]
    m = positive_mask(patterns, dualized, kinds, SPEC)

    assert m.dtype == torch.bool and torch.equal(m, m.T)
    assert not m.diagonal().any()
    assert m[0, 1] and m[0, 3]  # same pattern_id
    assert not m[0, 2]  # same tap reduction, but neither is a tap
    assert m[4, 0] and m[4, 2] and m[4, 3]  # a tap matches on the dualized id
    assert not m[4, 5]
    assert not m[5].any()


def test_positive_mask_refuses_a_non_exact_spec():
    loose = dataclasses.replace(
        SPEC, same_rhythm=dataclasses.replace(SPEC.same_rhythm, match="near")
    )
    with pytest.raises(ValueError, match="exact"):
        positive_mask(["A", "A"], ["X", "X"], ["library", "beatbox"], loose)


def test_positive_mask_needs_matching_lengths():
    with pytest.raises(ValueError, match="one entry per clip"):
        positive_mask(["A", "A"], ["X"], ["library", "beatbox"], SPEC)


# --------------------------------------------------------------------------- SupCon


def test_supcon_matches_a_hand_computation():
    emb = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]), dim=1)
    pos = torch.tensor([[False, True, False], [True, False, False], [False, False, False]])
    t = 0.5
    s = (emb @ emb.T).tolist()

    def term(i, p, others):
        return s[i][p] / t - math.log(sum(math.exp(s[i][o] / t) for o in others))

    expected = -(term(0, 1, [1, 2]) + term(1, 0, [0, 2])) / 2  # anchor 2 has no positive
    assert supcon_loss(emb, pos, t).item() == pytest.approx(expected, rel=1e-5)


def test_supcon_prefers_clustered_embeddings():
    labels = ["A", "A", "B", "B", "C", "C"]
    pos = positive_mask(labels, labels, ["library"] * 6, SPEC)
    clustered = torch.eye(3).repeat_interleave(2, dim=0)  # each pair identical
    scrambled = torch.eye(3)[[0, 1, 2, 0, 1, 2]]  # each pair orthogonal
    assert supcon_loss(clustered, pos, 0.1) < supcon_loss(scrambled, pos, 0.1)


def test_supcon_needs_a_positive_somewhere():
    emb = torch.eye(3)
    with pytest.raises(ValueError, match="no clip in the batch has a positive"):
        supcon_loss(emb, torch.zeros(3, 3, dtype=torch.bool), 0.1)


@pytest.mark.parametrize("seed", [0, 6])  # 6 collapsed before the encoder's init fixes
def test_encoder_learns_a_toy_retrieval_task(seed):
    """4 patterns x 3 noisy views each: training should pull each pattern's views together.

    With 2 positives among 11 other clips, a perfect model's loss is about log 2 = 0.69.
    """
    torch.manual_seed(seed)
    templates = torch.rand(4, 40, 3)
    labels = [str(p) for p in range(4) for _ in range(3)]
    kinds = kind_ids(["library", "beatbox", "tap"] * 4)
    pos = positive_mask(labels, labels, ["library"] * 12, SPEC)
    lengths = torch.full((12,), 40)

    model = RhythmEncoder(3, CFG.encoder)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(100):
        x = templates.repeat_interleave(3, dim=0) + 0.1 * torch.randn(12, 40, 3)
        loss = supcon_loss(model(x, lengths, kinds), pos, CFG.loss.temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # as the trainer will
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < 1.0
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:5]) / 5

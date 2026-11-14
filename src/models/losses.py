"""Training loss (README §6) and which clip pairs count as the same rhythm.

    from src.models.losses import positive_mask, supcon_loss

    pos = positive_mask(pattern_ids, dualized_ids, kinds, spec)    # (N, N) bool
    loss = supcon_loss(emb, pos, temperature=cfg.loss.temperature)

Positives follow the answer key's rules (README §5.4; music spec §2 and §7), so
training pulls together exactly the clips D scores as matches:

- a pair that includes a tap matches when their dualized_pattern_ids are equal,
  because a tap has only two hands and many 3-voice patterns share one reduction;
- any other pair matches when their pattern_ids are equal.

Both rules assume the spec's `same_rhythm.match: exact`. positive_mask refuses any
other setting, so a change to the spec can't silently leave training out of step
with D's answer key.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from src.music.spec import MusicSpec


def positive_mask(
    pattern_ids: Sequence[str],
    dualized_ids: Sequence[str],
    kinds: Sequence[str],
    spec: MusicSpec,
) -> torch.Tensor:
    """(N, N) bool: True where clips i and j are the same rhythm. The diagonal is False."""
    if spec.same_rhythm.match != "exact":
        raise ValueError(
            f"positive_mask implements same_rhythm.match 'exact', but the spec says "
            f"{spec.same_rhythm.match!r}; update it to match D's answer key"
        )
    n = len(pattern_ids)
    if not (len(dualized_ids) == len(kinds) == n):
        raise ValueError("pattern_ids, dualized_ids, and kinds need one entry per clip")
    pattern = torch.as_tensor(pd.factorize(pd.Series(list(pattern_ids)))[0])
    dualized = torch.as_tensor(pd.factorize(pd.Series(list(dualized_ids)))[0])
    tap = torch.as_tensor(np.asarray(kinds) == "tap")

    same_pattern = pattern[:, None] == pattern[None, :]
    same_dualized = dualized[:, None] == dualized[None, :]
    involves_tap = tap[:, None] | tap[None, :]
    mask = torch.where(involves_tap, same_dualized, same_pattern)
    mask.fill_diagonal_(False)
    return mask


def supcon_loss(emb: torch.Tensor, pos_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020, the L_out form).

    emb: (N, d), rows L2-normalized. pos_mask: (N, N) bool from positive_mask. For each
    anchor with at least one positive, the loss is minus the mean, over its positives,
    of log softmax(similarity / temperature) taken over every other clip in the batch.
    Anchors without a positive in the batch are skipped; a batch with none raises.
    """
    if emb.ndim != 2 or emb.shape[0] < 2:
        raise ValueError(f"need (N >= 2, d) embeddings, got {tuple(emb.shape)}")
    n = emb.shape[0]
    if pos_mask.shape != (n, n) or pos_mask.dtype != torch.bool:
        raise ValueError(f"pos_mask must be a ({n}, {n}) bool tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    pos_mask = pos_mask.to(emb.device)
    self_mask = torch.eye(n, dtype=torch.bool, device=emb.device)
    logits = (emb @ emb.T / temperature).masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positives = pos_mask & ~self_mask
    n_pos = positives.sum(dim=1)
    anchors = n_pos > 0
    if not bool(anchors.any()):
        raise ValueError("no clip in the batch has a positive; sample several views per pattern")
    mean_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1) / n_pos.clamp(min=1)
    return -mean_log_prob[anchors].mean()

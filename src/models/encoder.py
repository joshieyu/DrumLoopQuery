"""The rhythm encoder (README §6): features in, an embedding of length 1 out.

    from src.models.encoder import RhythmEncoder, kind_ids, pad_features

    model = RhythmEncoder(n_features=3, cfg=cfg.encoder)
    x, lengths = pad_features([r1_a, r1_b])                     # (2, T, 3), (2,)
    emb = model(x, lengths, kind_ids(["library", "beatbox"]))   # (2, 128), unit rows

One network for every input, so queries and library loops land in the same space.
In order:

1. A linear projection of each feature frame to d_model, then LayerNorm.
2. A learned token for the input's kind (library loop, beatbox, or tap) in front of
   the frames, so the network knows what it is listening to.
3. Sinusoidal positions, then n_layers pre-norm Transformer blocks.
4. Attention pooling: a learned score per frame, softmax over the clip's frames (the
   kind token is left out, so the output has to come from the audio).
5. A linear head and L2 normalisation, so cosine similarity is a dot product (the
   form src/eval/evaluate.py scores).

Steps 1, 4, and the kind token's small initial values keep the clip's own content
from being drowned out by what every clip shares (positions, kind token) at the start
of training. Without them, a toy task collapsed to one embedding for every clip on
some seeds (tests/test_model_losses.py). Train with gradient clipping as well.

The attention layers keep separate query, key, and value projections (rather than
torch's fused nn.MultiheadAttention) so a LoRA adapter can wrap them later. Padding
frames are masked in attention and pooling, so a clip's embedding doesn't depend on
the other clips in its batch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.config import EncoderConfig

INPUT_KINDS = ("library", "beatbox", "tap")


def kind_ids(kinds: Sequence[str]) -> torch.Tensor:
    """Kind names ("library", "beatbox", "tap") to the encoder's token indices."""
    unknown = sorted({k for k in kinds if k not in INPUT_KINDS})
    if unknown:
        raise ValueError(f"unknown input kinds {unknown}; expected {INPUT_KINDS}")
    return torch.tensor([INPUT_KINDS.index(k) for k in kinds], dtype=torch.long)


def pad_features(
    features: Sequence[np.ndarray | torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack clips of different lengths: (B, T_max, F) zero-padded, and lengths (B,)."""
    if not features:
        raise ValueError("no clips to pad")
    tensors = [torch.as_tensor(np.asarray(f), dtype=torch.float32) for f in features]
    widths = {t.shape[1] if t.ndim == 2 else None for t in tensors}
    if len(widths) != 1 or None in widths:
        raise ValueError("every clip needs a 2-D (frames, features) array of the same width")
    if any(t.shape[0] == 0 for t in tensors):
        raise ValueError("a clip has no frames")
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    return nn.utils.rnn.pad_sequence(tensors, batch_first=True), lengths


def sinusoidal_positions(
    n: int, d: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """The Transformer's fixed sine/cosine position codes: (n, d)."""
    position = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    rate = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d)
    )
    codes = torch.zeros(n, d, device=device, dtype=torch.float32)
    codes[:, 0::2] = torch.sin(position * rate)
    codes[:, 1::2] = torch.cos(position * rate[: d // 2])
    return codes.to(dtype)


class SelfAttention(nn.Module):
    """Multi-head self-attention with separate q, k, v, and output projections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d). valid: (B, T), True for real tokens; padding is never attended."""
        b, t, d = x.shape

        def heads(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            heads(self.q(x)),
            heads(self.k(x)),
            heads(self.v(x)),
            attn_mask=valid[:, None, None, :],  # True = may be attended to
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Pre-norm Transformer block: attention, then a GELU feed-forward, each residual."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), valid))
        return x + self.drop(self.ff(self.norm2(x)))


class RhythmEncoder(nn.Module):
    """Shared encoder for library loops and queries (README §6)."""

    def __init__(self, n_features: int, cfg: EncoderConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d)
        self.input_norm = nn.LayerNorm(d)
        self.kind_token = nn.Embedding(len(INPUT_KINDS), d)
        nn.init.normal_(self.kind_token.weight, std=0.02)  # small, as in BERT and ViT
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.ff_dim, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(d)
        self.pool_score = nn.Linear(d, 1)
        self.head = nn.Linear(d, cfg.embed_dim)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor, kinds: torch.Tensor
    ) -> torch.Tensor:
        """features (B, T, F) zero-padded, lengths (B,), kinds (B,) -> (B, embed_dim)."""
        if features.ndim != 3 or features.shape[2] != self.n_features:
            raise ValueError(
                f"features must be (B, T, {self.n_features}), got {tuple(features.shape)}"
            )
        b, t, _ = features.shape
        if lengths.shape != (b,) or kinds.shape != (b,):
            raise ValueError("lengths and kinds need one entry per clip")
        if bool((lengths < 1).any()) or bool((lengths > t).any()):
            raise ValueError("every length must be between 1 and the padded length")

        device = features.device
        frames = self.input_norm(self.input_proj(features))
        token = self.kind_token(kinds.to(device))[:, None, :]
        x = torch.cat([token, frames], dim=1)  # position 0 is the kind token
        x = self.drop(x + sinusoidal_positions(t + 1, x.shape[2], device, x.dtype))
        # Real tokens: the kind token plus each clip's own frames.
        valid = torch.arange(t + 1, device=device)[None, :] <= lengths.to(device)[:, None]

        for block in self.blocks:
            x = block(x, valid)
        x = self.norm(x)

        frames_only = valid.clone()
        frames_only[:, 0] = False  # pool the audio frames, not the kind token
        scores = self.pool_score(x).squeeze(-1).masked_fill(~frames_only, float("-inf"))
        pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * x).sum(dim=1)
        return F.normalize(self.head(pooled), dim=-1)


def n_trainable(model: nn.Module) -> int:
    """Number of trainable parameters (README §6 budget: under 5M, MERT excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

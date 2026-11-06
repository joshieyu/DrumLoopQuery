"""Write a system's embeddings in the file format D's scorer reads (README §12).

    from src.models.export import embedding_path, write_embeddings

    path = embedding_path("r1_supcon", "syn_beatbox")
    # -> results/embeddings/r1_supcon_syn_beatbox.npz
    write_embeddings(path, query_ids, query_emb, library_ids, library_emb)

The file holds query_ids (Nq,), library_ids (Nl,), query_emb (Nq, d), and
library_emb (Nl, d), every row L2-normalized. It is checked here against the rules
src/eval/evaluate.py enforces, so a bad file fails when it is written rather than
when D scores it. Embeddings are numpy arrays; from torch, pass
`tensor.detach().cpu().numpy()`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = REPO_ROOT / "results" / "embeddings"
NORM_TOLERANCE = 1e-4  # stricter than evaluate.py's 1e-3, so files always pass there


def embedding_path(system: str, testset: str, root: str | Path = EMBEDDINGS_DIR) -> Path:
    """results/embeddings/<system>_<testset>.npz, the name README §12 gives."""
    for name, value in (("system", system), ("testset", testset)):
        if not value or "/" in value or "\\" in value:
            raise ValueError(f"{name} must be a non-empty name without slashes, got {value!r}")
    return Path(root) / f"{system}_{testset}.npz"


def write_embeddings(
    path: str | Path,
    query_ids: Sequence[str],
    query_emb: np.ndarray,
    library_ids: Sequence[str],
    library_emb: np.ndarray,
) -> Path:
    """Check and save one system file. Returns the path written."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"{path}: the file name must end in .npz")
    query_ids = _ids(query_ids, "query_ids")
    library_ids = _ids(library_ids, "library_ids")
    query_emb = _embeddings(query_emb, len(query_ids), "query_emb")
    library_emb = _embeddings(library_emb, len(library_ids), "library_emb")
    if query_emb.shape[1] != library_emb.shape[1]:
        raise ValueError(
            f"query_emb has {query_emb.shape[1]} dimensions but library_emb has "
            f"{library_emb.shape[1]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        query_ids=query_ids,
        library_ids=library_ids,
        query_emb=query_emb,
        library_emb=library_emb,
    )
    return path


def _ids(ids: Sequence[str], name: str) -> np.ndarray:
    values = [str(i) for i in ids]
    if not values:
        raise ValueError(f"{name} is empty")
    if any(not v for v in values):
        raise ValueError(f"{name} contains an empty id")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return np.array(values, dtype=str)


def _embeddings(emb: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != n_rows:
        raise ValueError(f"{name} has shape {emb.shape}, expected ({n_rows}, d)")
    if not np.isfinite(emb).all():
        raise ValueError(f"{name} contains NaN or infinity")
    worst = float(np.abs(np.linalg.norm(emb, axis=1) - 1.0).max())
    if worst > NORM_TOLERANCE:
        raise ValueError(f"{name} rows must have length 1 (worst is off by {worst:.3g})")
    return emb

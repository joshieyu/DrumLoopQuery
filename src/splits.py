"""Deterministic participant assignment and approved pattern split precedence."""

import numpy as np
import pandas as pd

SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}


def resolve_patterns(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every copy within the most held-out split; remove other split copies."""
    rank = table["split"].map(SPLIT_RANK)
    if rank.isna().any():
        raise ValueError("Unknown pattern split")
    winners = rank.groupby(table.pattern_id).transform("max")
    return table[rank == winners].copy(), table[rank != winners].copy()


def assign_participants(people: list[str], *, counts: dict[str, int], seed: int) -> pd.DataFrame:
    """Save one stable seeded partition rather than choosing participants per query."""
    people = sorted(set(people), key=lambda p: int(p.removeprefix("P")))
    if set(counts) != set(SPLIT_RANK) or sum(counts.values()) != len(people):
        raise ValueError("Participant counts must cover the whole dataset in three splits")
    if any(not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("Participant counts must be positive integers")
    order = np.random.default_rng(seed).permutation(people)
    rows, start = [], 0
    for split in SPLIT_RANK:
        for person in order[start : start + counts[split]]:
            rows.append({"participant_id": str(person), "split": split, "seed": seed})
        start += counts[split]
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)

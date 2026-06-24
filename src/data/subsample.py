"""Deterministic subsampling to a fixed corpus size."""

from __future__ import annotations

import logging
import random
from typing import TypeVar

from src.config import SUBSET_SIZE, SEED

T = TypeVar("T")
logger = logging.getLogger(__name__)


def deterministic_subsample(
    items: list[T],
    n: int = SUBSET_SIZE,
    seed: int = SEED,
) -> list[T]:
    """
    Return exactly n items sampled without replacement using a fixed seed.

    If len(items) <= n, returns all items with a warning (no padding).
    Same seed always produces the same subset — required for size-matching
    bridge corpora to ADIta_Lad for a fair training comparison.
    """
    if len(items) <= n:
        logger.warning(
            "Corpus has %d items, which is <= requested subset size %d. "
            "Using all available pairs.",
            len(items), n,
        )
        return list(items)
    rng = random.Random(seed)
    sample = rng.sample(items, n)
    logger.info("Subsampled %d → %d pairs (seed=%d)", len(items), n, seed)
    return sample

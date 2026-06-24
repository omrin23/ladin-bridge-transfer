"""Unit tests for data/subsample.py."""

import pytest
from src.data.subsample import deterministic_subsample


def test_subsample_returns_correct_size():
    items = list(range(1000))
    result = deterministic_subsample(items, n=100, seed=42)
    assert len(result) == 100


def test_subsample_is_deterministic():
    items = list(range(1000))
    r1 = deterministic_subsample(items, n=100, seed=42)
    r2 = deterministic_subsample(items, n=100, seed=42)
    assert r1 == r2


def test_different_seeds_give_different_results():
    items = list(range(1000))
    r1 = deterministic_subsample(items, n=100, seed=42)
    r2 = deterministic_subsample(items, n=100, seed=99)
    assert r1 != r2


def test_subsample_smaller_than_n_returns_all():
    items = list(range(10))
    result = deterministic_subsample(items, n=100, seed=42)
    assert len(result) == 10
    assert sorted(result) == items


def test_subsample_preserves_items():
    items = [f"item_{i}" for i in range(500)]
    result = deterministic_subsample(items, n=200, seed=42)
    assert all(r in items for r in result)
    assert len(set(result)) == 200  # no duplicates

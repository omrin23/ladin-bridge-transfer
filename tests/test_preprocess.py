"""Unit tests for data/preprocess.py."""

import pytest
from src.data.preprocess import normalize_text, preprocess_pairs


def test_normalize_nfkc():
    # Ligature fi (U+FB01) → fi
    assert normalize_text("ﬁle") == "file"


def test_normalize_strips_control_chars():
    # Null byte and BOM should be stripped
    assert normalize_text("hello\x00world") == "helloworld"
    assert normalize_text("﻿text") == "text"


def test_normalize_strips_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_empty_pairs_dropped():
    pairs = [("", "ciao"), ("hello", ""), ("", "")]
    result = preprocess_pairs(pairs)
    assert result == []


def test_valid_pair_kept():
    pairs = [("Hello world", "Ciao mondo")]
    result = preprocess_pairs(pairs)
    assert len(result) == 1
    assert result[0] == ("Hello world", "Ciao mondo")


def test_length_ratio_filter():
    # One side is 10x longer than the other — should be dropped
    long_src = "word " * 100
    short_tgt = "x"
    pairs = [(long_src, short_tgt)]
    result = preprocess_pairs(pairs, max_length_ratio=9.0)
    assert result == []


def test_duplicate_pairs_dropped():
    pairs = [("Hello", "Ciao"), ("Hello", "Ciao"), ("Bye", "Arrivederci")]
    result = preprocess_pairs(pairs)
    assert len(result) == 2
    assert result[0] == ("Hello", "Ciao")
    assert result[1] == ("Bye", "Arrivederci")


def test_special_chars_handled():
    # Pairs with special Unicode chars should not raise
    pairs = [("Café résumé", "Tëst stríng")]
    result = preprocess_pairs(pairs)
    assert len(result) == 1


def test_min_chars_filter():
    pairs = [("a", "b"), ("hello", "world")]
    result = preprocess_pairs(pairs, min_chars=2)
    assert len(result) == 1
    assert result[0] == ("hello", "world")

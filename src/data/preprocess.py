"""
Preprocessing: Unicode NFKC normalization, control-character stripping,
empty-pair filtering, length-ratio filtering, token-length filtering,
and exact-pair deduplication.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional

from src.config import MAX_SEQ_LEN

logger = logging.getLogger(__name__)

# Pairs with char-length ratio exceeding this are almost certainly misaligned
_DEFAULT_MAX_RATIO: float = 9.0
_DEFAULT_MIN_CHARS: int = 1


def normalize_text(text: str) -> str:
    """NFKC normalization + strip ASCII control characters (keep tab/newline)."""
    text = unicodedata.normalize("NFKC", text)
    # Strip control chars (Cc) and format chars (Cf — includes BOM U+FEFF,
    # zero-width spaces, direction marks), except tab and newline.
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\t\n"
    ).strip()


def _approx_token_length(text: str, tokenizer) -> int:
    """Token count without special tokens, used for length filtering."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def preprocess_pairs(
    pairs: list[tuple[str, str]],
    tokenizer=None,
    max_seq_len: int = MAX_SEQ_LEN,
    max_length_ratio: float = _DEFAULT_MAX_RATIO,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> list[tuple[str, str]]:
    """
    Clean and filter (src, tgt) string pairs.

    Steps applied in order:
    1. NFKC normalization + control-char stripping on both sides
    2. Drop pairs where either side is empty or shorter than min_chars
    3. Drop pairs where char-length ratio exceeds max_length_ratio
    4. Drop pairs where either side exceeds max_seq_len tokens
       (only when a tokenizer is provided — skipped in unit tests)
    5. Drop exact duplicate (src, tgt) pairs (first occurrence kept)
    """
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    n_empty = n_ratio = n_token = n_dup = 0

    for raw_src, raw_tgt in pairs:
        src = normalize_text(raw_src)
        tgt = normalize_text(raw_tgt)

        if len(src) < min_chars or len(tgt) < min_chars:
            n_empty += 1
            continue

        ratio = max(len(src), len(tgt)) / max(min(len(src), len(tgt)), 1)
        if ratio > max_length_ratio:
            n_ratio += 1
            continue

        if tokenizer is not None:
            if (
                _approx_token_length(src, tokenizer) > max_seq_len
                or _approx_token_length(tgt, tokenizer) > max_seq_len
            ):
                n_token += 1
                continue

        pair = (src, tgt)
        if pair in seen:
            n_dup += 1
            continue
        seen.add(pair)
        result.append(pair)

    n_in = len(result) + n_empty + n_ratio + n_token + n_dup
    logger.info(
        "preprocess_pairs: %d → %d kept "
        "(dropped: %d empty/short, %d ratio, %d token-len, %d dup)",
        n_in, len(result), n_empty, n_ratio, n_token, n_dup,
    )
    return result

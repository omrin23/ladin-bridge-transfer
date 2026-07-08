"""Language codes, roles, and metadata for the Ladin bridge-transfer project."""

from __future__ import annotations

from dataclasses import dataclass

# NLLB language codes
LADIN: str = "lld_Latn"  # target low-resource language (added + seeded from Italian)
ITA: str = "ita_Latn"    # fixed anchor / pivot — NOT a bridge condition
ENG: str = "eng_Latn"    # available in OPUS Books (en-it) but not used as bridge condition
FRA: str = "fra_Latn"    # bridge language (OPUS Books fr-it) + secondary evaluation
SPA: str = "spa_Latn"    # bridge language (OPUS Books es-it) + secondary evaluation
POR: str = "por_Latn"    # bridge language (OPUS Books it-pt) + secondary evaluation

# All five languages used in evaluation
EVAL_LANGS: tuple[str, ...] = (LADIN, ITA, FRA, SPA, POR)

# Bridge languages only (Italian is the pivot, bridging ita→ita is degenerate)
BRIDGE_LANGS: tuple[str, ...] = (FRA, SPA, POR)

# FLORES+ Ladin variety — Val Badia, matching ADIta_Lad (sfrontull/lld_valbadia-ita)
LADIN_FLORES_VARIETY: str = "lld_Latn"

# Condition labels used in result file names and JSON fields
CONDITIONS: tuple[str, ...] = (
    "zero_shot",
    "direct",
    "bridge_fr",
    "bridge_es",
    "bridge_pt",
)

BRIDGE_CONDITION_MAP: dict[str, str] = {
    "bridge_fr": FRA,
    "bridge_es": SPA,
    "bridge_pt": POR,
}


@dataclass(frozen=True)
class TranslationPair:
    """A directed translation pair (src → tgt) with NLLB language codes."""

    src: str
    tgt: str

    def __str__(self) -> str:
        return f"{self.src}->{self.tgt}"

    def key(self) -> str:
        """Key format used in results JSON `scores` dict."""
        return f"{self.src}->{self.tgt}"


def primary_eval_pairs() -> list[TranslationPair]:
    """Ladin↔Italian in both directions (primary evaluation)."""
    return [TranslationPair(LADIN, ITA), TranslationPair(ITA, LADIN)]


def secondary_eval_pairs() -> list[TranslationPair]:
    """Ladin→{French, Spanish, Portuguese} (secondary, zero-shot generalization)."""
    return [TranslationPair(LADIN, lang) for lang in BRIDGE_LANGS]


def all_eval_pairs() -> list[TranslationPair]:
    """All evaluation pairs in the order they appear in result files."""
    return primary_eval_pairs() + secondary_eval_pairs()

"""Load and validate config from YAML; expose named constants for all hyperparameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Named constants — single source of truth for every hyperparameter
# ---------------------------------------------------------------------------
MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
MAX_SEQ_LEN: int = 128
BATCH_SIZE: int = 16
GRAD_ACCUM: int = 2
LEARNING_RATE: float = 2e-4
EPOCHS_BRIDGE: int = 1
EPOCHS_LADIN: int = 3
BEAM_SIZE: int = 5
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.1
TARGET_MODULES: list[str] = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
SEED: int = 42
SUBSET_SIZE: int = 18139

# NLLB language codes
LADIN: str = "lld_Latn"
ITA: str = "ita_Latn"
FRA: str = "fra_Latn"
SPA: str = "spa_Latn"
POR: str = "por_Latn"

# Dataset sources
LADIN_HF_REPO: str = "sfrontull/lld_valbadia-ita"
FLORES_HF_REPO: str = "openlanguagedata/flores_plus"
SYNTH_HF_REPO: str | None = None  # SDLad-Ita; confirm HF repo before ablation step

# OPUS Books HF dataset config names (alphabetical 2-letter pair codes)
# opus_books has direct Romance↔Italian pairs; opus-100 is English-centric only
OPUS_CONFIGS: dict[str, str] = {
    FRA: "fr-it",
    SPA: "es-it",
    POR: "it-pt",
}

# 2-letter keys used inside the OPUS-100 `translation` column
OPUS_LANG_KEYS: dict[str, str] = {
    FRA: "fr",
    SPA: "es",
    POR: "pt",
    ITA: "it",
}

# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return as dict; returns empty dict if file absent."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_drive_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return the Google Drive root path (Colab) or local fallback."""
    cfg = cfg or load_config()
    drive_root = cfg.get("paths", {}).get(
        "drive_root", "/content/drive/MyDrive/ladin-bridge-transfer"
    )
    return Path(drive_root)

---
name: dataset-sources
description: Exact data sources for the Ladin bridge-transfer MT project (ADIta_Lad, FLORES+, OPUS, synthetic)
metadata:
  type: project
---

Resolved data sources for the Ladin↔Italian NLLB bridge-transfer project:

- **ADIta_Lad** (primary, authentic Italian–Ladin, 18,139 pairs) = HF `sfrontull/lld_valbadia-ita` (18.1k records — count matches exactly). Variety: **Val Badia**.
- **Ladin variety = Val Badia** → must load the matching FLORES+ `lld_Latn` Val Badia config (FLORES+ also has Gherdëina; mismatch silently inflates error).
- **FLORES+ devtest** = HF `openlanguagedata/flores_plus` (gated, CC-BY-SA, needs HF login + license accept). Eval only, never in training. 1,012 multi-parallel sentences.
- **OPUS bridge** {fr,es,pt}↔it = OPUS mixed-domain, subsampled deterministically to 18,139 each.
- **SDLad-Ita** (synthetic, ablation only) = from the same line of work (paper "Exploring NLP Benchmarks in an Extremely Low-Resource Setting", arXiv 2509.03962, Frontull/Nuha et al). **Exact HF repo not yet confirmed** — ablation-only so not a blocker for the main pipeline; resolve before the ablation step.
- Base model is `facebook/nllb-200-distilled-600M` (note: the source paper used NLLB-1.3B; our project deliberately uses the 600M distilled).

See [[project-overview]] if present.

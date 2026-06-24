# 02 — LaTeX / Report Requirements (Final Write-Up)

Spec for the **8-page research report**, written **after** results exist. Read
`00_GENERAL_INSTRUCTIONS.md` for the design and `01_CODING_INSTRUCTIONS.md` for
the results files this report consumes. The report is filled from
`results/main_results.csv`, `results/similarity.csv`, and
`results/ablation.csv` — every number traces to a results file.

---

## 1. Template & format

- Use the **course-provided Overleaf template**. The conventions below assume an
  **ACL-style** template (the NLP-standard); **confirm the exact template and
  whether the 8-page limit excludes references/appendix** against the course
  guidelines before finalizing.
- **Max 8 pages** of body. Keep references (and any appendix, if permitted)
  outside the limit only if the template/guidelines say so — otherwise budget
  them in.
- Compile cleanly: **no undefined macros, no compile errors, no overfull boxes**
  that break the margin. Mentally check every `\ref`/`\cite` resolves.

---

## 2. Section structure (target page budget)

1. **Abstract** (~150 words) — problem, method, headline result.
2. **Introduction** (~0.75 pg) — low-resource Ladin; the bridging question (RQ1)
   and the similarity question (RQ2); contributions as a bullet list.
3. **Related Work** (~0.75 pg) — NLLB / massively multilingual MT; cross-lingual
   & intermediate-task transfer; low-resource Romance / Rhaeto-Romance MT; Ladin
   resources.
4. **Data** (~0.75 pg) — ADIta_Lad, SDLad-Ita (flagged synthetic), OPUS bridge
   subsets (size-matched, domain-mismatch caveat), FLORES+ devtest. A dataset
   table.
5. **Method** (~1.5 pg) — NLLB-200-distilled-600M; the **added `lld_Latn` code
   seeded from Italian** (state this clearly — it defines the zero-shot baseline
   as an Italian proxy); LoRA setup; the bridge→Ladin two-stage pipeline;
   similarity metrics (subword Jaccard + encoder cosine, both on the NLLB model
   itself).
6. **Experimental Setup** (~0.5 pg) — conditions, hyperparameters (table or
   prose), FLORES+ devtest (1,012 sentences), BLEU + chrF via sacrebleu (give the
   signature), seeds.
7. **Results** (~1.25 pg) — main results table; state whether bridging beat
   direct fine-tuning (RQ1).
8. **Analysis** (~1 pg) — similarity ranking vs. transfer ranking (RQ2), reported
   as a **weak directional correlate** with Spearman ρ noted as suggestive;
   qualitative error analysis (the 25/variant manual typing); ablation
   (synthetic data impact).
9. **Limitations** (~0.5 pg) — added (not native) Ladin → proxy zero-shot; only 3
   bridges → no power for H2; synthetic data only in ablation; OPUS domain
   mismatch; single base model, small data, noise.
10. **Conclusion** (~0.25 pg) — answer RQ1/RQ2 honestly; one line of future work.

Adjust budgets to fit 8 pages; never skip heading levels.

---

## 3. Tables (use `booktabs`; no vertical rules)

- **Main results** (`results/main_results.csv`): rows = conditions {Zero-shot
  (proxy), Direct, Bridge-FR, Bridge-ES, Bridge-PT}; columns grouped by direction
  (lld→ita, ita→lld) × {BLEU, chrF}; optionally the secondary lld→{fra,spa,por}
  in an appendix table. Bold the best per column. `\toprule/\midrule/\bottomrule`.
- **Similarity** (`results/similarity.csv`): rows = {Italian (ref), French,
  Spanish, Portuguese}; columns = Jaccard subword overlap, mean cosine distance,
  rank. Note in the caption these come from the NLLB tokenizer/encoder on FLORES+
  devtest.
- **Ablation** (`results/ablation.csv`): Direct and best-bridge × {ADIta_Lad,
  ADIta_Lad+SDLad-Ita} × scores.
- Captions are **above** tables and **self-contained** (readable without the
  body text).

---

## 4. Figures

- **Similarity-vs-transfer scatter:** x = similarity-to-Ladin (e.g., cosine
  proximity), y = transfer gain over Direct (chrF Δ), one point per bridge
  language; annotate with the Spearman ρ. Caption **below**, self-contained.
- Prefer vector output (PDF) from `matplotlib`; avoid rasterized plots.
- Optional: training/validation curves in an appendix if space allows.

---

## 5. Notation & style conventions

- Model/dataset/metric names in small caps or a consistent style:
  `\textsc{Nllb}`, `\textsc{Flores+}`, `\textsc{Adita\_Lad}`, `\textsc{Bleu}`,
  `\textsc{chrF}`.
- Define every acronym on first use (LoRA, MT, NLLB, BLEU, chrF, PEFT).
- Use `\emph{}` only for the first introduction of a technical term.
- Active voice ("We add a new language code…", not "A new code is added…").
- Consistent symbols throughout (don't mix e.g. $h_t$ and $\mathbf{h}_t$).
- Every empirical claim is backed by a results table/figure; every external
  claim by a citation.

---

## 6. Bibliography (BibTeX — verify each entry's author/year/venue)

Cite the original papers, not blogs. At minimum:
- **NLLB-200** (NLLB Team et al., 2022).
- **FLORES+ / FLORES-200** (and OLDI for the `+` expansion).
- The **Ladin datasets paper** (ADIta_Lad / SDLad-Ita, 2026) — verify exact
  authors/title/venue.
- **LoRA** (Hu et al., 2021).
- **PEFT** library (HuggingFace) — `@misc` with accessed date.
- **sacrebleu** (Post, 2018) and **chrF** (Popović, 2015).
- **OPUS** (Tiedemann, 2012).
- Optionally **Mediomatix / Romansh** work as related low-resource Rhaeto-Romance
  context.

Use the NLLB new-language recipe (David Dale tutorial) as a `@misc`
methodological reference for the added-language procedure if you describe it.

---

## 7. Numbers come from results files (no hand-entered values)

- Pull every score from `results/*.csv`; do not retype numbers from memory.
- Keep decimal precision consistent (e.g., one decimal for BLEU/chrF).
- If a second seed was run, report mean ± range for the key conditions.

---

## 8. Pre-submission checklist

- [ ] Within **8 pages** (per confirmed template rules).
- [ ] Compiles with **no errors / undefined refs / overfull margins**.
- [ ] Every table/figure referenced in text via `\ref`; captions self-contained.
- [ ] Zero-shot clearly labeled as **Italian-proxy** baseline.
- [ ] H2 reported as a **directional trend**, not a powered claim.
- [ ] Synthetic data appears **only** in the ablation, flagged as synthetic.
- [ ] Limitations section present and honest.
- [ ] All citations verified (author, year, venue); `.bib` used, no hardcoded
      references.
- [ ] Student IDs/emails and any required course metadata included.

# Intelligent Contract Review Assistant

**Break Through Tech AI Studio — Fall 2026 Challenge Project**
**Industry Partner:** Accenture | **Problem Type:** NLP (clause classification + risk scoring) | **Environment:** Free-tier Google Colab

---

## What This Project Does

Enterprise legal and procurement teams review tens of thousands of contracts a year, hunting for the handful of clauses that carry risk. This project builds an ML pipeline that:

1. **Detects and classifies** key clauses in a commercial contract using the [CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad) (510 real contracts, 41 clause categories, expert-annotated).
2. **Scores each detected clause** Low / Medium / High using a transparent, rule-based risk layer that can be audited and tuned.
3. **Rolls up** clause-level risk to a contract-level triage score so reviewers know which contracts to open first.

---

## Scope

| In scope | Out of scope (stretch only) |
|---|---|
| Clause detection / classification (CUAD labels) | LLM plain-language summarization |
| Explainable rule-based risk scoring | Trained risk or favorability model |
| End-to-end pipeline + evaluation | Deployment / hosted API |

---

## 15-Minute Quickstart (Google Colab)

**Prerequisites:** A Google account; no paid API or GPU credits needed.

```
1. Open Google Colab: https://colab.research.google.com
2. File → Open notebook → GitHub tab
   Paste this repo URL and open  notebooks/00_setup_check.ipynb
3. Runtime → Change runtime type → T4 GPU → Save
4. Runtime → Run all
```

The setup notebook verifies your GPU, installs dependencies, and loads a sample from CUAD via Hugging Face — no local data download needed.

Then work through the notebooks in order:

| Notebook | Purpose |
|---|---|
| `00_setup_check.ipynb` | Verify environment; load sample data |
| `01_explore_cuad.ipynb` | Guided EDA; understand the class-imbalance challenge |
| `02_baseline.ipynb` | TF-IDF / keyword baseline to beat |

Source code for the reusable pipeline lives in `src/`.

---

## Repository Layout

```
contract-review-assistant/
├── README.md
├── data/README.md          ← how to get CUAD (data is NOT committed here)
├── notebooks/
│   ├── 00_setup_check.ipynb
│   ├── 01_explore_cuad.ipynb
│   └── 02_baseline.ipynb
├── src/
│   ├── data_utils.py       ← loading, normalization, splitting
│   ├── chunking.py         ← sentence/paragraph windowing strategies
│   ├── model.py            ← clause detector (fine-tuned encoder stub)
│   ├── risk_scoring.py     ← four-signal risk scoring layer
│   └── pipeline.py         ← end-to-end: contract in → register out
├── docs/
│   ├── milestones.md       ← monthly deliverables
│   ├── risk_scoring.md     ← signal definitions and weight rationale
│   └── glossary.md         ← contract + ML terminology
├── requirements.txt
└── LICENSE
```

---

## Key Design Decisions

- **Chunk classification first.** Sentences/paragraphs are embedded and multi-label classified. This is the gentler on-ramp; span extraction is a documented stretch goal.
- **No paid APIs.** The fine-tuning job targets a base-size encoder (~110M params) that trains on a free Colab T4.
- **Explainable risk scores.** Risk is computed from four auditable signals (category base weight, absence flip, favorability keyword pass, template deviation). No black-box risk model.

---

## Dataset

CUAD v1 — CC BY 4.0 — public EDGAR filings, no PII. See [`data/README.md`](data/README.md) for access instructions.

---

## License

This project code is released under the MIT License. The CUAD dataset is CC BY 4.0 (see `LICENSE`).

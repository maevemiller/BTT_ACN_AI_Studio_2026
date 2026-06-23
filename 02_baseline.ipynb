# Risk-Scoring Layer

## Overview

CUAD provides no risk labels — it tells you *what* a clause is, not whether it is dangerous. Rather than inventing labels, this layer builds a transparent severity score from signals that are computable from artifacts already produced by the detection pipeline.

The score is **auditable logic, not a trained network** — a virtue in legal tech where explainability matters.

---

## Formula

```
clause_risk = base_weight[category] × (
    0.30 × absence_flip  +
    0.30 × favorability  +
    0.40 × template_deviation
)
```

Per-clause score range: **0 – 5**

| Bucket | Score range | Label |
|---|---|---|
| Low | ≤ 1.0 | Low |
| Medium | 1.0 – 3.0 | Medium |
| High | > 3.0 | High |

Contract-level score: `max(clause_scores)` — conservative triage (flag the contract if any clause is High).

---

## Signals

### 1. Category Base Weight

A static lookup table mapping each clause category to a severity multiplier (1–5), set by the Accenture advisor. High-stakes categories (indemnification, liability cap, IP assignment) receive higher base weights than lower-stakes administrative clauses.

```python
CATEGORY_BASE_WEIGHTS = {
    "Indemnification": 5,
    "Limitation of Liability": 5,
    "IP Ownership Assignment": 4,
    "Termination for Convenience": 4,
    "Auto-Renewal": 3,
    "Governing Law": 3,
    "Change of Control": 4,
    # ... extend with advisor input
}
```

### 2. Absence Flip (0 or 1)

Applies to **protective clauses only** (e.g., limitation of liability, indemnification). If the detector finds the clause is *absent* (`is_impossible = true`) and the category is protective, this signal fires (value = 1). The absence of a protective clause is the risk, not its presence.

```python
def absence_flip(category: str, detected: bool) -> float:
    if category in PROTECTIVE_CLAUSES and not detected:
        return 1.0
    return 0.0
```

### 3. Favorability (−1, 0, or +1)

A directional keyword pass on the detected clause span. Checks whether the language favors the reviewing party or the counterparty.

- `+1` — counterparty-favoring language (raises risk)
- `0` — neutral / balanced
- `−1` — our-party-favoring language (lowers risk)

Keyword lists are defined per category in `src/risk_scoring.py` and should be reviewed / extended with advisor input.

### 4. Template Deviation (0 – 1)

Cosine distance between the clause's embedding and the category's "standard" centroid (computed from all training-set examples of that clause type). A clause that looks very different from typical examples of its type may contain unusual terms worth flagging.

```
deviation = 1 - cosine_similarity(clause_embedding, category_centroid)
```

---

## Validation & Calibration

Because the score has no ground-truth labels, F1 cannot be computed. Instead:

1. Accenture advisor hand-ranks ~30 sampled clauses.
2. Compute **Spearman rank correlation** between advisor ranking and model score ordering.
3. Report **bucket agreement** (% of clauses where model bucket matches advisor bucket).
4. Run **sensitivity analysis**: shift each weight ±0.10 and measure how much the High/Medium boundary moves.

Frame results as *calibration against expert judgment*, not accuracy.

---

## Tuning Notes

- Start with the weights above (0.30 / 0.30 / 0.40) as the initial configuration.
- After the calibration exercise, adjust weights to maximize Spearman correlation on the advisor-ranked set.
- Document every weight change and the rationale — the audit trail is part of the deliverable.

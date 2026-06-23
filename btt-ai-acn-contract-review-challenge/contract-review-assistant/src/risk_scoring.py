"""
Four-signal risk scoring layer.

Formula:
    clause_risk = base_weight[category] × (
        0.30 × absence_flip  +
        0.30 × favorability  +
        0.40 × template_deviation
    )

Buckets:  ≤1.0 → Low | 1.0–3.0 → Medium | >3.0 → High
Contract: max(clause_scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

RiskLabel = Literal["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Configuration (advisor should review and tune)
# ---------------------------------------------------------------------------

CATEGORY_BASE_WEIGHTS: dict[str, float] = {
    "Indemnification": 5.0,
    "Limitation of Liability": 5.0,
    "IP Ownership Assignment": 4.0,
    "Termination for Convenience": 4.0,
    "Change of Control": 4.0,
    "Non-Compete": 3.5,
    "Exclusivity": 3.5,
    "Auto-Renewal": 3.0,
    "Governing Law": 3.0,
    "Most Favored Nation": 3.0,
    "Warranty Duration": 2.5,
    "Insurance": 2.5,
}

# Protective clauses: absence is the risk
PROTECTIVE_CLAUSES: set[str] = {
    "Limitation of Liability",
    "Insurance",
}

# Directional keywords per category.  Extend with advisor input.
COUNTERPARTY_KEYWORDS: dict[str, list[str]] = {
    "Indemnification": ["shall indemnify us", "indemnify and hold harmless the company"],
    "Limitation of Liability": ["unlimited liability", "no cap", "without limit"],
    "Non-Compete": ["worldwide", "in perpetuity", "unlimited"],
}
OUR_PARTY_KEYWORDS: dict[str, list[str]] = {
    "Indemnification": ["shall indemnify the vendor", "mutual indemnification"],
    "Limitation of Liability": ["liability shall not exceed", "aggregate cap"],
}

SIGNAL_WEIGHTS = {"absence": 0.30, "favorability": 0.30, "deviation": 0.40}
BUCKET_THRESHOLDS = {"medium": 1.0, "high": 3.0}


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

def absence_flip(category: str, detected: bool) -> float:
    """Returns 1.0 if a protective clause is absent, else 0.0."""
    if category in PROTECTIVE_CLAUSES and not detected:
        return 1.0
    return 0.0


def favorability_score(category: str, clause_text: str) -> float:
    """
    Returns +1.0 (counterparty-favoring, raises risk),
             0.0 (neutral), or -1.0 (our-party-favoring, lowers risk).
    """
    text_lower = clause_text.lower()
    for kw in COUNTERPARTY_KEYWORDS.get(category, []):
        if kw.lower() in text_lower:
            return 1.0
    for kw in OUR_PARTY_KEYWORDS.get(category, []):
        if kw.lower() in text_lower:
            return -1.0
    return 0.0


def template_deviation(
    clause_embedding: np.ndarray,
    category_centroid: np.ndarray,
) -> float:
    """Cosine distance between a clause embedding and its category centroid."""
    num = np.dot(clause_embedding, category_centroid)
    denom = (np.linalg.norm(clause_embedding) * np.linalg.norm(category_centroid)) + 1e-9
    cosine_sim = num / denom
    return float(1.0 - cosine_sim)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ClauseRiskResult:
    category: str
    detected: bool
    clause_text: str
    score: float
    label: RiskLabel
    signals: dict[str, float] = field(default_factory=dict)


def score_clause(
    category: str,
    detected: bool,
    clause_text: str,
    clause_embedding: np.ndarray | None = None,
    category_centroid: np.ndarray | None = None,
) -> ClauseRiskResult:
    base = CATEGORY_BASE_WEIGHTS.get(category, 2.0)

    sig_absence = absence_flip(category, detected)
    sig_favor = favorability_score(category, clause_text) if detected else 0.0

    if detected and clause_embedding is not None and category_centroid is not None:
        sig_deviation = template_deviation(clause_embedding, category_centroid)
    else:
        sig_deviation = 0.5  # default when embeddings unavailable

    w = SIGNAL_WEIGHTS
    modifier = w["absence"] * sig_absence + w["favorability"] * sig_favor + w["deviation"] * sig_deviation
    modifier = max(0.0, modifier)  # clamp; favorability can pull below zero

    raw_score = base * modifier

    if raw_score <= BUCKET_THRESHOLDS["medium"]:
        label: RiskLabel = "Low"
    elif raw_score <= BUCKET_THRESHOLDS["high"]:
        label = "Medium"
    else:
        label = "High"

    return ClauseRiskResult(
        category=category,
        detected=detected,
        clause_text=clause_text,
        score=round(raw_score, 3),
        label=label,
        signals={
            "base_weight": base,
            "absence_flip": sig_absence,
            "favorability": sig_favor,
            "template_deviation": sig_deviation,
        },
    )


def contract_risk_rollup(clause_results: list[ClauseRiskResult]) -> tuple[float, RiskLabel]:
    """Conservative rollup: max clause score determines contract label."""
    if not clause_results:
        return 0.0, "Low"
    max_score = max(r.score for r in clause_results)
    if max_score <= BUCKET_THRESHOLDS["medium"]:
        label: RiskLabel = "Low"
    elif max_score <= BUCKET_THRESHOLDS["high"]:
        label = "Medium"
    else:
        label = "High"
    return round(max_score, 3), label

"""
End-to-end pipeline: contract text in → risk-tagged clause register out.

Usage:
    from src.pipeline import ContractReviewPipeline, ClauseRegisterRow

    pipe = ContractReviewPipeline.from_pretrained("checkpoints/model.pt")
    register, contract_label = pipe.run(contract_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .chunking import sentence_chunks, Chunk
from .data_utils import normalize_text
from .model import ClauseClassifier, load_model, load_tokenizer, TARGET_CATEGORIES
from .risk_scoring import (
    ClauseRiskResult,
    RiskLabel,
    contract_risk_rollup,
    score_clause,
)


@dataclass
class ClauseRegisterRow:
    contract_name: str
    category: str
    detected: bool
    clause_excerpt: str
    risk_score: float
    risk_label: RiskLabel
    signals: dict[str, float] = field(default_factory=dict)


class ContractReviewPipeline:
    """
    Wraps chunking → detection → risk scoring into a single callable.

    Parameters
    ----------
    model:            Fine-tuned ClauseClassifier.
    tokenizer:        Matching HuggingFace tokenizer.
    category_centroids: Dict mapping category name → mean embedding vector.
    detection_threshold: Sigmoid threshold for calling a clause present.
    """

    def __init__(
        self,
        model: ClauseClassifier,
        tokenizer,
        category_centroids: dict[str, np.ndarray] | None = None,
        detection_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.category_centroids = category_centroids or {}
        self.detection_threshold = detection_threshold

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        model_name: str = "microsoft/deberta-v3-base",
        centroids_path: str | Path | None = None,
        **kwargs,
    ) -> "ContractReviewPipeline":
        model = load_model(model_path, model_name=model_name)
        tokenizer = load_tokenizer(model_name)
        centroids: dict[str, np.ndarray] = {}
        if centroids_path:
            raw = np.load(centroids_path, allow_pickle=True).item()
            centroids = {k: np.array(v) for k, v in raw.items()}
        return cls(model=model, tokenizer=tokenizer, category_centroids=centroids, **kwargs)

    def run(
        self,
        contract_text: str,
        contract_name: str = "unknown",
    ) -> tuple[list[ClauseRegisterRow], RiskLabel]:
        """
        Process one contract.

        Returns:
            register:       One ClauseRegisterRow per TARGET_CATEGORY.
            contract_label: Contract-level risk label (max of clause scores).
        """
        text = normalize_text(contract_text)
        chunks: list[Chunk] = sentence_chunks(text)

        # --- Detection (stub: replace with real inference) ---
        detected_clauses: dict[str, str] = self._detect_clauses(chunks)

        # --- Risk scoring ---
        clause_results: list[ClauseRiskResult] = []
        for category in TARGET_CATEGORIES:
            detected = category in detected_clauses
            excerpt = detected_clauses.get(category, "")
            centroid = self.category_centroids.get(category)
            # TODO: compute embedding of excerpt for template_deviation
            result = score_clause(
                category=category,
                detected=detected,
                clause_text=excerpt,
                clause_embedding=None,
                category_centroid=centroid,
            )
            clause_results.append(result)

        _, contract_label = contract_risk_rollup(clause_results)

        register = [
            ClauseRegisterRow(
                contract_name=contract_name,
                category=r.category,
                detected=r.detected,
                clause_excerpt=r.clause_text[:300],
                risk_score=r.score,
                risk_label=r.label,
                signals=r.signals,
            )
            for r in clause_results
        ]

        return register, contract_label

    def to_dataframe(self, register: list[ClauseRegisterRow]) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in register])

    # ------------------------------------------------------------------
    # Internal helpers (stubs)
    # ------------------------------------------------------------------

    def _detect_clauses(self, chunks: list[Chunk]) -> dict[str, str]:
        """
        Run the fine-tuned model over chunks and return a dict of
        {category: best_matching_chunk_text} for detected categories.

        TODO: implement actual inference loop.
        """
        # Placeholder — always returns empty dict until model is trained
        return {}

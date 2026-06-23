"""
Clause detection model: fine-tuned transformer encoder for multi-label
chunk classification over CUAD clause categories.

Stub structure — fill in the training loop and evaluation in the notebooks,
then refactor reusable pieces here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# Clause categories to target (expand with advisor input)
TARGET_CATEGORIES: list[str] = [
    "Indemnification",
    "Limitation of Liability",
    "IP Ownership Assignment",
    "Termination for Convenience",
    "Auto-Renewal",
    "Governing Law",
    "Change of Control",
    "Non-Compete",
    "Exclusivity",
    "Most Favored Nation",
    "Warranty Duration",
    "Insurance",
]


class ClauseClassifier(nn.Module):
    """
    Multi-label classifier over a pre-trained encoder.

    Architecture:
        encoder (frozen or fine-tuned) → mean-pool CLS token → dropout → linear head
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = len(TARGET_CATEGORIES),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


def load_tokenizer(model_name: str = "microsoft/deberta-v3-base") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def save_model(model: ClauseClassifier, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, **kwargs) -> ClauseClassifier:
    model = ClauseClassifier(**kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

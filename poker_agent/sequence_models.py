from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SequenceBatch:
    action_token_ids: Any
    numeric_state: Any
    attention_mask: Any
    labels: Any | None = None


def build_focal_loss(class_weights: Any, gamma: float = 2.0) -> Any:
    """Build focal loss for imbalanced neural poker-action models.

    Torch is an optional research dependency, so it is imported lazily.
    """
    import torch
    import torch.nn.functional as functional

    weights = torch.as_tensor(class_weights, dtype=torch.float32)

    def loss_fn(logits: Any, targets: Any) -> Any:
        local_weights = weights.to(logits.device)
        ce_loss = functional.cross_entropy(logits, targets, weight=local_weights, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** gamma) * ce_loss).mean()

    return loss_fn


def build_transformer_policy(
    *,
    num_action_tokens: int,
    numeric_dim: int,
    num_classes: int,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.15,
) -> Any:
    """Create a compact transformer policy for temporal betting histories.

    This is intended for research experiments once full hand/action sequences are
    available. It combines a transformer encoder over action tokens with a small
    numeric-state encoder.
    """
    import torch
    from torch import nn

    class TransformerPokerPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(num_action_tokens, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.numeric_encoder = nn.Sequential(
                nn.LayerNorm(numeric_dim),
                nn.Linear(numeric_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        def forward(self, batch: SequenceBatch) -> Any:
            token_embeddings = self.token_embedding(batch.action_token_ids)
            key_padding_mask = batch.attention_mask == 0
            sequence_hidden = self.sequence_encoder(token_embeddings, src_key_padding_mask=key_padding_mask)
            valid_lengths = batch.attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled_sequence = (sequence_hidden * batch.attention_mask.unsqueeze(-1)).sum(dim=1) / valid_lengths
            numeric_hidden = self.numeric_encoder(batch.numeric_state)
            return self.classifier(torch.cat([pooled_sequence, numeric_hidden], dim=-1))

    return TransformerPokerPolicy()

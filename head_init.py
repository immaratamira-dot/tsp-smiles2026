"""
head_init.py — Final layer initialization (student-implemented).

Small-scale initialization keeps the initial logits near zero, which means
cross-entropy loss starts near log(100) ~ 4.6 — the worst-case uniform
prediction. This is intentional: it avoids exploding loss at step 1, which
would cause large SPSA gradient estimates and destabilise early updates.
"""

import torch
import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the 100-class head with small-scale weights and zero bias."""
    nn.init.xavier_uniform_(layer.weight)
    layer.weight.data.mul_(0.01)  # Scale down to keep initial logits small.
    nn.init.zeros_(layer.bias)

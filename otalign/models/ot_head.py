import torch
import torch.nn as nn


class OTHead(nn.Module):
    """Base class for OT Heads."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearOTHead(OTHead):
    """A simple linear projection head for OT."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.proj = nn.Linear(config["input_dim"], config["output_dim"])

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(embeddings)


class MLPOTHead(OTHead):
    """An MLP projection head for OT."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.mlp = nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config["hidden_dim"], config["output_dim"]),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(embeddings)


def get_ot_head(config: dict) -> OTHead:
    """Factory function to get an OT head."""
    head_type = config.get("type", "linear")
    if head_type == "linear":
        return LinearOTHead(config)
    elif head_type == "mlp":
        return MLPOTHead(config)
    else:
        raise ValueError(f"Unknown OT head type: {head_type}")

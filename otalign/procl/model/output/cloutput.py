from dataclasses import dataclass

import torch
from transformers.utils.generic import ModelOutput


@dataclass
class CLPredictionOutput(ModelOutput):
    """ """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_state: torch.FloatTensor | None = None
    attentions: torch.FloatTensor | None = None

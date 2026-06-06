"""Ankh backbone with a ConvBert contrastive-learning head (AnkhCL).

The base T5 encoder weights come from Ankh-Large; the ConvBert head is trained
on top for a contrastive objective. At inference we load with
``freeze_base=True, is_scratch=False`` and read the head output.

Ported from DeepFoldProtein/plmMSA (``src/procl/model/ankh.py``): the base
encoder now goes through the shared T5 backend helper, so it uses the
``turbot5`` FlashAttention encoder when available and falls back to vanilla
``transformers.T5EncoderModel`` otherwise. The ``freeze_base`` / ``is_scratch``
constructor flags are retained for backward compatibility with existing
checkpoints and callers.
"""

import torch
import torch.nn as nn

from otalign.models.plm_adaptors._t5_backend import (
    T5_CONFIG_CLS,
    T5_ENCODER_CLS,
    T5_PRETRAINED_CLS,
)
from otalign.procl.model.head.convbert import ConvBertForHead
from otalign.procl.model.output.cloutput import CLPredictionOutput


# Base T5 classes resolved from the active backend (turbot5 flash when usable,
# else stock transformers). AnkhCL must subclass the matching `T5PreTrainedModel`
# so `from_pretrained` builds the right config (turbot5's adds `use_triton`,
# `attention_type`, ...).
T5Config = T5_CONFIG_CLS
T5PreTrainedModel = T5_PRETRAINED_CLS


_HEAD_INPUT_DIM = 1536
_HEAD_HIDDEN_DIM = _HEAD_INPUT_DIM // 2
_HEAD_NUM_HEADS = 8
_HEAD_NUM_HIDDEN_LAYERS = 1
_HEAD_KERNEL_SIZE = 7


class AnkhCL(T5PreTrainedModel):
    def __init__(self, config: T5Config, freeze_base: bool = True, is_scratch: bool = False):
        super().__init__(config)
        self.transformer = T5_ENCODER_CLS(config)
        self.freeze_base = freeze_base
        self.d_model = config.d_model
        if self.freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        if not is_scratch:
            self.head = ConvBertForHead(
                input_dim=_HEAD_INPUT_DIM,
                nhead=_HEAD_NUM_HEADS,
                hidden_dim=_HEAD_HIDDEN_DIM,
                num_hidden_layers=_HEAD_NUM_HIDDEN_LAYERS,
                kernel_size=_HEAD_KERNEL_SIZE,
                dropout=0.0,
            )
        self.activation = nn.Tanh()

    def add_convbert_for_train(self, dropout: float):
        self.head = ConvBertForHead(
            input_dim=_HEAD_INPUT_DIM,
            nhead=_HEAD_NUM_HEADS,
            hidden_dim=_HEAD_HIDDEN_DIM,
            num_hidden_layers=_HEAD_NUM_HIDDEN_LAYERS,
            kernel_size=_HEAD_KERNEL_SIZE,
            dropout=dropout,
        )

    def _compute_hidden_state(self, tokens, attention_mask):
        return self._extract_hidden_state(tokens, attention_mask)

    def _extract_hidden_state(self, tokens, attention_mask) -> torch.Tensor:
        return self.transformer(
            tokens,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=True,
        ).last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        last_hidden_state = self._compute_hidden_state(input_ids, attention_mask)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())
        last_hidden_state = self.head(last_hidden_state, extended_attention_mask)
        last_hidden_state = self.activation(last_hidden_state)
        return CLPredictionOutput(
            loss=None,
            logits=None,
            last_hidden_state=last_hidden_state,
            attentions=None,
        )

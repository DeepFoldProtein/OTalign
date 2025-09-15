import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel, T5PreTrainedModel

from otalign.procl.model.head.convbert import ConvBertForHead
from otalign.procl.model.output.cloutput import CLPredictionOutput


class AnkhCL(T5PreTrainedModel):
    def __init__(self, config: T5Config, freeze_base, is_scratch):
        super().__init__(config)
        self.transformer = T5EncoderModel(config)
        self.freeze_base = freeze_base
        self.d_model = config.d_model
        if self.freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
            # self.transformer.eval()
        if not is_scratch:
            self.head = ConvBertForHead(
                input_dim=1536,
                nhead=8,
                hidden_dim=1536 // 2,
                num_hidden_layers=1,
                kernel_size=7,
                dropout=0,
            )
        self.activation = nn.Tanh()

    def add_convbert_for_train(self, dropout: float):
        self.head = ConvBertForHead(
            input_dim=1536,
            nhead=8,
            hidden_dim=1536 // 2,
            num_hidden_layers=1,
            kernel_size=7,
            dropout=dropout,
        )

    def _compute_hidden_state(self, tokens, attention_mask):
        return self._extract_hidden_state(tokens, attention_mask)

    def _extract_hidden_state(self, tokens, attention_mask) -> torch.Tensor:
        last_hidden_state = self.transformer(
            tokens,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=True,
        ).last_hidden_state
        return last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        last_hidden_state = self._compute_hidden_state(input_ids, attention_mask)

        last_hidden_state = last_hidden_state
        input_shape = input_ids.shape
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        last_hidden_state = self.head(last_hidden_state, extended_attention_mask)
        last_hidden_state = self.activation(last_hidden_state)
        return CLPredictionOutput(
            loss=None,
            logits=None,
            hidden_state=last_hidden_state,
            attentions=None,
        )

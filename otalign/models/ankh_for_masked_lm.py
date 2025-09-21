import torch.nn as nn
from transformers import T5Config, T5EncoderModel, T5PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


class AnkhForMaskedLM(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.transformer = T5EncoderModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

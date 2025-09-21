from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class MLMCollator:
    """
    Data collator for Masked Language Modeling.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    max_length: int = 1024

    def __call__(self, seqs: List[str]) -> Dict[str, Any]:
        """
        Tokenize and mask sequences for MLM.
        """
        tokenized_inputs = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tokenized_inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"Tokenizer output 'input_ids' must be a torch.Tensor, but got {type(input_ids)}")

        inputs, labels = self.mask_tokens(input_ids)
        return {"input_ids": inputs, "labels": labels, "attention_mask": tokenized_inputs["attention_mask"]}

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        masked_inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token_id is not None:
            assert isinstance(self.tokenizer.pad_token_id, int)
            padding_mask = labels == self.tokenizer.pad_token_id
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        if self.tokenizer.mask_token_id is not None:
            assert isinstance(self.tokenizer.mask_token_id, int)
            masked_inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        masked_inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return masked_inputs, labels

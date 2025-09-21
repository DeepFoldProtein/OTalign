from typing import Callable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import BasePLMAdaptor, EmbeddingOutput
from .policies import make_mask_from_lengths


def _split_into_batches(xs: Sequence[str], batch_size: int) -> List[List[str]]:
    return [list(xs[i : i + batch_size]) for i in range(0, len(xs), batch_size)]


class HFEncoderAdaptor(BasePLMAdaptor):
    """
    Generic Hugging Face encoder-like adaptor.
    It accepts a tokenizer and a model, and a `trim_policy` callable that
    maps token ids matrix [B, T] -> a NEW [B, L] ids with specials removed
    (and returns also per-sequence lengths if needed).
    It then selects the corresponding hidden states and returns [B, L, D].
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        *,
        trim_policy: Callable[..., Tuple[torch.Tensor, List[int], torch.Tensor]],
        token_field: str = "input_ids",
        attention_mask_field: str = "attention_mask",
        model_output_field: str = "last_hidden_state",
        pad_to_multiple_of: Optional[int] = None,
        use_encoder: bool = True,  # if False, call model(**inputs) anyway (for decoder-only)
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.trim_policy = trim_policy
        self.token_field = token_field
        self.attention_mask_field = attention_mask_field
        self.model_output_field = model_output_field
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_encoder = use_encoder

    def encode(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        fp16: bool = False,
        disable_grad: bool = True,
    ) -> EmbeddingOutput:
        model = self.model
        tok = self.tokenizer
        device = device or next(model.parameters()).device
        autocast_dtype = torch.float16 if fp16 else None
        all_embeds: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_lengths: List[int] = []

        if dtype is None:
            dtype = next(model.parameters()).dtype

        was_training = model.training
        model.eval()

        cm = torch.no_grad() if disable_grad else torch.enable_grad()
        with cm:
            for chunk in _split_into_batches(sequences, batch_size):
                seq_batch = list(chunk)
                if hasattr(self, "preproc_batch") and callable(getattr(self, "preproc_batch")):
                    seq_batch = self.preproc_batch(seq_batch)

                enc = tok(seq_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=True, pad_to_multiple_of=self.pad_to_multiple_of)
                input_ids: torch.Tensor = enc[self.token_field].to(device)
                attn_mask: torch.Tensor = enc[self.attention_mask_field].to(device)

                # Forward
                with torch.autocast(device_type=str(device).split(":")[0], dtype=autocast_dtype) if fp16 else torch.enable_grad():
                    outputs = model(**{self.token_field: input_ids, self.attention_mask_field: attn_mask, "output_hidden_states": True})  # type: ignore
                    if hasattr(outputs, self.model_output_field):
                        hidden: torch.Tensor = getattr(outputs, self.model_output_field)
                    else:
                        # Fallback for MaskedLMOutput which has 'hidden_states' tuple
                        hidden: torch.Tensor = outputs.hidden_states[-1]

                # Trim specials on token_ids, then select same positions in hidden
                trimmed_ids, lengths, keep_mask = self.trim_policy(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                # keep_mask: [B, T] True where original token should be kept as a residue
                # Select hidden states
                B, _, D = hidden.shape
                maxL = int(keep_mask.sum(dim=1).max().item()) if B > 0 else 0
                # assert isinstance(maxL, int)
                # pack kept positions per row into a padded [B, maxL, D]
                kept_embeds = []
                for b in range(B):
                    sel = hidden[b][keep_mask[b]]  # [L_b, D]
                    if sel.numel() == 0:
                        pad = torch.zeros((0, D), device=hidden.device, dtype=hidden.dtype)
                    else:
                        pad = sel
                    if sel.size(0) < maxL:
                        pad = torch.cat([pad, hidden.new_zeros((maxL - sel.size(0), D))], dim=0)
                    kept_embeds.append(pad.unsqueeze(0))
                kept_embeds = torch.cat(kept_embeds, dim=0) if kept_embeds else torch.empty((B, 0, D), device=hidden.device, dtype=hidden.dtype)
                all_embeds.append(kept_embeds)
                all_masks.append(make_mask_from_lengths(lengths, device=hidden.device))
                all_lengths.extend([int(x) for x in lengths])

        if was_training:
            model.train()

        # Concatenate
        if all_embeds:
            max_len = max(e.shape[1] for e in all_embeds)

            padded_embeds = [torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[1])) for e in all_embeds]
            residue_embeddings = torch.cat(padded_embeds, dim=0)

            padded_masks = [torch.nn.functional.pad(m, (0, max_len - m.shape[1])) for m in all_masks]
            attention_mask = torch.cat(padded_masks, dim=0)
        else:
            residue_embeddings = torch.empty((0, 0, 0), device=device, dtype=dtype)
            attention_mask = torch.empty((0, 0), device=device, dtype=torch.bool)

        return EmbeddingOutput(residue_embeddings=residue_embeddings.to(dtype=dtype), attention_mask=attention_mask, per_sequence_lengths=all_lengths, extras={})

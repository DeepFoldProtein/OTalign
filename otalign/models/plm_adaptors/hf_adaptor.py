import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import BasePLMAdaptor, EmbeddingOutput
from .policies import make_mask_from_lengths, pack_left_aligned


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
        fp16_unsafe: bool = False,  # T5/ProtT5/Ankh overflow to NaN in fp16 — use bf16 instead
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.trim_policy = trim_policy
        self.token_field = token_field
        self.attention_mask_field = attention_mask_field
        self.model_output_field = model_output_field
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_encoder = use_encoder
        self.fp16_unsafe = fp16_unsafe

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

        # Resolve the reduced-precision autocast dtype. fp16 overflows to NaN on
        # T5-family encoders (ProtT5 / Ankh / AnkhCL); for those we substitute
        # bf16 (preferred) or fall back to fp32 rather than emit garbage.
        autocast_dtype: Optional[torch.dtype] = None
        if fp16:
            if self.fp16_unsafe:
                if device.type == "cuda" and torch.cuda.is_bf16_supported():
                    autocast_dtype = torch.bfloat16
                    warnings.warn(f"{type(model).__name__}: fp16 is numerically unstable for this model; using bfloat16 instead.", stacklevel=2)
                else:
                    warnings.warn(f"{type(model).__name__}: fp16 is numerically unstable for this model and bf16 is unavailable; running in fp32.", stacklevel=2)
            else:
                autocast_dtype = torch.float16
        use_autocast = autocast_dtype is not None
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
                with torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext():
                    model_kwargs = {self.token_field: input_ids, self.attention_mask_field: attn_mask}
                    try:
                        outputs = model(**model_kwargs, output_hidden_states=True)
                    except TypeError:
                        outputs = model(**model_kwargs)
                    if hasattr(outputs, self.model_output_field):
                        hidden: torch.Tensor = getattr(outputs, self.model_output_field)
                    else:
                        # Fallback for MaskedLMOutput which has 'hidden_states' tuple
                        hidden: torch.Tensor = outputs.hidden_states[-1]

                # Trim specials on token_ids, then select same positions in hidden
                trimmed_ids, lengths, keep_mask = self.trim_policy(input_ids, attn_mask)
                # keep_mask: [B, T] True where original token should be kept as a residue
                # Select hidden states
                # Pack kept positions per row into a left-aligned padded [B, maxL, D]
                # with the same vectorized scatter the trim policy uses for ids.
                # maxL comes from `lengths` (already on CPU) to avoid a per-batch sync.
                maxL = max(lengths) if lengths else 0
                kept_embeds = pack_left_aligned(hidden, keep_mask, maxL)
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

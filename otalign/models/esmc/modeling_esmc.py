"""Clean-room ESM-C encoder matching the official esm-SDK checkpoint layout.

Reference: github.com/evolutionaryscale/esm (``esm/models/esmc.py``,
``esm/layers/{transformer_stack,blocks,attention,rotary,regression_head}.py``).
Module/parameter names mirror the reference so that the official state dicts at
``biohub/esmc-{300m,600m,6b}-2024-12`` load with ``strict=True``.

Only the inference path is reimplemented: standard (non-flash) multi-head
attention via ``scaled_dot_product_attention`` and pure-PyTorch rotary
embeddings, which is numerically identical to the reference's eager path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# (d_model, n_heads, n_layers) per released size, from esm/pretrained.py.
ESMC_CONFIGS: dict[str, dict[str, int]] = {
    "300m": {"d_model": 960, "n_heads": 15, "n_layers": 30},
    "600m": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
    "6b": {"d_model": 2560, "n_heads": 40, "n_layers": 80},
}
_VOCAB_SIZE = 64  # embedding rows; tokenizer uses ids 0..32 (see SEQUENCE_VOCAB)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """GPT-NeoX-style rotary embedding (non-interleaved), cos/sin built in fp32."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seqlen: int, device, dtype):
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (B, S, H, D)
        cos, sin = self._cos_sin(q.size(1), q.device, q.dtype)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.layernorm_qkv = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_ln = nn.LayerNorm(d_model, bias=bias) if qk_layernorm else nn.Identity()
        self.k_ln = nn.LayerNorm(d_model, bias=bias) if qk_layernorm else nn.Identity()
        self.rotary = RotaryEmbedding(self.d_head)

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor | None) -> torch.Tensor:
        qkv = self.layernorm_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)
        B, L, _ = q.shape
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        # (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)

        attn_mask = None
        if seq_id is not None:
            # True where two positions may attend (matches the reference's `==`,
            # so pad-vs-pad stays True and no query row is fully masked -> no NaN).
            same = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)  # (B, S, S)
            attn_mask = same.unsqueeze(1)  # (B, 1, S, S)

        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ctx = ctx.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(ctx)


def _swiglu_hidden(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def _swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Sequential:
    hidden = _swiglu_hidden(expansion_ratio, d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden * 2, bias=bias),
        SwiGLU(),
        nn.Linear(hidden, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, residue_scaling_factor: float, expansion_ratio: float = 8 / 3, bias: bool = False, qk_layernorm: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm)
        self.ffn = _swiglu_ln_ffn(d_model, expansion_ratio, bias)
        self.scaling_factor = residue_scaling_factor

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(x, seq_id) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        scaling = math.sqrt(n_layers / 36)
        self.blocks = nn.ModuleList(
            [UnifiedTransformerBlock(d_model, n_heads, scaling) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, seq_id)
        return self.norm(x)


@dataclass
class ESMCEncoderOutput:
    last_hidden_state: torch.Tensor


class ESMCEncoder(nn.Module):
    """ESM-C encoder. ``forward`` returns the post-final-norm residue embeddings.

    The ``sequence_head`` (masked-LM head) is built so the official checkpoint
    loads strictly; it is not needed for embeddings but is exposed for
    verification / masked-LM use.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, vocab_size: int = _VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = TransformerStack(d_model, n_heads, n_layers)
        self.sequence_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: object) -> ESMCEncoderOutput:
        seq_id = attention_mask.to(torch.bool) if attention_mask is not None else None
        x = self.embed(input_ids)
        x = self.transformer(x, seq_id)
        return ESMCEncoderOutput(last_hidden_state=x)

    def sequence_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.forward(input_ids, attention_mask)
        return self.sequence_head(out.last_hidden_state)


def load_esmc(size: str, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> ESMCEncoder:
    """Build an ESMCEncoder and strict-load official weights from HuggingFace.

    Args:
        size: one of ``"300m"``, ``"600m"``, ``"6b"``.
        device / dtype: where/what precision to place the loaded model.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    if size not in ESMC_CONFIGS:
        raise ValueError(f"Unknown ESMC size {size!r}; expected one of {sorted(ESMC_CONFIGS)}")
    repo_id = f"biohub/esmc-{size}-2024-12"
    ckpts = [f for f in list_repo_files(repo_id) if f.endswith((".pth", ".pt", ".bin"))]
    if not ckpts:
        raise FileNotFoundError(f"No state-dict checkpoint found in {repo_id}")
    state_dict = torch.load(hf_hub_download(repo_id, ckpts[0]), map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = ESMCEncoder(**ESMC_CONFIGS[size])
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device=torch.device(device), dtype=dtype) if dtype is not None else model.to(torch.device(device))

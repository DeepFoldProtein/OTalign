"""T5 encoder backend selection with optional FlashAttention via ``turbot5``.

The reference server (DeepFoldProtein/plmMSA) loads every T5-family backbone
(ProtT5, Ankh-Large, and the Ankh-CL contrastive backbone) through
``turbot5.T5EncoderForMaskedLM(..., attention_type="flash")`` for lower-memory,
faster attention on long protein sequences.

Here we make that an *optional* optimization: when ``turbot5`` is importable and
a CUDA device is present we use the flash encoder, otherwise we fall back to the
vanilla ``transformers.T5EncoderModel``. Both expose ``.last_hidden_state``, so
the rest of the adaptor stack is agnostic to which backend is in use.
"""

from __future__ import annotations

from typing import Any

import torch


try:  # turbot5 is an optional dependency (the `flash` extra).
    import turbot5  # type: ignore  # noqa: F401

    TURBOT5_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means no flash backend.
    TURBOT5_AVAILABLE = False


def flash_backend_usable() -> bool:
    """Whether the turbot5 flash backend should actually be used.

    FlashAttention kernels require a CUDA device; on CPU-only hosts (and CI) we
    stay on the vanilla transformers encoder even if turbot5 is installed.
    """
    return TURBOT5_AVAILABLE and torch.cuda.is_available()


def t5_flash_kwargs() -> dict[str, Any]:
    """``from_pretrained`` kwargs that request the flash backend, when usable.

    Returned dict is empty for the fallback path so that callers never pass
    ``attention_type`` to a vanilla ``T5Config`` (which does not accept it).
    """
    return {"attention_type": "flash"} if flash_backend_usable() else {}


def _resolve_t5_base():
    """Resolve the active T5 base classes ``(Config, EncoderModel, PreTrainedModel)``.

    On the flash path we use turbot5's own subclasses: turbot5's ``T5Config``
    defaults the extra fields its kernels read (``use_triton``,
    ``attention_type``, ...), and its ``T5PreTrainedModel`` wires that config as
    ``config_class`` so ``from_pretrained`` builds a turbot5 config. The
    fallback uses the stock transformers classes.
    """
    if flash_backend_usable():
        from turbot5.heads.t5_heads import (  # type: ignore
            T5Config,
            T5EncoderForMaskedLM,
            T5PreTrainedModel,
        )

        return T5Config, T5EncoderForMaskedLM, T5PreTrainedModel

    from transformers import T5Config, T5EncoderModel, T5PreTrainedModel

    return T5Config, T5EncoderModel, T5PreTrainedModel


# Resolved once per process. Class-definition sites (e.g. AnkhCL's base class)
# need a concrete class at import time, so these are module-level constants.
T5_CONFIG_CLS, T5_ENCODER_CLS, T5_PRETRAINED_CLS = _resolve_t5_base()


def load_t5_pretrained(cls, model_name: str, **hf_kwargs: Any):
    """``cls.from_pretrained`` with the flash backend requested when usable.

    Single seam that owns flash-kwargs injection for every T5-family class
    (the bare encoder via :func:`load_t5_encoder`, and the AnkhCL wrapper). The
    ``attention_type`` kwarg is added only when the flash backend is active, so
    callers never pass it to a vanilla ``T5Config``.
    """
    return cls.from_pretrained(model_name, **t5_flash_kwargs(), **hf_kwargs)


def load_t5_encoder(model_name: str, *, output_attentions: bool = False, **hf_kwargs: Any):
    """Load a T5 encoder, preferring the turbot5 flash backend when usable.

    Returns a model whose forward yields an object with ``.last_hidden_state``.
    """
    return load_t5_pretrained(T5_ENCODER_CLS, model_name, output_attentions=output_attentions, **hf_kwargs)

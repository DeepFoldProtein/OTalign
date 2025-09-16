from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CacheConfig:
    dataset_name: str
    model_name: str
    adaptor_name: str
    adaptor_version: str
    dtype: str  # "fp16" | "fp32" | "bf16"
    policy: str  # e.g., "drop_first_last_active"
    tokenizer_pretok: Optional[str]
    shard_size: int  # rows per shard
    extra: Optional[dict] = None

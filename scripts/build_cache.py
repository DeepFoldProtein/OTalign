import argparse
import typing
from pathlib import Path

import torch
from tqdm.auto import tqdm

from otalign.cache.config import CacheConfig
from otalign.cache.lmdb_writer import LMDBCacheWriter
from otalign.cache.npz_writer import NPZCacheWriter
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint
from scripts.dataset_utils import iter_pairs_from_dataset


def build_cache(
    dataset: str,
    model: str,
    output_root: str,
    base_model_for_checkpoint: typing.Optional[str] = None,
    dtype: str = "fp32",
    batch_size: int = 4,
    shard_size: int = 2000,
    device: str = "cpu",
    cache_type: str = "lmdb",
    map_size: int = 10 * 1024**3,
    no_tqdm: bool = False,
    pbar: typing.Optional[tqdm] = None,
):
    """Builds an embedding cache for a given dataset and model."""
    torch_device = torch.device(device)
    model_path = Path(model)
    model_name_for_config: str

    if model_path.is_dir():
        print(f"INFO: Loading fine-tuned checkpoint from: {model}")
        if not base_model_for_checkpoint:
            raise ValueError("--base_model_for_checkpoint is required when --model is a directory.")
        print(f"INFO: Using base model for checkpoint: {base_model_for_checkpoint}")

        model_name_for_config = base_model_for_checkpoint
        adaptor, policy, adaptor_name = get_plm_adaptor_and_configs(base_model_for_checkpoint, for_masked_lm=True)
        loaded_model = load_peft_model_from_checkpoint(adaptor.model, str(model_path))
        adaptor.model = loaded_model
    else:
        print(f"INFO: Using base model: {model}")
        model_name_for_config = model
        adaptor, policy, adaptor_name = get_plm_adaptor_and_configs(model)

    torch.set_grad_enabled(False)

    dataset_name = dataset.split(",")[0].split("/")[-1]
    cfg = CacheConfig(
        dataset_name=dataset_name,
        model_name=model_name_for_config,
        adaptor_name=adaptor_name,
        adaptor_version="1",
        dtype=dtype,
        policy=policy,
        tokenizer_pretok=None,
        shard_size=shard_size,
        extra={},
    )
    if cache_type == "npz":
        writer = NPZCacheWriter(output_root, cfg)
    elif cache_type == "lmdb":
        writer = LMDBCacheWriter(output_root, cfg, map_size=map_size)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")

    ds_iterator = iter_pairs_from_dataset(dataset)

    adaptor.model.to(torch_device)

    processed_ids = set()
    batch_to_process = []

    # If no external pbar is provided, create a new one
    if pbar is None:
        pbar = tqdm(disable=no_tqdm)
    else:
        # If an external pbar is provided, ensure it's not disabled
        pbar.disable = no_tqdm

    def process_and_write_batch(batch):
        if not batch:
            return
        seqs = [s for _, s in batch]
        ids = [i for i, _ in batch]
        out = adaptor.encode(seqs, batch_size=len(batch), device=torch_device, fp16=dtype == "fp16")
        writer.append_batch(ids, out.residue_embeddings, out.attention_mask, out.per_sequence_lengths)
        pbar.update(len(batch))

    for ex_raw in ds_iterator:
        if not ex_raw:
            continue
        ex = typing.cast(dict, ex_raw)

        # Process seq1
        seq1_id, seq1 = ex["seq1_id"], ex["seq1"]
        if seq1_id not in processed_ids:
            batch_to_process.append((seq1_id, seq1))
            processed_ids.add(seq1_id)

        # Process seq2
        seq2_id, seq2 = ex["seq2_id"], ex["seq2"]
        if seq2_id not in processed_ids:
            batch_to_process.append((seq2_id, seq2))
            processed_ids.add(seq2_id)

        # Write batch if full
        if len(batch_to_process) >= batch_size:
            process_and_write_batch(batch_to_process)
            batch_to_process = []

    # Write the final batch
    process_and_write_batch(batch_to_process)

    pbar.close()
    writer.close()
    print("Cache built:", writer.root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--model", required=True, help="Name of the PLM to use (e.g., 'ankh-base') or path to a PEFT checkpoint directory.")
    ap.add_argument("--base_model_for_checkpoint", type=str, help="Base model name if --model is a checkpoint path.")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--cache_type", type=str, default="lmdb", choices=["npz", "lmdb"])
    ap.add_argument("--map_size", type=int, default=10 * 1024**3, help="LMDB map size in bytes. Default is 10GB.")
    ap.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    build_cache(
        dataset=args.dataset,
        model=args.model,
        output_root=args.output_root,
        base_model_for_checkpoint=args.base_model_for_checkpoint,
        dtype=args.dtype,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        device=args.device,
        cache_type=args.cache_type,
        map_size=args.map_size,
        no_tqdm=args.no_tqdm,
    )


if __name__ == "__main__":
    main()

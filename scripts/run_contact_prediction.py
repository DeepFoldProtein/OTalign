import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
from datasets import load_dataset
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint


# --- Basic Setup ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --- Constants ---
EFFECTIVE_MAX_LEN = 1022


# --- PyTorch Dataset and Model ---
# TODO: User needs to provide the path to the proteinnet parquet files
DATA_DIR = "data/proteinnet"  # Assuming this path


class ContactPredictionDataset(Dataset):
    """A PyTorch dataset for contact prediction."""

    def __init__(self, hf_dataset, contact_threshold=8.0):
        self.data = hf_dataset
        self.contact_threshold = contact_threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item["primary"]
        tertiary_coords = np.array(item["tertiary"]).reshape(-1, 3)
        mask = np.array(item["valid_mask"])

        # Calculate pairwise distance matrix
        dist_matrix = squareform(pdist(tertiary_coords))

        # Create contact map
        contact_map = (dist_matrix < self.contact_threshold).astype(np.int64)

        # Apply mask to contact map
        mask_2d = mask[:, None] & mask[None, :]
        contact_map[~mask_2d] = 0  # These values will be ignored by the loss mask

        # Use index as a unique ID if 'id' field is not present
        item_id = item.get("id", str(idx))

        return {
            "sequence": sequence,
            "labels": torch.from_numpy(contact_map),
            "id": item_id,
            "mask": torch.from_numpy(mask).bool(),
        }


class ContactPredictionModel(nn.Module):
    """A 2D CNN for contact prediction."""

    def __init__(self, input_dim, num_filters=64, kernel_size=3, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, input_dim)  # Optional projection

        layers = []

        # Initial convolution to increase channel dimension
        layers.append(nn.Conv2d(input_dim * 2, num_filters, kernel_size=1))
        layers.append(nn.ReLU())

        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())

        self.cnn_layers = nn.Sequential(*layers)

        self.output_conv = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, embeddings):
        # embeddings: (B, L, D)

        # Outer product to create a 2D representation
        L = embeddings.size(1)
        emb_i = embeddings.unsqueeze(2).expand(-1, -1, L, -1)
        emb_j = embeddings.unsqueeze(1).expand(-1, L, -1, -1)

        pair_representation = torch.cat([emb_i, emb_j], dim=3)  # (B, L, L, 2*D)
        pair_representation = pair_representation.permute(0, 3, 1, 2).contiguous()  # (B, 2*D, L, L)

        # Pass through CNN
        conv_out = self.cnn_layers(pair_representation)

        # Final prediction
        logits = self.output_conv(conv_out).squeeze(1)  # (B, L, L)

        # Symmetrize logits
        logits = (logits + logits.transpose(1, 2)) / 2

        return logits


def get_collate_fn(tokenizer):
    """Returns a collate function that uses the tokenizer's max length."""
    # The model seems to truncate to 1024 tokens, so we match that for the labels.
    # The effective sequence length is 1022 after accounting for special tokens.
    effective_max_len = EFFECTIVE_MAX_LEN

    def collate_fn(batch):
        """Pads and truncates sequences and labels in a batch."""
        sequences = [item["sequence"] for item in batch]
        labels = [item["labels"] for item in batch]
        ids = [item["id"] for item in batch]
        masks = [item["mask"] for item in batch]

        # Truncate labels and masks to the effective max length
        truncated_labels = [lab[:effective_max_len, :effective_max_len] for lab in labels]
        truncated_masks = [m[:effective_max_len] for m in masks]

        # Pad labels
        padded_labels = torch.full((len(batch), effective_max_len, effective_max_len), fill_value=0, dtype=torch.long)
        for i, lab in enumerate(truncated_labels):
            L = lab.shape[0]
            padded_labels[i, :L, :L] = lab

        # Pad masks
        padded_masks = torch.zeros((len(batch), effective_max_len), dtype=torch.bool)
        for i, m in enumerate(truncated_masks):
            L = m.shape[0]
            padded_masks[i, :L] = m

        return {"sequences": sequences, "labels": padded_labels, "ids": ids, "masks": padded_masks}

    return collate_fn


# --- Main Benchmark Logic ---


def calculate_long_range_precision(logits, labels, valid_mask, seq_sep=24):
    """Calculates top-L and top-L/5 long-range precision."""
    L_padded = valid_mask.shape[0]
    L_true = int(valid_mask.sum().item())
    if L_true == 0:
        return {"top_l_precision": 0.0, "top_l5_precision": 0.0}

    # 1. Create long-range mask based on padded length
    long_range_mask = torch.abs(torch.arange(L_padded)[:, None] - torch.arange(L_padded)[None, :]) >= seq_sep
    long_range_mask = long_range_mask.to(logits.device)

    # 2. Create final evaluation mask from the 1D valid_mask
    valid_2d_mask = valid_mask[:, None] & valid_mask[None, :]
    eval_mask = valid_2d_mask & long_range_mask

    # 3. Flatten and filter based on the evaluation mask
    # We only care about the upper triangle for unique pairs
    eval_mask = torch.triu(eval_mask, diagonal=1)
    logits_flat = logits[eval_mask]
    labels_flat = labels[eval_mask]

    if labels_flat.numel() == 0:
        return {"top_l_precision": 0.0, "top_l5_precision": 0.0}

    # 4. Sort by prediction score
    sorted_indices = torch.argsort(logits_flat, descending=True)
    sorted_labels = labels_flat[sorted_indices]

    # 5. Calculate precision for Top-L/5
    k_l5 = L_true // 5
    top_l5_precision = 0.0
    if k_l5 > 0:
        # If we have fewer valid pairs than k, use the number of valid pairs
        num_eval = min(k_l5, sorted_labels.numel())
        if num_eval > 0:
            top_l5_correct = sorted_labels[:num_eval].sum().item()
            top_l5_precision = top_l5_correct / num_eval

    # 6. Calculate precision for Top-L
    k_l1 = L_true
    top_l1_precision = 0.0
    if k_l1 > 0:
        num_eval = min(k_l1, sorted_labels.numel())
        if num_eval > 0:
            top_l1_correct = sorted_labels[:num_eval].sum().item()
            top_l1_precision = top_l1_correct / num_eval

    return {"top_l_precision": top_l1_precision, "top_l5_precision": top_l5_precision}


def evaluate_on_test_set(accelerator, cp_model, plm_adaptor, args, DATA_DIR, epoch=None):
    """Evaluates the model on the test set for a given epoch."""
    results_data = []
    test_files = [str(p) for p in Path(DATA_DIR).glob("test.parquet")]
    if not test_files:
        logging.warning("No test files found. Skipping evaluation.")
        return []

    if accelerator.is_main_process:
        log_msg = "--- Evaluating on Test Set ---"
        if epoch is not None:
            log_msg = f"--- Evaluating on Test Set for Epoch {epoch + 1} ---"
        logging.info(log_msg)

    test_hf_ds = load_dataset("parquet", data_files={"test": test_files}, split="test")
    test_dataset = ContactPredictionDataset(test_hf_ds)
    collate_fn_with_tokenizer = get_collate_fn(plm_adaptor.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_tokenizer)
    test_loader = accelerator.prepare(test_loader)

    unwrapped_cp_model = accelerator.unwrap_model(cp_model)
    unwrapped_cp_model.eval()  # Make sure model is in eval mode

    all_lr_metrics_list = []
    all_preds_flat_list = []
    all_labels_flat_list = []

    with torch.no_grad():
        desc = "Evaluating"
        if epoch is not None:
            desc = f"Evaluating Epoch {epoch + 1}"
        progress_bar = tqdm(test_loader, desc=desc, disable=not accelerator.is_main_process)
        for batch in progress_bar:
            sequences, labels, valid_mask = batch["sequences"], batch["labels"], batch["masks"]
            ids = batch["ids"]

            emb_out = plm_adaptor.encode(sequences, device=accelerator.device, disable_grad=True)
            embeddings = emb_out.residue_embeddings.to(accelerator.device)

            logits = unwrapped_cp_model(embeddings)

            L_emb = embeddings.shape[1]
            # PAD logits to ensure consistent tensor shapes across all processes for gathering.
            # The subsequent metric calculations are robust to padding via the valid_mask.
            if L_emb < EFFECTIVE_MAX_LEN:
                pad_amount = EFFECTIVE_MAX_LEN - L_emb
                logits = torch.nn.functional.pad(logits, (0, pad_amount, 0, pad_amount))

            # Do not slice labels and valid_mask, as they are already padded to a
            # consistent size by the collate function. Slicing here would create
            # tensors of varying sizes, causing a deadlock in the gather operation.

            logits_gathered = accelerator.gather(logits)
            labels_gathered = accelerator.gather(labels)
            valid_mask_gathered = accelerator.gather(valid_mask)
            ids_gathered = gather_object(ids)

            if accelerator.is_main_process:
                logits_gathered = cast(torch.Tensor, logits_gathered).cpu()
                labels_gathered = cast(torch.Tensor, labels_gathered).cpu()
                valid_mask_gathered = cast(torch.Tensor, valid_mask_gathered).cpu()

                for i in range(len(ids_gathered)):
                    logit = logits_gathered[i]
                    label = labels_gathered[i]
                    v_mask = valid_mask_gathered[i]
                    protein_id = ids_gathered[i]

                    # --- Standard Accuracy Calculation ---
                    pred = (torch.sigmoid(logit) > 0.5).long()
                    mask = v_mask[:, None] & v_mask[None, :]
                    pred_flat = pred[mask]
                    label_flat = label[mask]

                    accuracy = 0.0
                    if label_flat.numel() > 0:
                        accuracy = (pred_flat == label_flat).float().mean().item()
                        all_preds_flat_list.append(pred_flat)
                        all_labels_flat_list.append(label_flat)

                    # --- Long-Range Precision Calculation ---
                    lr_metrics = calculate_long_range_precision(logit, label, v_mask)
                    all_lr_metrics_list.append(lr_metrics)

                    # --- Combine and Store Metrics ---
                    metrics = {"accuracy": accuracy, **lr_metrics}
                    meta = {"model": args.model}
                    if epoch is not None:
                        meta["epoch"] = epoch + 1
                    results_data.append({"id": protein_id, "metrics": metrics, "meta": meta})

    if accelerator.is_main_process:
        if all_labels_flat_list:
            all_preds_flat = torch.cat(all_preds_flat_list)
            all_labels_flat = torch.cat(all_labels_flat_list)
            overall_accuracy = (all_preds_flat == all_labels_flat).float().mean().item()
            log_prefix = ""
            if epoch is not None:
                log_prefix = f"Epoch {epoch + 1} "
            logging.info(f"{log_prefix}Overall Test Accuracy: {overall_accuracy:.4f}")

            # Calculate and log mean long-range precision
            avg_top_l_precision = np.mean([m["top_l_precision"] for m in all_lr_metrics_list])
            avg_top_l5_precision = np.mean([m["top_l5_precision"] for m in all_lr_metrics_list])
            logging.info(f"{log_prefix}Overall Top-L Long-Range Precision: {avg_top_l_precision:.4f}")
            logging.info(f"{log_prefix}Overall Top-L/5 Long-Range Precision: {avg_top_l5_precision:.4f}")
        else:
            logging.info("No valid labels found to calculate overall accuracy.")

    return results_data


def run_contact_prediction_benchmark(args):
    """Main function to run the Contact Prediction benchmark."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    if accelerator.is_main_process:
        logging.info(f"Using device: {device}")

    # 1. Load Model and Adaptor
    if accelerator.is_main_process:
        logging.info(f"Loading model: {args.model}")
    model_path = Path(args.model)
    base_model_name = args.base_model_for_checkpoint if model_path.is_dir() else args.model

    plm_adaptor, _, _ = get_plm_adaptor_and_configs(base_model_name, for_masked_lm=True)
    hidden_size = plm_adaptor.model.config.hidden_size

    if model_path.is_dir():
        if accelerator.is_main_process:
            logging.info(f"Loading LoRA checkpoint from {model_path} with base model {base_model_name}")
        plm_adaptor.model = load_peft_model_from_checkpoint(plm_adaptor.model, str(model_path))

    model = plm_adaptor.model
    model.eval()

    # 3. Train or Load Contact Prediction Model
    cp_model = ContactPredictionModel(input_dim=hidden_size)
    plm_adaptor.model.to(device)

    results_data = []

    if args.eval_only:
        if accelerator.is_main_process:
            logging.info(f"Evaluation only. Loading pretrained Contact Prediction model from {args.cp_model_path}")
        cp_model.load_state_dict(torch.load(args.cp_model_path, map_location=device))
        cp_model = accelerator.prepare(cp_model)

        eval_results = evaluate_on_test_set(accelerator, cp_model, plm_adaptor, args, DATA_DIR)
        results_data.extend(eval_results)
    else:
        # 2. Load Dataset
        if accelerator.is_main_process:
            logging.info(f"Loading dataset from: {DATA_DIR}")

        # Load training data
        train_files = [str(p) for p in Path(DATA_DIR).glob("train*.parquet")]
        train_hf_ds = load_dataset("parquet", data_files={"train": train_files}, split="train")
        train_dataset = ContactPredictionDataset(train_hf_ds)
        collate_fn_with_tokenizer = get_collate_fn(plm_adaptor.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_tokenizer)

        # Load validation data
        valid_files = [str(p) for p in Path(DATA_DIR).glob("valid*.parquet")]
        valid_loader = None
        if valid_files:
            if accelerator.is_main_process:
                logging.info("Loading validation dataset.")
            valid_hf_ds = load_dataset("parquet", data_files={"valid": valid_files}, split="valid")
            valid_dataset = ContactPredictionDataset(valid_hf_ds)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_tokenizer)

        # 3. Train Contact Prediction Model
        if accelerator.is_main_process:
            logging.info("Training model for Contact Prediction...")
        optimizer = torch.optim.Adam(cp_model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        cp_model, optimizer, train_loader = accelerator.prepare(cp_model, optimizer, train_loader)
        if valid_loader:
            valid_loader = accelerator.prepare(valid_loader)

        best_valid_loss = float("inf")

        for epoch in range(args.epochs):
            cp_model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} (Train)", disable=not accelerator.is_main_process)
            for batch in progress_bar:
                with accelerator.accumulate(cp_model):
                    sequences, labels, valid_mask = batch["sequences"], batch["labels"], batch["masks"]

                    with torch.no_grad():
                        emb_out = plm_adaptor.encode(sequences, device=device, disable_grad=True)
                        embeddings = emb_out.residue_embeddings.to(device)

                    logits = cp_model(embeddings)

                    L = embeddings.shape[1]
                    labels = labels[:, :L, :L]
                    valid_mask = valid_mask[:, :L]

                    mask = (valid_mask[:, None, :] & valid_mask[:, :, None]).float()
                    loss = criterion(logits, labels.float())
                    loss = (loss * mask).sum() / mask.sum()

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if accelerator.is_main_process:
                logging.info(f"Epoch {epoch + 1} finished. Average Training Loss: {avg_loss:.4f}")

            # --- Validation Step ---
            if valid_loader:
                cp_model.eval()
                total_valid_loss = 0
                with torch.no_grad():
                    progress_bar_valid = tqdm(valid_loader, desc=f"Epoch {epoch + 1} (Valid)", disable=not accelerator.is_main_process)
                    for batch in progress_bar_valid:
                        sequences, labels, valid_mask = batch["sequences"], batch["labels"], batch["masks"]

                        emb_out = plm_adaptor.encode(sequences, device=device, disable_grad=True)
                        embeddings = emb_out.residue_embeddings.to(device)

                        logits = cp_model(embeddings)

                        L = embeddings.shape[1]
                        labels = labels[:, :L, :L]
                        valid_mask = valid_mask[:, :L]

                        mask = (valid_mask[:, None, :] & valid_mask[:, :, None]).float()
                        loss = criterion(logits, labels.float())
                        loss = (loss * mask).sum() / mask.sum()
                        total_valid_loss += loss.item()

                avg_valid_loss = total_valid_loss / len(valid_loader)
                if accelerator.is_main_process:
                    logging.info(f"Epoch {epoch + 1} Average Validation Loss: {avg_valid_loss:.4f}")
                    if avg_valid_loss < best_valid_loss:
                        best_valid_loss = avg_valid_loss
                        logging.info(f"New best validation loss. Saving model to {outdir / 'cp_model.pt'}")
                        unwrapped_model_to_save = accelerator.unwrap_model(cp_model)
                        torch.save(unwrapped_model_to_save.state_dict(), outdir / "cp_model.pt")

            # --- Evaluation on test set after each epoch ---
            eval_results = evaluate_on_test_set(accelerator, cp_model, plm_adaptor, args, DATA_DIR, epoch=epoch)
            results_data.extend(eval_results)

        # After training, if no validation set was used, save the final model
        if not valid_loader:
            unwrapped_cp_model = accelerator.unwrap_model(cp_model)
            if accelerator.is_main_process:
                logging.info(f"No validation set. Saving final model to {outdir / 'cp_model.pt'}")
                torch.save(unwrapped_cp_model.state_dict(), outdir / "cp_model.pt")

    # 5. Save Results
    if accelerator.is_main_process:
        logging.info(f"Saving results to {str(outdir)}")
        with open(outdir / "results.jsonl", "w") as f:
            for item in results_data:
                f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Contact Prediction Benchmark.")
    parser.add_argument("--model", required=True, help="Name of the PLM or path to a PEFT checkpoint.")
    parser.add_argument("--base_model_for_checkpoint", type=str, help="Base model name if --model is a checkpoint.")
    parser.add_argument("--outdir", required=True, help="Directory to outputs.")
    parser.add_argument("--data_dir", type=str, default="data/proteinnet", help="Directory containing proteinnet parquet files.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the model.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--eval_only", action="store_true", help="If set, skip training and only run evaluation.")
    parser.add_argument("--cp_model_path", type=str, help="Path to a pretrained cp_model.pt file. Required if --eval_only is set.")
    args = parser.parse_args()

    if args.eval_only and not args.cp_model_path:
        parser.error("--cp_model_path is required when --eval_only is set.")

    global DATA_DIR
    DATA_DIR = args.data_dir

    run_contact_prediction_benchmark(args)


if __name__ == "__main__":
    main()

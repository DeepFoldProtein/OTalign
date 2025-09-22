import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import accelerate
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint


# --- Basic Setup ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- 3-state mapping for decapoda-research/protein_secondary_structure ---
ss3_map = {
    "H": 0,  # Helix
    "E": 1,  # Strand
    "C": 2,  # Coil
    " ": -100,  # Padding character in this dataset
}


ss8_map = {
    "H": 0,  # Alpha Helix
    "B": 1,  # Bridge
    "E": 2,  # Strand
    "G": 3,  # 3_10 Helix
    "I": 4,  # Pi Helix
    "T": 5,  # Turn
    "S": 6,  # Bend
    "C": 7,  # Coil
    " ": -100,  # Padding
}


# --- PyTorch Dataset and Model ---
DATASET_ID = "proteinea/secondary_structure_prediction"


class SSPDataset(Dataset):
    """A PyTorch dataset for secondary structure prediction."""

    def __init__(self, hf_dataset, kind: Literal["ssp3", "ssp8"] = "ssp8"):
        self.data = hf_dataset
        assert kind in ("ssp3", "ssp8")
        self.kind = kind

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item["input"]
        ssp_str_labels = item["dssp3"] if self.kind == "ssp3" else item["dssp8"]
        # Convert to numeric labels, ignoring padding
        ss_map = ss3_map if self.kind == "ssp3" else ss8_map
        ss_labels = torch.tensor([ss_map[s] for s in ssp_str_labels if s in ss_map], dtype=torch.long)
        # Use index as a unique ID if 'id' field is not present
        item_id = item.get("id", idx)
        return {"sequence": sequence, "labels": ss_labels, "id": item_id}


class SSPModel(nn.Module):
    """A simple linear head for SSP classification."""

    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        return self.classifier(embeddings)


def get_collate_fn(tokenizer):
    """Returns a collate function that uses the tokenizer's max length."""
    max_len = getattr(tokenizer, "model_max_length", 1024)
    if max_len is None:
        max_len = 1024
    # The model seems to truncate to 1024 tokens, so we match that for the labels.
    # The effective sequence length is 1022 after accounting for special tokens.
    effective_max_len = 1022

    def collate_fn(batch):
        """Pads and truncates sequences and labels in a batch."""
        sequences = [item["sequence"] for item in batch]
        labels = [item["labels"] for item in batch]
        ids = [item["id"] for item in batch]

        # Truncate labels to the effective max length
        truncated_labels = [lab[:effective_max_len] for lab in labels]

        # Pad labels
        padded_labels = torch.full((len(batch), effective_max_len), fill_value=-100, dtype=torch.long)
        for i, lab in enumerate(truncated_labels):
            padded_labels[i, : len(lab)] = lab

        return {"sequences": sequences, "labels": padded_labels, "ids": ids}

    return collate_fn


# --- Main Benchmark Logic ---


def evaluate_on_test_sets(accelerator, ssp_model, plm_adaptor, args, ssp_kind: Literal["ssp3", "ssp8"] = "ssp8", epoch=None):
    """Evaluates the model on all test sets for a given epoch."""
    results_data = []
    test_data_labels = ["CASP12", "CASP13", "CASP14", "TS115", "CB513"]

    unwrapped_ssp_model = accelerator.unwrap_model(ssp_model)
    unwrapped_ssp_model.eval()

    for test_label in test_data_labels:
        if accelerator.is_main_process:
            log_msg = f"--- Evaluating on {test_label} ---"
            if epoch is not None:
                log_msg = f"--- Evaluating on {test_label} for Epoch {epoch + 1} ---"
            logging.info(log_msg)

        test_data_files = {"test": f"{test_label}.csv"}
        test_hf_ds = load_dataset(DATASET_ID, data_files=test_data_files, split="test")
        test_dataset = SSPDataset(test_hf_ds, kind=ssp_kind)
        collate_fn_with_tokenizer = get_collate_fn(plm_adaptor.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_tokenizer)
        test_loader = accelerator.prepare(test_loader)

        all_preds_gathered_batches, all_labels_gathered_batches, all_ids_gathered_batches = [], [], []
        with torch.no_grad():
            desc = f"Evaluating {test_label}"
            if epoch is not None:
                desc = f"Evaluating {test_label} (Epoch {epoch + 1})"
            progress_bar = tqdm(test_loader, desc=desc, disable=not accelerator.is_main_process)
            for batch in progress_bar:
                sequences, labels = batch["sequences"], batch["labels"]
                ids = batch["ids"]

                emb_out = plm_adaptor.encode(sequences, device=accelerator.device, disable_grad=True)
                embeddings = emb_out.residue_embeddings.to(accelerator.device)

                logits = unwrapped_ssp_model(embeddings)  # (B, L, C)
                predictions = torch.argmax(logits, dim=-1)

                # Pad predictions to match label length before gathering
                if predictions.shape[1] < labels.shape[1]:
                    pad_size = labels.shape[1] - predictions.shape[1]
                    # Use -100 for padding, consistent with label padding
                    padding = torch.full((predictions.shape[0], pad_size), -100, device=predictions.device, dtype=predictions.dtype)
                    predictions = torch.cat([predictions, padding], dim=1)

                # Gather predictions and labels from all processes
                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)
                ids_gathered = accelerate.utils.gather_object(ids)

                all_preds_gathered_batches.append(predictions_gathered)
                all_labels_gathered_batches.append(labels_gathered)
                all_ids_gathered_batches.extend(ids_gathered)

        if accelerator.is_main_process:
            all_preds_cat = torch.cat(all_preds_gathered_batches, dim=0)
            all_labels_cat = torch.cat(all_labels_gathered_batches, dim=0)
            all_ids = all_ids_gathered_batches

            all_preds_unpadded, all_labels_unpadded = [], []

            for i in range(len(all_ids)):
                seq_len = (all_labels_cat[i] != -100).sum().item()
                pred_unpadded = all_preds_cat[i, :seq_len].cpu().numpy()
                label_unpadded = all_labels_cat[i, :seq_len].cpu().numpy()

                accuracy = np.mean(pred_unpadded == label_unpadded)

                meta = {"model": args.model, "test_set": test_label}
                if epoch is not None:
                    meta["epoch"] = epoch + 1

                results_data.append(
                    {
                        "pair_id": all_ids[i],
                        "seq1_id": all_ids[i],
                        "seq2_id": "n/a",
                        "metrics": {"accuracy": accuracy},
                        "meta": meta,
                    }
                )
                all_preds_unpadded.extend(pred_unpadded)
                all_labels_unpadded.extend(label_unpadded)

            overall_accuracy = np.mean(np.array(all_preds_unpadded) == np.array(all_labels_unpadded))
            log_prefix = ""
            if epoch is not None:
                log_prefix = f"Epoch {epoch + 1} "
            logging.info(f"{log_prefix}Overall Test Accuracy for {test_label}: {overall_accuracy:.4f}")

    return results_data


def run_ssp_benchmark(args):
    """Main function to run the SSP benchmark."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
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

    # Determine number of classes based on ssp_kind
    ssp_kind = args.ssp_kind
    num_classes = 8 if ssp_kind == "ssp8" else 3
    if accelerator.is_main_process:
        logging.info(f"Running benchmark for {ssp_kind} with {num_classes} classes.")

    # 3. Train or Load Linear Probe
    ssp_model = SSPModel(input_dim=hidden_size, num_classes=num_classes)
    plm_adaptor.model.to(device)

    results_data = []

    if args.eval_only:
        if accelerator.is_main_process:
            logging.info(f"Evaluation only. Loading pretrained SSP model from {args.ssp_model_path}")
        ssp_model.load_state_dict(torch.load(args.ssp_model_path, map_location=device))
        ssp_model = accelerator.prepare(ssp_model)

        eval_results = evaluate_on_test_sets(accelerator, ssp_model, plm_adaptor, args, ssp_kind=ssp_kind)
        results_data.extend(eval_results)
    else:
        # 2. Load Dataset
        if accelerator.is_main_process:
            logging.info(f"Loading dataset: {DATASET_ID}")

        # Load training data
        train_data_files = {"train": "training_hhblits.csv"}
        train_hf_ds = load_dataset(DATASET_ID, data_files=train_data_files, split="train")
        train_dataset = SSPDataset(train_hf_ds, kind=ssp_kind)
        collate_fn_with_tokenizer = get_collate_fn(plm_adaptor.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_tokenizer)

        # 3. Train Linear Probe
        if accelerator.is_main_process:
            logging.info("Training linear probe for SSP...")
        optimizer = torch.optim.Adam(ssp_model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padded labels

        ssp_model, optimizer, train_loader = accelerator.prepare(ssp_model, optimizer, train_loader)

        # Enable gradients for training the probe
        with torch.set_grad_enabled(True):
            for epoch in range(args.epochs):
                ssp_model.train()
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=not accelerator.is_main_process)
                for batch in progress_bar:
                    with accelerator.accumulate(ssp_model):
                        sequences, labels = batch["sequences"], batch["labels"]

                        with torch.no_grad():
                            emb_out = plm_adaptor.encode(sequences, device=device, disable_grad=True)
                            embeddings = emb_out.residue_embeddings.to(device)

                        # Get logits (B, L, C)
                        logits = ssp_model(embeddings)

                        # Permute logits to (B, C, L) for CrossEntropyLoss
                        logits_permuted = logits.permute(0, 2, 1)

                        # Labels are (B, L), but might be longer than embeddings due to collate_fn padding.
                        # Truncate labels to match embedding length.
                        labels = labels[:, : embeddings.shape[1]]

                        loss = criterion(logits_permuted, labels)
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                if accelerator.is_main_process:
                    logging.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

                # --- Evaluation on test set after each epoch ---
                eval_results = evaluate_on_test_sets(accelerator, ssp_model, plm_adaptor, args, ssp_kind=ssp_kind, epoch=epoch)
                results_data.extend(eval_results)

        # Save the final model
        unwrapped_ssp_model = accelerator.unwrap_model(ssp_model)
        if accelerator.is_main_process:
            torch.save(unwrapped_ssp_model.state_dict(), outdir / "ssp_model.pt")

    # 5. Save Results
    if accelerator.is_main_process:
        logging.info(f"Saving results to {str(outdir)}")
        with open(outdir / "results.jsonl", "w") as f:
            for item in results_data:
                f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Secondary Structure Prediction Benchmark.")
    parser.add_argument("--model", required=True, help="Name of the PLM or path to a PEFT checkpoint.")
    parser.add_argument("--base_model_for_checkpoint", type=str, help="Base model name if --model is a checkpoint.")
    parser.add_argument("--outdir", required=True, help="Directory to outputs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the linear probe.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the probe.")
    parser.add_argument("--ssp_kind", type=str, default="ssp8", choices=["ssp3", "ssp8"], help="Type of SSP classification (ssp3 or ssp8).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--eval_only", action="store_true", help="If set, skip training and only run evaluation.")
    parser.add_argument("--ssp_model_path", type=str, help="Path to a pretrained ssp_model.pt file. Required if --eval_only is set.")
    args = parser.parse_args()

    if args.eval_only and not args.ssp_model_path:
        parser.error("--ssp_model_path is required when --eval_only is set.")

    run_ssp_benchmark(args)


if __name__ == "__main__":
    main()

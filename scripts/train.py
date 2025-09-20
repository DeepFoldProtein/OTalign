import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from otalign.align.cost import pairwise_cosine
from otalign.data.cath import CATHDataset
from otalign.data.collator import OTAlignCollator
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn
from otalign.metrics.alignment import in_band_mass
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint
from otalign.utils.ddp import get_rank, get_world_size, init_distributed_mode, is_main_process


def setup_logging(log_dir: Path, run_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = [logging.FileHandler(log_dir / f"{run_name}.log")]
    if is_main_process():
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def generalized_kl_divergence(q, p):
    eps = 1e-8
    return torch.sum(q * (torch.log(q + eps) - torch.log(p + eps)) - q + p)


def save_checkpoint(model, optimizer, epoch, config, wandb_run_id, is_best=False):
    # Create a subdirectory for the current run
    chkpt_dir = Path(config["checkpoint_dir"]) / wandb_run_id
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    # PEFT models save the adapter weights, not the full model
    model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    # The path for PEFT's save_pretrained method is a directory
    chkpt_path = chkpt_dir / f"epoch_{epoch + 1}"
    model_to_save.save_pretrained(str(chkpt_path))

    # Save optimizer state separately
    optimizer_path = chkpt_dir / f"optimizer_epoch_{epoch + 1}.pt"
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        optimizer_path,
    )

    logging.info(f"Saved checkpoint to {chkpt_path} and optimizer state to {optimizer_path}")


def train(config_path: str, eval_before_train: bool = False, eval_only: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Setup ---
    init_distributed_mode()
    device = torch.device(config["device"])
    setup_logging(Path(config["log_dir"]), config["run_name"])

    wandb_run_id = "unknown_run"
    if is_main_process():
        run = wandb.init(project=config["project_name"], name=config["run_name"], config=config)
        if run:
            wandb_run_id = run.id
            logging.info(f"Starting run '{config['run_name']}' (ID: {wandb_run_id}) with config:\n{yaml.dump(config)}")
        else:
            logging.warning("wandb.init() failed. Checkpoints will be saved in a directory named 'unknown_run'.")

    # --- 1. DATASET ---
    logging.info("Loading datasets...")
    train_dataset = CATHDataset(data_root=config["data_root"], split="train")
    val_dataset = CATHDataset(data_root=config["data_root"], split="validation")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if get_world_size() > 1 else None
    collator = OTAlignCollator()

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collator, num_workers=config["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=collator, num_workers=config["num_workers"])

    # --- 2. MODEL ---
    logging.info(f"Loading PLM adaptor for '{config['model_name']}'...")
    plm_adaptor, _, _ = get_plm_adaptor_and_configs(config["model_name"])
    model = plm_adaptor.model.to(device)

    # --- DEBUG: Print model modules ---
    # for name, module in model.named_modules():
    #     print(name)
    # import sys; sys.exit()
    # --- END DEBUG ---

    for param in model.parameters():
        param.requires_grad = False

    if config.get("resume_from_checkpoint"):
        lora_model = load_peft_model_from_checkpoint(model, config["resume_from_checkpoint"])
    else:
        logging.info("Applying new LoRA with PEFT...")

        lora_params = config["lora"].copy()
        if "dropout" in lora_params:
            lora_params["lora_dropout"] = lora_params.pop("dropout")

        lora_config = LoraConfig(**lora_params)
        lora_model = get_peft_model(model, lora_config)

    if is_main_process():
        lora_model.print_trainable_parameters()

    if get_world_size() > 1:
        lora_model = nn.parallel.DistributedDataParallel(lora_model, device_ids=[get_rank()])

    # --- Initial Evaluation ---
    if eval_before_train or eval_only:
        if is_main_process():
            logging.info(f"Running evaluation{' only' if eval_only else ' before training'}...")
            model_to_eval = lora_model.module if get_world_size() > 1 else lora_model
            evaluate(model_to_eval, plm_adaptor, val_loader, device, config, epoch=-1)

    if eval_only:
        if is_main_process():
            logging.info("Evaluation finished. Exiting.")
            if wandb.run:
                wandb.finish()
        return

    # --- 3. TRAINING ---
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=config["lr"])
    start_epoch = 0

    if config.get("resume_from_checkpoint"):
        # PEFT loads model adapters, now load optimizer state
        optimizer_path = Path(config["resume_from_checkpoint"]).parent / f"optimizer_{Path(config['resume_from_checkpoint']).name}.pt"
        if os.path.exists(optimizer_path):
            chkpt = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(chkpt["optimizer_state_dict"])
            start_epoch = chkpt["epoch"] + 1
            logging.info(f"Resumed optimizer state from {optimizer_path}. Starting at epoch {start_epoch}.")

    logging.info("Starting training...")
    for epoch in range(start_epoch, config["epochs"]):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        lora_model.train()

        # Add tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", disable=not is_main_process())
        for i, batch in enumerate(train_pbar):
            if is_main_process() and wandb.run and getattr(wandb.run, "state", "running") != "running":
                logging.warning("wandb run stopped from UI. Stopping training.")
                break
            seqs1, seqs2, gt_alignments, is_positive, lens1, lens2 = (
                batch["seqs1"],
                batch["seqs2"],
                batch["gt_alignments"],
                batch["is_positive"],
                batch["lens1"],
                batch["lens2"],
            )

            emb1_out = plm_adaptor.encode(seqs1, device=device, disable_grad=False)
            emb2_out = plm_adaptor.encode(seqs2, device=device, disable_grad=False)
            emb1 = emb1_out.residue_embeddings
            emb2 = emb2_out.residue_embeddings

            if not torch.isfinite(emb1).all() or not torch.isfinite(emb2).all():
                logging.warning("Embeddings contain non-finite values. Skipping batch.")
                continue

            cost_matrix = pairwise_cosine(emb1, emb2)

            B, N, M = cost_matrix.shape
            lens1, lens2 = lens1.to(device), lens2.to(device)
            mask1 = torch.arange(N, device=device)[None, :] < lens1[:, None]
            mask2 = torch.arange(M, device=device)[None, :] < lens2[:, None]

            a = mask1.float() / lens1[:, None].clamp(min=1).float()
            b = mask2.float() / lens2[:, None].clamp(min=1).float()

            # Add a large cost to padded areas to ignore them
            cost_matrix[~(mask1[:, :, None] * mask2[:, None, :])] = 1e6

            transport_plan, _, _ = unbalanced_sinkhorn(cost_matrix, a, b, config["uot"]["num_iter"], config["uot"]["reg"], config["uot"]["reg_m"], config["uot"]["reg_m"], mask_a=mask1, mask_b=mask2)

            pos_mask = is_positive.to(device)
            neg_mask = ~pos_mask
            loss = torch.tensor(0.0, device=device)

            if pos_mask.any():
                gt_alignments = gt_alignments.to(device)
                gt_align = gt_alignments[pos_mask]
                pred_plan_pos = transport_plan[pos_mask]
                l_alignment = generalized_kl_divergence(gt_align, pred_plan_pos)
                l_sparsity = torch.sum(torch.abs(pred_plan_pos), dim=[1, 2]).mean()
                loss += l_alignment + config["loss"]["lambda_pos"] * l_sparsity

            if neg_mask.any():
                pred_plan_neg = transport_plan[neg_mask]
                l_emptiness = torch.sum(torch.abs(pred_plan_neg), dim=[1, 2]).mean()
                loss += config["loss"]["lambda_neg"] * l_emptiness

            optimizer.zero_grad()
            loss = loss / config.get("gradient_accumulation_steps", 1)
            loss.backward()

            if (i + 1) % config.get("gradient_accumulation_steps", 1) == 0:
                optimizer.step()
                optimizer.zero_grad()

            if is_main_process():
                train_pbar.set_postfix(loss=loss.item())
                if i % config["log_interval"] == 0:
                    wandb.log({"train/loss": loss.item()})

        if is_main_process() and wandb.run and getattr(wandb.run, "state", "running") != "running":
            break

        # --- 4. EVALUATION & CHECKPOINTING ---
        if is_main_process():
            logging.info(f"Epoch {epoch + 1} finished. Running evaluation...")
            model_to_eval = lora_model.module if get_world_size() > 1 else lora_model
            if not evaluate(model_to_eval, plm_adaptor, val_loader, device, config, epoch):
                logging.warning("wandb run stopped during evaluation. Finishing training.")
                break

            if (epoch + 1) % config["save_checkpoint_freq"] == 0:
                save_checkpoint(lora_model, optimizer, epoch, config, wandb_run_id)

    if is_main_process() and wandb.run:
        wandb.finish()


def evaluate(model, plm_adaptor, data_loader, device, config, epoch) -> bool:
    """Returns False if training should stop."""
    model.eval()

    total_val_loss = 0.0
    in_band_masses = []
    band_width = config.get("eval_band_width", 5)

    with torch.no_grad():
        # Add tqdm for progress bar
        desc = "Initial Evaluation" if epoch == -1 else f"Evaluating Epoch {epoch + 1}"
        eval_pbar = tqdm(data_loader, desc=desc, disable=not is_main_process())
        for batch in eval_pbar:
            if is_main_process() and wandb.run and getattr(wandb.run, "state", "running") != "running":
                return False

            seqs1, seqs2, gt_alignments, is_positive, lens1, lens2 = (
                batch["seqs1"],
                batch["seqs2"],
                batch["gt_alignments"],
                batch["is_positive"],
                batch["lens1"],
                batch["lens2"],
            )

            emb1 = plm_adaptor.encode(seqs1, device=device).residue_embeddings
            emb2 = plm_adaptor.encode(seqs2, device=device).residue_embeddings

            original_cost_matrix = pairwise_cosine(emb1, emb2)
            cost_matrix = original_cost_matrix.clone()

            B, N, M = cost_matrix.shape
            lens1, lens2 = lens1.to(device), lens2.to(device)
            mask1 = torch.arange(N, device=device)[None, :] < lens1[:, None]
            mask2 = torch.arange(M, device=device)[None, :] < lens2[:, None]

            a = mask1.float() / lens1[:, None].clamp(min=1).float()
            b = mask2.float() / lens2[:, None].clamp(min=1).float()

            # Add a large cost to padded areas to ignore them
            cost_matrix[~(mask1[:, :, None] * mask2[:, None, :])] = 1e6

            transport_plan, _, _ = unbalanced_sinkhorn(cost_matrix, a, b, config["uot"]["num_iter"], config["uot"]["reg"], config["uot"]["reg_m"], config["uot"]["reg_m"], mask_a=mask1, mask_b=mask2)

            # Calculate loss
            pos_mask_loss = is_positive.to(device)
            neg_mask_loss = ~pos_mask_loss
            loss = torch.tensor(0.0, device=device)

            if pos_mask_loss.any():
                gt_alignments_loss = gt_alignments.to(device)
                gt_align_loss = gt_alignments_loss[pos_mask_loss]
                pred_plan_pos_loss = transport_plan[pos_mask_loss]
                l_alignment = generalized_kl_divergence(gt_align_loss, pred_plan_pos_loss)
                l_sparsity = torch.sum(torch.abs(pred_plan_pos_loss), dim=[1, 2]).mean()
                loss += l_alignment + config["loss"]["lambda_pos"] * l_sparsity

            if neg_mask_loss.any():
                pred_plan_neg_loss = transport_plan[neg_mask_loss]
                l_emptiness = torch.sum(torch.abs(pred_plan_neg_loss), dim=[1, 2]).mean()
                loss += config["loss"]["lambda_neg"] * l_emptiness
            total_val_loss += loss.item()

            pos_mask = is_positive.cpu()

            if pos_mask.any():
                gt_align = gt_alignments[pos_mask].to(device)
                pred_plan_pos = transport_plan[pos_mask]

                # Normalize pred_plan_pos for in_band_mass
                pred_plan_sum = torch.sum(pred_plan_pos, dim=(1, 2), keepdim=True).clamp(min=1e-8)
                normalized_pred_plan = pred_plan_pos / pred_plan_sum

                for j in range(gt_align.size(0)):
                    mass = in_band_mass(normalized_pred_plan[j], gt_align[j], band_width=band_width)
                    in_band_masses.append(mass)

    avg_val_loss = total_val_loss / len(data_loader)
    avg_in_band_mass = np.mean(in_band_masses) if in_band_masses else 0

    log_prefix = "Initial Evaluation" if epoch == -1 else f"Epoch {epoch + 1} Validation"
    logging.info(f"--- {log_prefix} Results ---")
    logging.info(f"Average Loss: {avg_val_loss:.4f}")
    logging.info(f"Average In-band Mass (w={band_width}): {avg_in_band_mass:.4f}")
    logging.info("---------------------------------")

    if is_main_process() and wandb.run:
        wandb_log = {
            "val/loss": avg_val_loss,
            f"val/in_band_mass_w_{band_width}": avg_in_band_mass,
        }
        if epoch != -1:
            wandb_log["epoch"] = epoch + 1
        wandb.log(wandb_log)

    return True


def main():
    parser = argparse.ArgumentParser(description="Train OTAlign model with LoRA.")
    parser.add_argument("config_path", help="Path to the configuration YAML file.")
    parser.add_argument("--eval_before_train", action="store_true", help="Run an initial evaluation before starting training.")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only on the validation set and exit.")
    args = parser.parse_args()

    train(config_path=args.config_path, eval_before_train=args.eval_before_train, eval_only=args.eval_only)


if __name__ == "__main__":
    main()

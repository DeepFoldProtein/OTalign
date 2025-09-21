import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb
from otalign.align.cost import pairwise_cosine
from otalign.data.cath import CATHDataset
from otalign.data.collator import OTAlignCollator
from otalign.data.mlm_collator import MLMCollator
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn
from otalign.metrics.alignment import vectorized_in_band_mass, vectorized_recall_in_band
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint


def setup_logging(log_dir: Path, run_name: str, accelerator: Accelerator):
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = [logging.FileHandler(log_dir / f"{run_name}.log")]
    if accelerator.is_main_process:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def generalized_kl_divergence(q, p):
    eps = 1e-8
    # Return per-sample loss by summing over the matrix dimensions (1, 2)
    return torch.sum(q * (torch.log(q + eps) - torch.log(p + eps)) - q + p, dim=(1, 2))


class WandbStopTrainingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.accelerator: Accelerator | None = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.accelerator and self.accelerator.is_main_process and wandb.run and getattr(wandb.run, "state", "running") != "running":
            logging.warning("wandb run stopped from UI. Stopping training.")
            control.should_training_stop = True
        return control


class CustomEvalAndSaveCallback(TrainerCallback):
    """A custom callback that evaluates and saves the model at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.trainer: Trainer | None = None
        self.accelerator: Accelerator | None = None

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer is None:
            logging.warning("Trainer not injected into CustomEvalAndSaveCallback. Skipping eval and save.")
            return

        if state.epoch is None or args.output_dir is None:
            logging.warning(f"Cannot evaluate/save at epoch end. Epoch: {state.epoch}, Output Dir: {args.output_dir}")
            return

        if self.accelerator and self.accelerator.is_main_process:
            epoch = int(state.epoch)
            logging.info(f"Epoch {epoch} finished. Running evaluation...")
            metrics = self.trainer.evaluate()
            logging.info(f"Evaluation metrics for epoch {epoch}: {metrics}")

            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            self.trainer.save_model(output_dir)
            logging.info(f"Saved model checkpoint to {output_dir}")


class OTAlignTrainer(Trainer):
    def __init__(self, *args, plm_adaptor, custom_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.plm_adaptor = plm_adaptor
        self.config = custom_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.plm_adaptor.model = model
        device = self.accelerator.device

        seqs1, seqs2, gt_alignments, is_positive, lens1, lens2 = (
            inputs["seqs1"],
            inputs["seqs2"],
            inputs["gt_alignments"],
            inputs["is_positive"],
            inputs["lens1"],
            inputs["lens2"],
        )
        mlm_input_ids1, mlm_labels1, mlm_attention_mask1 = (inputs["mlm_input_ids1"], inputs["mlm_labels1"], inputs["mlm_attention_mask1"])
        mlm_input_ids2, mlm_labels2, mlm_attention_mask2 = (inputs["mlm_input_ids2"], inputs["mlm_labels2"], inputs["mlm_attention_mask2"])

        disable_grad = not self.is_in_train
        emb1_out = self.plm_adaptor.encode(seqs1, device=device, disable_grad=disable_grad)
        emb2_out = self.plm_adaptor.encode(seqs2, device=device, disable_grad=disable_grad)
        emb1 = emb1_out.residue_embeddings
        emb2 = emb2_out.residue_embeddings

        if not torch.isfinite(emb1).all() or not torch.isfinite(emb2).all():
            logging.warning("Embeddings contain non-finite values. Skipping batch.")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_outputs:
                # Return empty tensors that can be collated but identified in compute_metrics
                return (loss, (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)))
            return loss

        cost_matrix = pairwise_cosine(emb1, emb2)
        B, N, M = cost_matrix.shape
        lens1, lens2 = lens1.to(device), lens2.to(device)
        mask1 = torch.arange(N, device=device)[None, :] < lens1[:, None]
        mask2 = torch.arange(M, device=device)[None, :] < lens2[:, None]
        a = mask1.float() / lens1[:, None].clamp(min=1).float()
        b = mask2.float() / lens2[:, None].clamp(min=1).float()
        cost_matrix[~(mask1[:, :, None] * mask2[:, None, :])] = 1e6

        transport_plan, _, _ = unbalanced_sinkhorn(
            cost_matrix, a, b, self.config["uot"]["num_iter"], self.config["uot"]["reg"], self.config["uot"]["reg_m"], self.config["uot"]["reg_m"], mask_a=mask1, mask_b=mask2
        )

        pos_mask = is_positive.to(device)
        loss = torch.tensor(0.0, device=device)
        mlm_loss_val = torch.tensor(0.0, device=device)
        log_payload = {}

        if "lambda_mlm" in self.config["loss"] and self.config["loss"]["lambda_mlm"] > 0 and self.is_in_train:
            mlm_out1 = model(input_ids=mlm_input_ids1, attention_mask=mlm_attention_mask1, labels=mlm_labels1)
            mlm_out2 = model(input_ids=mlm_input_ids2, attention_mask=mlm_attention_mask2, labels=mlm_labels2)
            if mlm_out1.loss is not None and mlm_out2.loss is not None:
                # In a multi-GPU setup, the loss can be a vector. We take the mean to get a scalar.
                mlm_loss = (mlm_out1.loss + mlm_out2.loss).mean()
                weighted_mlm_loss = self.config["loss"]["lambda_mlm"] * mlm_loss
                loss += weighted_mlm_loss
                mlm_loss_val = mlm_loss.detach()
                log_payload["train/mlm_loss"] = mlm_loss.item()
                log_payload["train/weighted_mlm_loss"] = weighted_mlm_loss.item()

        if pos_mask.any():
            gt_alignments_dev = gt_alignments.to(device)
            pred_plan_pos = transport_plan[pos_mask]

            # Slice the ground truth tensor to match the prediction's shape for the loss calculation.
            # We don't modify the main gt_alignments tensor so that the Trainer can correctly
            # gather tensors of a consistent shape (padded to max_len).
            _, N_plan, M_plan = pred_plan_pos.shape
            gt_align_full = gt_alignments_dev[pos_mask]

            # Slice gt_alignments to match the shape of pred_plan_pos.
            # The shape of gt_alignments is padded to max_len, so we need to handle cases
            # where N_plan or M_plan are larger than the padded dimensions.
            slice_N = min(N_plan, gt_align_full.shape[1])
            slice_M = min(M_plan, gt_align_full.shape[2])
            gt_align_sliced = gt_align_full[:, :slice_N, :slice_M]

            # Pad gt_align_sliced if it's smaller than pred_plan_pos
            if gt_align_sliced.shape[1] < N_plan or gt_align_sliced.shape[2] < M_plan:
                padding = (0, M_plan - gt_align_sliced.shape[2], 0, N_plan - gt_align_sliced.shape[1])
                gt_align_sliced = torch.nn.functional.pad(gt_align_sliced, padding, "constant", 0)

            l_alignment = generalized_kl_divergence(gt_align_sliced, pred_plan_pos).mean()
            l_sparsity = torch.sum(torch.abs(pred_plan_pos), dim=[1, 2]).mean()
            weighted_l_sparsity = self.config["loss"]["lambda_pos"] * l_sparsity
            loss += l_alignment + weighted_l_sparsity
            if self.is_in_train:
                log_payload["train/l_alignment"] = l_alignment.item()
                log_payload["train/l_sparsity"] = l_sparsity.item()
                log_payload["train/weighted_l_sparsity"] = weighted_l_sparsity.item()

        if (~pos_mask).any():
            pred_plan_neg = transport_plan[~pos_mask]
            l_emptiness = torch.sum(torch.abs(pred_plan_neg), dim=[1, 2]).mean()
            weighted_l_emptiness = self.config["loss"]["lambda_neg"] * l_emptiness
            loss += weighted_l_emptiness
            if self.is_in_train:
                log_payload["train/l_emptiness"] = l_emptiness.item()
                log_payload["train/weighted_l_emptiness"] = weighted_l_emptiness.item()

        if self.is_in_train and self.accelerator.is_main_process and log_payload:
            wandb.log(log_payload)

        if return_outputs:
            # Pad transport_plan to the max length before returning so the Trainer can gather it
            B, N, M = transport_plan.shape
            max_len1 = self.config.get("max_len1")
            max_len2 = self.config.get("max_len2")

            if max_len1 and max_len2:
                padded_transport_plan = torch.nn.functional.pad(transport_plan, (0, max_len2 - M, 0, max_len1 - N), "constant", 0)
            else:
                padded_transport_plan = transport_plan

            return (loss, (padded_transport_plan, gt_alignments, is_positive, mlm_loss_val))
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # The trainer expects (loss, logits, labels)
        # We can return the outputs as "logits" and None for labels,
        # as compute_metrics knows how to unpack the outputs tuple.
        return (loss, outputs, None)


def train(config_path: str, eval_only: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_full_head = config.get("train_full_head", False)
    eval_before_train = config.get("eval_before_train", False)

    logging.info("Loading datasets...")
    train_dataset = CATHDataset(data_root=config["data_root"], split="train")
    val_dataset = CATHDataset(data_root=config["data_root"], split="validation")

    plm_adaptor, _, _ = get_plm_adaptor_and_configs(config["model_name"], for_masked_lm=True)
    mlm_collator = MLMCollator(tokenizer=plm_adaptor.tokenizer)
    collator = OTAlignCollator(
        mlm_collator=mlm_collator,
        max_len1=config.get("max_len1"),
        max_len2=config.get("max_len2"),
    )

    def compute_metrics(p: EvalPrediction):
        # Unpack predictions, which might be nested in a tuple
        predictions = p.predictions
        if isinstance(predictions, tuple) and len(predictions) == 1:
            predictions = predictions[0]

        if not isinstance(predictions, tuple) or len(predictions) != 4:
            logging.warning(f"Unexpected predictions format in compute_metrics. Got type {type(predictions)}.")
            return {}
        transport_plans, gt_alignments, is_positives, mlm_losses = predictions
        band_width = config.get("eval_band_width", 5)

        # Filter out empty tensors from skipped batches
        valid_indices = [i for i, arr in enumerate(transport_plans) if arr.size > 0]
        if not valid_indices:
            return {f"in_band_mass_w_{band_width}": 0, f"recall_in_band_w_{band_width}": 0}

        transport_plans = np.array([transport_plans[i] for i in valid_indices])
        gt_alignments = np.array([gt_alignments[i] for i in valid_indices])
        is_positives = np.array([is_positives[i] for i in valid_indices])
        mlm_losses = np.array([mlm_losses[i] for i in valid_indices])

        # Flatten the arrays if they are nested
        if is_positives.ndim > 1:
            is_positives = np.concatenate(is_positives)
            transport_plans = np.concatenate(transport_plans)
            gt_alignments = np.concatenate(gt_alignments)

        in_band_masses, recalls_in_band = [], []
        pos_mask = is_positives.astype(bool)

        if pos_mask.any():
            # Move tensors to GPU if available for accelerated computation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gt_align = torch.from_numpy(gt_alignments[pos_mask]).to(device)
            pred_plan_pos = torch.from_numpy(transport_plans[pos_mask]).to(device)

            pred_plan_sum = torch.sum(pred_plan_pos, dim=(1, 2), keepdim=True).clamp(min=1e-8)
            normalized_pred_plan = pred_plan_pos / pred_plan_sum

            # Fully vectorized computation
            masses = vectorized_in_band_mass(normalized_pred_plan, gt_align, band_width)
            recalls = vectorized_recall_in_band(normalized_pred_plan, gt_align, band_width)

            in_band_masses.extend(masses.cpu().numpy())
            recalls_in_band.extend(recalls.cpu().numpy())

        metrics = {
            f"in_band_mass_w_{band_width}": np.mean(in_band_masses) if in_band_masses else 0,
            f"recall_in_band_w_{band_width}": np.mean(recalls_in_band) if recalls_in_band else 0,
        }
        if "lambda_mlm" in config["loss"] and config["loss"]["lambda_mlm"] > 0 and mlm_losses.size > 0:
            metrics["mlm_loss"] = np.mean([lv for lv in mlm_losses if lv is not None])
        return metrics

    logging.info(f"Loading PLM adaptor for '{config['model_name']}'...")
    model = plm_adaptor.model

    # Freeze parameters based on training strategy
    if not train_full_head:
        for param in model.parameters():
            param.requires_grad = False
    else:
        logging.info("Training the full language model head.")
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        lora_model = load_peft_model_from_checkpoint(model, resume_from_checkpoint)
    else:
        logging.info("Applying new LoRA with PEFT...")
        lora_params = config["lora"].copy()
        if "dropout" in lora_params:
            lora_params["lora_dropout"] = lora_params.pop("dropout")
        lora_config = LoraConfig(**lora_params)
        lora_model = get_peft_model(model, lora_config)

    run_name = config.get("run_name", "invalid_id")

    training_args_dict = {
        "run_name": run_name,
        "num_train_epochs": config["epochs"],
        "learning_rate": config["lr"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "logging_steps": config["log_interval"],
        "eval_strategy": "epoch",
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
        "label_names": [],
        "report_to": config.get("report_to"),
    }
    training_args = TrainingArguments(**training_args_dict)

    # Create the custom callback
    eval_save_callback = CustomEvalAndSaveCallback()
    wandb_callback = WandbStopTrainingCallback()

    trainer = OTAlignTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        plm_adaptor=plm_adaptor,
        custom_config=config,
        callbacks=[wandb_callback, eval_save_callback],
        compute_metrics=compute_metrics,
    )

    accelerator = trainer.accelerator

    setup_logging(Path(config["log_dir"]), run_name, accelerator)

    output_dir = Path(config["checkpoint_dir"]) / run_name
    trainer.args.output_dir = str(output_dir)

    wandb_callback.accelerator = accelerator
    eval_save_callback.accelerator = accelerator

    if accelerator.is_main_process:
        lora_model.print_trainable_parameters()
        logging.info("--- Verifying Training Arguments ---")
        logging.info(f"  Output Dir: {trainer.args.output_dir}")
        logging.info(f"  Learning Rate: {trainer.args.learning_rate}")
        logging.info(f"  Num Train Epochs: {trainer.args.num_train_epochs}")
        logging.info(f"  Train Batch Size per Device: {trainer.args.per_device_train_batch_size}")
        logging.info(f"  Gradient Accumulation Steps: {trainer.args.gradient_accumulation_steps}")
        logging.info("------------------------------------")

    # Inject the trainer instance into the callback
    eval_save_callback.trainer = trainer

    if eval_before_train or eval_only:
        logging.info(f"Running evaluation{' only' if eval_only else ' before training'}...")
        metrics = trainer.evaluate()
        if accelerator.is_main_process:
            logging.info(f"Initial evaluation metrics: {metrics}")

    if eval_only:
        if accelerator.is_main_process:
            logging.info("Evaluation finished. Exiting.")
        return

    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train OTAlign model with LoRA.")
    parser.add_argument("config_path", help="Path to the configuration YAML file.")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only on the validation set and exit.")
    args = parser.parse_args()

    train(config_path=args.config_path, eval_only=args.eval_only)


if __name__ == "__main__":
    main()

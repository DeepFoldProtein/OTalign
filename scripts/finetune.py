import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed

from otalign.align.cost import pairwise_cosine
from otalign.data.cath import CATHDataset
from otalign.data.collator import OTAlignCollator
from otalign.data.mlm_collator import MLMCollator
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn
from otalign.models.ot_head import get_ot_head
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


class UncertaintyWeighter(nn.Module):
    """
    Kendall & Gal task-uncertainty weighting.
    - Use add_task(...) once per task at init.
    - Call forward(losses, actives) each step to get total loss & diagnostics.
    """

    def __init__(self):
        super().__init__()
        self.logvars = nn.ParameterDict()
        self.min_logvars = {}
        self.max_logvars = {}

    def add_task(self, name: str, init_logvar: float = 0.0, trainable: bool = True, min_logvar: float = -4.0, max_logvar: float = 4.0):
        p = nn.Parameter(torch.tensor(float(init_logvar)))
        p.requires_grad_(trainable)
        self.logvars[name] = p
        self.min_logvars[name] = min_logvar
        self.max_logvars[name] = max_logvar

    @torch.no_grad()
    def clamp_(self):
        for name, p in self.logvars.items():
            p.clamp_(self.min_logvars[name], self.max_logvars[name])

    def forward(self, losses: dict, actives: dict):
        """
        losses: {name: scalar Tensor}
        actives: {name: bool}  # if False, exclude both L_i and +s_i
        Returns: total_loss, details(dict)
        """
        total = 0.0
        details = {}
        for name, Li in losses.items():
            si = self.logvars[name]
            active = bool(actives.get(name, True))
            if not active:
                wi = torch.exp(-si.detach())
                details[name] = {"active": False, "L": Li.detach().item(), "w": wi.item(), "logvar": si.detach().item()}
                continue
            wi = torch.exp(-si)  # learnable weight
            term = wi * Li + si  # α_i=1 when active
            total = total + term
            details[name] = {"active": True, "L": Li.detach().item(), "w": wi.detach().item(), "logvar": si.detach().item()}
        return total, details


class WandbStopTrainingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.accelerator: Accelerator | None = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.accelerator and self.accelerator.is_main_process and wandb.run and getattr(wandb.run, "state", "running") != "running":
            logging.warning("wandb run stopped from UI. Stopping training.")
            control.should_training_stop = True
        return control


class UncertaintyWeightingCallback(TrainerCallback):
    def __init__(self, uncertainty_weighter):
        self.uncertainty_weighter = uncertainty_weighter

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.uncertainty_weighter and self.uncertainty_weighter.training:
            self.uncertainty_weighter.clamp_()


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
    def __init__(self, *args, plm_adaptor, custom_config, uncertainty_weighter=None, ot_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plm_adaptor = plm_adaptor
        self.config = custom_config
        self.uncertainty_weighter = uncertainty_weighter
        self.ot_head = ot_head
        if self.uncertainty_weighter:
            self.uncertainty_weighter.to(self.accelerator.device)
        if self.ot_head:
            self.ot_head.to(self.accelerator.device)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        super().save_model(output_dir, _internal_call)
        if self.ot_head and self.accelerator.is_main_process:
            ot_head_path = os.path.join(output_dir, "ot_head.pth")
            self.accelerator.save(self.accelerator.unwrap_model(self.ot_head).state_dict(), ot_head_path)
            logging.info(f"Saved OT head to {ot_head_path}")

    def create_optimizer(self):
        if self.optimizer is None:
            optim_config = self.config.get("optim", {})
            ot_head_lr = self.config.get("ot_head_lr", self.args.learning_rate)

            params_to_optimize = [{"params": self.model.parameters(), "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay}]

            if self.ot_head:
                params_to_optimize.append({"params": self.ot_head.parameters(), "lr": ot_head_lr, "weight_decay": self.args.weight_decay})

            if self.uncertainty_weighter and self.uncertainty_weighter.logvars:
                logvar_lr = optim_config.get("logvar_lr", self.args.learning_rate)
                params_to_optimize.append({"params": self.uncertainty_weighter.logvars.parameters(), "lr": logvar_lr, "weight_decay": 0.0})

            self.optimizer = torch.optim.AdamW(
                params_to_optimize,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        else:
            super().create_optimizer()

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

        if self.ot_head:
            emb1 = self.ot_head(emb1)
            emb2 = self.ot_head(emb2)

        if not torch.isfinite(emb1).all() or not torch.isfinite(emb2).all():
            logging.warning("Embeddings contain non-finite values. Skipping batch.")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_outputs:
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
        log_payload = {}

        # --- MLM Loss ---
        mlm_loss = None
        use_mlm = (self.uncertainty_weighter and "MLM" in self.uncertainty_weighter.logvars) or ("lambda_mlm" in self.config["loss"] and self.config["loss"]["lambda_mlm"] > 0)
        if use_mlm:
            mlm_out1 = model(input_ids=mlm_input_ids1, attention_mask=mlm_attention_mask1, labels=mlm_labels1)
            mlm_out2 = model(input_ids=mlm_input_ids2, attention_mask=mlm_attention_mask2, labels=mlm_labels2)
            if mlm_out1.loss is not None and mlm_out2.loss is not None:
                mlm_loss = (mlm_out1.loss + mlm_out2.loss).mean()

        mlm_loss_val = mlm_loss.detach() if mlm_loss is not None and torch.isfinite(mlm_loss) else torch.tensor(0.0, device=device)

        # --- OT/Plan Losses ---
        num_pos = pos_mask.sum()
        num_neg = (~pos_mask).sum()
        eps = 1e-8

        l_alignment = torch.tensor(0.0, device=device)
        l_sparsity = torch.tensor(0.0, device=device)
        plan_active = False
        if num_pos > 0:
            pred_plan_pos = transport_plan[pos_mask]
            gt_alignments_dev = gt_alignments.to(device)
            _, N_plan, M_plan = pred_plan_pos.shape
            gt_align_full = gt_alignments_dev[pos_mask]
            slice_N, slice_M = min(N_plan, gt_align_full.shape[1]), min(M_plan, gt_align_full.shape[2])
            gt_align_sliced = gt_align_full[:, :slice_N, :slice_M]
            if gt_align_sliced.shape[1] < N_plan or gt_align_sliced.shape[2] < M_plan:
                padding = (0, M_plan - gt_align_sliced.shape[2], 0, N_plan - gt_align_sliced.shape[1])
                gt_align_sliced = torch.nn.functional.pad(gt_align_sliced, padding, "constant", 0)

            l_alignment_per_sample = generalized_kl_divergence(gt_align_sliced, pred_plan_pos) / (pred_plan_pos.sum((1, 2)) + eps)
            l_sparsity_per_sample = torch.sum(torch.abs(pred_plan_pos), dim=[1, 2])

            l_alignment = l_alignment_per_sample.sum() / num_pos
            l_sparsity = l_sparsity_per_sample.sum() / num_pos
            plan_active = True

        l_emptiness = torch.tensor(0.0, device=device)
        if num_neg > 0:
            pred_plan_neg = transport_plan[~pos_mask]
            l_emptiness_per_sample = torch.sum(torch.abs(pred_plan_neg), dim=[1, 2])
            l_emptiness = l_emptiness_per_sample.sum() / num_neg

        # --- Combine losses ---
        if self.uncertainty_weighter and self.is_in_train:
            task_losses, task_actives = {}, {}
            ot_loss_total = l_sparsity * num_pos + l_emptiness * num_neg
            ot_loss_norm = ot_loss_total / (num_pos + num_neg).clamp(min=1)
            task_losses["OT"] = ot_loss_norm
            task_actives["OT"] = True
            task_losses["PLAN"] = l_alignment
            task_actives["PLAN"] = plan_active
            mlm_active = mlm_loss is not None and torch.isfinite(mlm_loss)
            task_losses["MLM"] = mlm_loss if mlm_active else torch.tensor(0.0, device=device)
            task_actives["MLM"] = mlm_active

            loss, details = self.uncertainty_weighter(task_losses, task_actives)

            if self.accelerator.is_main_process:
                log_payload = {}
                for name, data in details.items():
                    log_payload[f"train/loss_{name}"] = data["L"]
                    log_payload[f"train/weight_{name}"] = data["w"]
                    log_payload[f"train/logvar_{name}"] = data["logvar"]
                    if data["active"]:
                        log_payload[f"train/weighted_loss_{name}"] = data["L"] * data["w"]
                log_payload["train/total_loss"] = loss.item()
                wandb.log(log_payload)
        else:
            loss = torch.tensor(0.0, device=device)
            if mlm_loss is not None and torch.isfinite(mlm_loss):
                weighted_mlm_loss = self.config["loss"]["lambda_mlm"] * mlm_loss
                loss += weighted_mlm_loss
                if self.is_in_train:
                    log_payload["train/mlm_loss"] = mlm_loss.item()
                    log_payload["train/weighted_mlm_loss"] = weighted_mlm_loss.item()
            if num_pos > 0:
                pos_loss = l_alignment + self.config["loss"]["lambda_pos"] * l_sparsity
                loss += pos_loss
                if self.is_in_train:
                    log_payload["train/l_alignment"] = l_alignment.item()
                    log_payload["train/l_sparsity"] = l_sparsity.item()
                    log_payload["train/weighted_l_sparsity"] = (self.config["loss"]["lambda_pos"] * l_sparsity).item()
            if num_neg > 0:
                neg_loss = self.config["loss"]["lambda_neg"] * l_emptiness
                loss += neg_loss
                if self.is_in_train:
                    log_payload["train/l_emptiness"] = l_emptiness.item()
                    log_payload["train/weighted_l_emptiness"] = neg_loss.item()
            if self.is_in_train and self.accelerator.is_main_process and log_payload:
                wandb.log(log_payload)

        if return_outputs:
            B, N, M = transport_plan.shape
            max_len1, max_len2 = self.config.get("max_len1"), self.config.get("max_len2")
            padded_transport_plan = torch.nn.functional.pad(transport_plan, (0, max_len2 - M, 0, max_len1 - N), "constant", 0) if max_len1 and max_len2 else transport_plan
            mlm_loss_val_expanded = mlm_loss_val.expand(B)
            return (loss, (padded_transport_plan, gt_alignments, is_positive, mlm_loss_val_expanded))
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

        # The trainer expects (loss, logits, labels).
        # For compute_metrics to be called, labels cannot be None.
        # We pass a dummy tensor for labels, as compute_metrics unpacks the ground truth from the 'outputs' tuple.
        dummy_labels = torch.zeros(inputs["lens1"].size(0), device=self.accelerator.device)
        return (loss, outputs, dummy_labels)


def finetune(config_path: str, eval_only: bool = False, seed: Optional[int] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # If seed is provided as an argument, it overrides the config.
    if seed is not None:
        config["seed"] = seed

    if "seed" in config:
        logging.info(f"Setting seed to {config['seed']}")
        set_seed(config["seed"])

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
            logging.warning(f"Unexpected predictions format in compute_metrics. Got type {type(predictions)}, len {len(predictions)}.")
            return {}
        transport_plans, gt_alignments, is_positives, mlm_losses = predictions
        band_width = config.get("eval_band_width", 5)

        # Filter out empty tensors from skipped batches
        valid_indices = [i for i, arr in enumerate(transport_plans) if arr.size > 0]
        if not valid_indices:
            return {f"in_band_mass_w_{band_width}": 0}

        transport_plans = np.array([transport_plans[i] for i in valid_indices])
        gt_alignments = np.array([gt_alignments[i] for i in valid_indices])
        is_positives = np.array([is_positives[i] for i in valid_indices])
        mlm_losses = np.array([mlm_losses[i] for i in valid_indices])

        # Flatten the arrays if they are nested
        if is_positives.ndim > 1:
            is_positives = np.concatenate(is_positives)
            transport_plans = np.concatenate(transport_plans)
            gt_alignments = np.concatenate(gt_alignments)
            mlm_losses = np.concatenate(mlm_losses)

        metrics = {}
        if "lambda_mlm" in config["loss"] and config["loss"]["lambda_mlm"] > 0 and mlm_losses.size > 0:
            valid_mlm_losses = [lv for lv in mlm_losses if lv is not None and lv > 0]
            if valid_mlm_losses:
                metrics["mlm_loss"] = np.mean(valid_mlm_losses)
        return metrics

    logging.info(f"Loading PLM adaptor for '{config['model_name']}'...")
    model = plm_adaptor.model

    # --- OT Head ---
    ot_head = None
    ot_head_config = config.get("ot_head", {})
    if ot_head_config.get("enabled", False):
        logging.info("OT Head is enabled. Initializing...")
        # Infer input dimension from the model's hidden size
        ot_head_config["input_dim"] = model.config.hidden_size
        ot_head = get_ot_head(ot_head_config)
        logging.info(f"Initialized OT Head: {ot_head}")

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
        if ot_head:
            ot_head_path = os.path.join(resume_from_checkpoint, "ot_head.pth")
            if os.path.exists(ot_head_path):
                logging.info(f"Loading OT head from {ot_head_path}")
                ot_head.load_state_dict(torch.load(ot_head_path))
            else:
                logging.warning(f"OT head checkpoint not found at {ot_head_path}. Starting with a fresh OT head.")
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

    # --- Uncertainty Weighter ---
    uncertainty_weighter = None
    callbacks = []
    if config.get("uncertainty", {}).get("enabled", False):
        uw_config = config["uncertainty"]
        uncertainty_weighter = UncertaintyWeighter()
        global_min_logvar = uw_config.get("min_logvar", -4.0)
        global_max_logvar = uw_config.get("max_logvar", 4.0)
        for task in uw_config.get("tasks", []):
            uncertainty_weighter.add_task(
                name=task["name"],
                init_logvar=task.get("init_logvar", 0.0),
                trainable=task.get("trainable", True),
                min_logvar=task.get("min_logvar", global_min_logvar),
                max_logvar=task.get("max_logvar", global_max_logvar),
            )
        callbacks.append(UncertaintyWeightingCallback(uncertainty_weighter))

    # Create the custom callback
    eval_save_callback = CustomEvalAndSaveCallback()
    wandb_callback = WandbStopTrainingCallback()
    callbacks.extend([wandb_callback, eval_save_callback])

    trainer = OTAlignTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        plm_adaptor=plm_adaptor,
        custom_config=config,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        uncertainty_weighter=uncertainty_weighter,
        ot_head=ot_head,
    )

    accelerator = trainer.accelerator

    output_dir = Path(config["checkpoint_dir"]) / run_name
    trainer.args.output_dir = str(output_dir)

    # Now that the output dir is set, setup logging and save config
    setup_logging(Path(config["log_dir"]), run_name, accelerator)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, output_dir / "config.yaml")
        logging.info(f"Saved config to {output_dir / 'config.yaml'}")

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

    logging.info("Starting finetuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Finetune OTAlign model with LoRA and an OT Head.")
    parser.add_argument("config_path", help="Path to the configuration YAML file.")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only on the validation set and exit.")
    parser.add_argument("--seed", type=int, default=None, help="An integer seed for reproducibility.")
    args = parser.parse_args()

    finetune(config_path=args.config_path, eval_only=args.eval_only, seed=args.seed)


if __name__ == "__main__":
    main()

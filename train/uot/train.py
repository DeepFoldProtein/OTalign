import argparse
import logging
import os
import pathlib
from typing import Any

import h5py
import torch
import torch.nn as nn
import wandb
import yaml
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from otalign.datasets.sabmark import SABmark
from otalign.datasets.sabmark import make_collate_fn as sabmark_collate_fn
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn as sinkhorn
from otalign.utils.log_utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)
JOB_ID = os.environ.get("SLURM_JOB_ID", "null")


class FrozenModelWrapper(nn.Module):
    """
    Wrapper around a frozen pretrained transformer model that caches embeddings
    (last hidden state excluding [CLS] and [SEP]) into an HDF5 file.

    Handles variable-length inputs by padding embeddings to either the batch's
    maximum length or a user-provided `pad_to_length`. Returns embeddings and mask.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        hdf5_path: str | None = None,
        device: torch.device | str = torch.get_default_device(),
        pad_to_length: int | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.pad_to_length = pad_to_length

        # Load tokenizer and model
        maxlen = None if self.pad_to_length is None else self.pad_to_length + 2
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, model_max_length=maxlen)
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.model.to(self.device)

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Open HDF5 file for read/write caching
        if hdf5_path:
            self.h5_file = h5py.File(hdf5_path, "a")
        else:
            self.h5_file = None

    def forward(self, input_texts: list):
        """
        Tokenizes a batch of texts, loads or computes embeddings (seq_len-2 x hidden_dim),
        pads them to either `pad_to_length` (if set) or the maximum length in the batch,
        filling with zeros on the right. Returns:
          - embeddings: Tensor of shape (batch_size, target_len, hidden_dim)
          - mask: Boolean Tensor of shape (batch_size, target_len) with True for valid tokens
        """
        # Tokenize batch
        encoding = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        embeddings_list = []
        lengths = []
        for idx in range(input_ids.size(0)):
            seq_ids = input_ids[idx].tolist()
            key = "_".join(map(str, seq_ids))

            if self.h5_file is not None and key in self.h5_file:
                np_emb = self.h5_file[key][:]  # type: ignore
                emb = torch.from_numpy(np_emb).to(self.device)
            else:
                single_ids = input_ids[idx].unsqueeze(0).to(self.device)
                single_mask = attention_mask[idx].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=single_ids, attention_mask=single_mask)
                    hidden = outputs.last_hidden_state
                    emb = hidden[:, 1:-1, :].squeeze(0)

                # np_data = emb.cpu().numpy()
                # self.h5_file.create_dataset(key, data=np_data, compression="gzip")
                # self.h5_file.flush()

            embeddings_list.append(emb)
            lengths.append(emb.size(0))

        # Determine target length
        if self.pad_to_length is not None:
            target_len = self.pad_to_length
        else:
            target_len = max(lengths)

        hidden_dim = embeddings_list[0].size(1)
        batch_size = len(embeddings_list)

        # Prepare padded tensor and mask
        padded = torch.zeros(batch_size, target_len, hidden_dim, device=self.device)
        mask = torch.zeros(batch_size, target_len, dtype=torch.bool, device=self.device)

        for i, (emb, seq_len) in enumerate(zip(embeddings_list, lengths)):
            if seq_len >= target_len:
                truncated = emb[:target_len]
                padded[i] = truncated
                mask[i] = torch.ones(target_len, dtype=torch.bool, device=self.device)
            else:
                padded[i, :seq_len] = emb
                mask[i, :seq_len] = True
                # remainder is already zero from initialization

        return padded, mask

    def close(self):
        """
        Close the underlying HDF5 file. Call when done using the wrapper.
        """
        if self.h5_file is not None:
            self.h5_file.close()


class ResidueHead(nn.Module):
    def __init__(
        self,
        input_dim_from_pretrained: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_prob: float = 0.5,
        residual: bool = False,
    ):
        """
        Initializes the LastHiddenLayerTransformer.

        Args:
            input_dim_from_pretrained (int): The dimensionality of the last hidden
                                             layer output from the `pretrained_model`.
            output_dim (int): The desired output dimensionality after transformation.
                              This could be the number of classes for classification,
                              or another feature dimension for further processing.
            hidden_dims (list, optional): A list of integers specifying the
                                          dimensions of additional hidden layers
                                          between the input and the final output layer.
                                          If None or empty, a direct linear
                                          transformation is applied. Defaults to None.
            dropout_prob (float, optional): The dropout probability to apply
                                            between transformation layers.
                                            Defaults to 0.5.
            residual (bool): Enable residual connection. Defaults to False.
        """
        super().__init__()

        self.enable_residual = residual
        self.input_dim_from_pretrained = input_dim_from_pretrained
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        if residual:
            assert output_dim == input_dim_from_pretrained

        layers = []
        current_dim = input_dim_from_pretrained

        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())  # Using ReLU as a common activation function
                layers.append(nn.Dropout(self.dropout_prob))
                current_dim = h_dim

        # Add the final output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Combine all transformation layers into a Sequential module
        self.transformation_head = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, hidden_state: torch.Tensor):
        """
        Applies the transformation head.

        Args:
            hidden_state (torch.Tensor): The input last hidden state from the pre-trained model.

        Returns:
            torch.Tensor: The transformed output from the embedding.
        """
        transformed_output = self.transformation_head(hidden_state)

        if self.enable_residual:
            output_state = hidden_state + transformed_output
        else:
            output_state = transformed_output

        return output_state


def normalize(
    input: torch.Tensor,
    mask: torch.Tensor,
    p: float = 2.0,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    denom = (input * mask[..., None]).norm(p, dim, keepdim=True).clamp_min(eps)
    return input / denom


class SinkhornOT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, mask_a, mask_b, num_iter, reg, lambda1, lambda2):
        """Discrete Optimal Transport."""

        assert a.size(0) == b.size(0)
        B = a.size(0)
        # M = a.size(1)
        # N = b.size(1)

        # Marginals
        with torch.no_grad():
            row_mask = mask_a.unsqueeze(-1)  # [..., m, 1]
            col_mask = mask_b.unsqueeze(-2)  # [..., 1, n]
            full_mask = row_mask & col_mask  # [..., m, n]

            u = mask_a.float()
            v = mask_b.float()
            u /= u.sum(-1, keepdim=True)
            v /= v.sum(-1, keepdim=True)

        # Cost
        norm_a = normalize(a, mask_a)
        norm_b = normalize(b, mask_b)
        cost = 1 - torch.einsum("bik,bjk->bij", norm_a, norm_b)

        # u = mask_a.new_ones((B, M)) / mask_a.sum(-1, keepdim=True)
        # v = mask_b.new_ones((B, N)) / mask_b.sum(-1, keepdim=True)

        # Sinkhron core
        plan = sinkhorn(cost, u, v, num_iter, reg, lambda1, lambda2, mask_a, mask_b)

        # Masked expected cost & entropy
        valid = full_mask  # alias

        # exp_cost = torch.where(full_mask, plan * cost, 0).sum(dim=(-2, -1))
        # neg_ent = torch.where(full_mask, plan * torch.log(plan), 0).sum(dim=(-2, -1))

        plan_clamped = plan.clamp_min(torch.finfo(plan.dtype).eps)
        exp_cost = (plan * cost)[valid].view(B, -1).sum(-1)
        neg_ent = (plan_clamped * plan_clamped.log())[valid].view(B, -1).sum(-1)

        objective = exp_cost + reg * neg_ent

        return {
            "expected_cost": exp_cost,
            "negative_entropy": neg_ent,
            "objective": objective,
            "cost_matrix": cost,
            "transport_plan": plan,
        }


def compute_losses(
    plan: torch.Tensor,
    target: torch.Tensor,
    full_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return each individual loss term (no weighting, reduced over batch).

    Args
    -----
    plan, target: (B, M, N) tensors on *the same* device.
    full_mask:     boolean mask of valid locations (B, M, N).

    Returns
    -------
    Dict with keys.
    """
    eps = torch.finfo(plan.dtype).eps

    # ------------------------------------------------------------------
    # Cross‑entropy on the *aligned* positions only.
    # ------------------------------------------------------------------
    num_active = target.sum(dim=(-1, -2))  # (B,)

    norm_plan = plan / plan.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    log_plan = norm_plan.clamp_min(eps).log()

    weighted_log = (target * log_plan).sum(dim=(-1, -2))  # (B,)
    per_item_ce = torch.where(num_active > 0, weighted_log / num_active, 0.0)
    ce_loss = -per_item_ce.mean()

    # ------------------------------------------------------------------
    # Negative entropy over the *valid* area (to encourage peaky plans).
    # ------------------------------------------------------------------
    neg_ent_terms = full_mask * norm_plan * log_plan
    neg_ent = -neg_ent_terms.sum(dim=(-1, -2)).mean()

    # ------------------------------------------------------------------
    # Soft precision surrogate: expected alignment probability mass.
    # ------------------------------------------------------------------
    sp_loss = -(norm_plan * target).sum(dim=(-1, -2)).mean()

    # For diagnostics only – not necessarily used in ``total_loss``.
    true_loss = -(norm_plan * target).sum(dim=(-1, -2)).mean()
    false_loss = (norm_plan * (~target.bool()).float()).sum(dim=(-1, -2)).mean()

    return {
        "ce_loss": ce_loss,
        "ent_loss": neg_ent,
        "sp_loss": sp_loss,
        "true_loss": true_loss,
        "false_loss": false_loss,
    }


def train(
    wrapper_model: torch.nn.Module,
    transform: torch.nn.Module,
    aligner: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    cfg,
    outdir: pathlib.Path,
):
    device = torch.device(cfg["device"])

    #
    optimizer = torch.optim.Adam(
        transform.parameters(),
        lr=cfg["learning_rate"],
        betas=cfg["betas"],
    )
    scaler = GradScaler(enabled=cfg.get("use_amp", True) and device.type == "cuda")

    # Ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    accumulation_steps = cfg["accumulation_steps"]

    for epoch in range(cfg["epoch"]):
        # TRAINING PHASE

        wrapper_model.train()
        transform.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_epoch = torch.tensor(0.0, device=cfg["device"], requires_grad=False)
        train_batches = 0

        pbar = tqdm(dataloaders["train"], leave=False, desc=f"Train {epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                # Forward -------------------------------------------------
                a_raw, mask_a = wrapper_model(batch["seq_a"])
                b_raw, mask_b = wrapper_model(batch["seq_b"])

                # single call through *transform*
                combined = torch.cat([a_raw, b_raw], dim=1)
                combined = transform(combined)
                a, b = combined.split([a_raw.size(1), b_raw.size(1)], dim=1)

                res = aligner(
                    a,
                    b,
                    mask_a,
                    mask_b,
                    cfg["num_iter"],
                    cfg["reg"],
                    cfg["lambda1"],
                    cfg["lambda2"],
                )
                plan = res["transport_plan"]
                losses = compute_losses(plan, batch["aln"], batch["aln_mask"])
                total_loss = torch.stack([cfg["loss_weights"][k] * v for k, v in losses.items()]).sum()
                total_loss = total_loss / accumulation_steps

            # Back‑prop ----------------------------------------------------
            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # --------------------------------------------------------------
            train_loss_epoch += total_loss.detach()
            train_batches += 1
            global_step += 1

            # Lightweight logging every 5 micro‑steps ----------------------
            pbar.set_postfix()
            if global_step % 5 == 0:
                wandb.log(
                    {
                        "step": global_step,
                        "train/total_loss": total_loss.detach().cpu(),
                        **{f"train/{k}": v.detach().cpu() for k, v in losses.items()},
                    },
                    # commit=False,
                )
                pbar.set_postfix({k: f"{v.item():.3e}" for k, v in {"total_loss": total_loss, **losses}.items()})

        avg_train_loss = train_loss_epoch / train_batches

        # VALIDATION PHASE (no grad)
        wrapper_model.eval()
        transform.eval()

        val_loss_epoch = torch.tensor(0.0, device=cfg["device"], requires_grad=False)
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(dataloaders["valid"], leave=False, desc=f"Valid {epoch}"):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

                a_raw, mask_a = wrapper_model(batch["seq_a"])
                b_raw, mask_b = wrapper_model(batch["seq_b"])

                combined = torch.cat([a_raw, b_raw], dim=1)
                combined = transform(combined)
                a, b = combined.split([a_raw.size(1), b_raw.size(1)], dim=1)

                res = aligner(
                    a,
                    b,
                    mask_a,
                    mask_b,
                    cfg["num_iter"],
                    cfg["reg"],
                    cfg["lambda1"],
                    cfg["lambda2"],
                )
                plan = res["transport_plan"]

                losses = compute_losses(plan, batch["aln"], batch["aln_mask"])
                total_loss = torch.stack([cfg["loss_weights"][k] * v for k, v in losses.items()]).sum()

                val_loss_epoch += total_loss.detach()
                val_batches += 1

        avg_val_loss = val_loss_epoch / val_batches

        # LOGGING, CKPT

        wandb.log(
            {
                "epoch": epoch,
                "train/avg_loss": avg_train_loss.cpu(),
                "valid/avg_loss": avg_val_loss.cpu(),
            }
        )

        if epoch % cfg.get("save_every", 1) == 0:
            ckpt_path = outdir / f"ckpt-{epoch}.pth"
            torch.save(transform.state_dict(), ckpt_path)
            torch.save(transform.state_dict(), outdir / "latest.pth")
            wandb.save(str(ckpt_path))

        logger.info("Epoch %-3d | train %.4f | valid %.4f", epoch, avg_train_loss, avg_val_loss)


def build_wrapper_model(cfg) -> torch.nn.Module:
    """Initialise the frozen, pretrained language model wrapper.

    * Expects keys ``model_checkpoint``, ``device`` and ``max_seqlen`` in *cfg*.
    * The returned module is *already* moved onto the requested device.
    """
    checkpoint: str = cfg.get("model_checkpoint", "facebook/esm1b_t33_650M_UR50S")
    wrapper = FrozenModelWrapper(
        checkpoint,
        device=cfg["device"],
        pad_to_length=cfg["max_seqlen"],
    )
    return wrapper.to(cfg["device"])


def build_transform_head(cfg) -> torch.nn.Module:
    """Create the small residue-level projection head that follows ESM-1b.

    * ``input_dim`` defaults to 1280 (ESM-1b hidden size) but can be
      overridden via *cfg*.  ``hidden_dims`` and ``output_dim`` are read
      from *cfg*.
    """
    input_dim: int = cfg.get("input_dim", 1280)
    output_dim: int = cfg["output_dim"]
    hidden_dims: list[int] = cfg["hidden_dims"]
    dropout: float = cfg.get("transform_dropout", 0.5)
    residual: bool = cfg.get("residual", False)

    head = ResidueHead(input_dim, output_dim, hidden_dims, dropout, residual)
    return head.to(cfg["device"])


def build_aligner(cfg) -> torch.nn.Module:
    """Return a configured Sinkhorn optimal-transport aligner."""
    return SinkhornOT().to(cfg["device"])


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    """Create *train* and *valid* dataloaders for SABmark."""

    # Build raw SABmark dataset
    sab_sup = SABmark(
        cfg["sabmark_ref_path"],
        cfg["sabmark_ids_path"],
        regex="sup_*",  # keeps superfamily subsets only
    )
    sab_twi = SABmark(
        cfg["sabmark_ref_path"],
        cfg["sabmark_ids_path"],
        regex="twi_*",  # keeps superfamily subsets only
    )

    train_set = sab_sup
    val_set = sab_twi

    # Common DataLoader kwargs
    dl_kwargs: dict[str, Any] = {
        "collate_fn": sabmark_collate_fn(cfg["max_seqlen"]),
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": cfg["pin_memory"],
        "persistent_workers": cfg["persistent_workers"],
        "prefetch_factor": cfg["prefetch_factor"],
    }

    return {
        "train": DataLoader(train_set, shuffle=True, **dl_kwargs),
        "valid": DataLoader(val_set, shuffle=False, **dl_kwargs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--sweep", action="store_true", help="Run as wandb sweep agent.")
    args = parser.parse_args()

    def run(config=None):
        # Support both normal runs and wandb sweeps
        if config is None:
            # Load config from file if not running as sweep
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
        else:
            # Config is passed by wandb agent (flatten Namespace)
            cfg = dict(config)
            # Merge with file config for defaults
            with open(args.config) as f:
                file_cfg = yaml.safe_load(f)
            file_cfg.update(cfg)
            cfg = file_cfg

        # wandb.init() inside each run, pick up sweep config
        wandb.init(project=cfg.get("wandb_project", "ot-training"), config=cfg)
        cfg = wandb.config

        # Ensure correct type for sweeped fields
        if isinstance(cfg.get("hidden_dims"), str):
            # Convert "[256,128]" to [256,128]
            import ast

            cfg["hidden_dims"] = ast.literal_eval(cfg["hidden_dims"])

        if isinstance(cfg.get("betas"), str):
            cfg["betas"] = tuple(float(x) for x in cfg["betas"].strip("[]()").split(","))

        # build / load models
        wrapper_model = build_wrapper_model(cfg)
        transform = build_transform_head(cfg)
        aligner = build_aligner(cfg)

        wandb.watch(transform, log="all")
        dataloaders = build_dataloaders(cfg)

        if args.sweep:
            run_id: str = wandb.run.id  # type: ignore
            wandb.config.update({"outdir": os.path.join("ckpt", run_id)}, allow_val_change=True)
            outdir = pathlib.Path(cfg["outdir"])
        else:
            outdir = pathlib.Path(cfg.get("outdir", "checkpoints"))

        try:
            train(
                wrapper_model=wrapper_model,
                transform=transform,
                aligner=aligner,
                dataloaders=dataloaders,
                cfg=cfg,
                outdir=outdir,
            )
        finally:
            logger.info("Training finished.")
            wandb.finish()

    if args.sweep:
        # Running as sweep agent
        run()
    else:
        run()

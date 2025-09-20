import logging

from peft import LoraConfig, PeftModel


def load_peft_model_from_checkpoint(base_model, checkpoint_path: str):
    """
    Loads a PEFT model from a checkpoint.

    This function loads the LoRA adapter weights from a checkpoint file and
    applies them to a given base model.

    Args:
        base_model: The base PLM model (e.g., from a PLM adaptor).
        checkpoint_path (str): Path to the checkpoint file (.pt).

    Returns:
        The PEFT model with loaded adapter weights.
    """
    logging.info(f"Loading PEFT model from checkpoint: {checkpoint_path}")

    # The PEFT library saves the adapter weights in a subdirectory.
    # We assume the checkpoint contains the full model state_dict, and we
    # can infer the adapter path or load it directly.
    # A more robust way is to save adapter_weights separately.
    # For simplicity here, we assume the checkpoint IS the adapter state dict.

    # First, create a LoraConfig. The values don't matter here as they will be
    # overwritten by the loaded checkpoint, but they must be provided.
    # A better approach would be to save the LoraConfig in the checkpoint.
    try:
        peft_config = LoraConfig.from_pretrained(checkpoint_path)
    except Exception:
        logging.warning(f"Could not load LoraConfig from {checkpoint_path}. Using default values. This may fail if the architecture has changed.")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Placeholder
        )

    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=peft_config)
    logging.info("Successfully loaded PEFT model from checkpoint.")
    return peft_model

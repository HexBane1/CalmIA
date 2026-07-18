"""
Developer B work package: Checkpoint loading.

Loads a MusicSequenceModel checkpoint saved by developer_a/train.py. This module
imports developer_a.model only to instantiate the model class -- it does not
depend on developer_a/dataset.py or developer_a/train.py, so Developer B is never
blocked by changes to Developer A's data pipeline or training loop internals.

Testing without a real checkpoint:
    Developer B can validate the entire inference pipeline (this file, sampler.py,
    postprocess.py) before Developer A produces a trained checkpoint by calling
    `build_dummy_checkpoint()` below, which saves a randomly initialized model in
    the exact schema Developer A's train.py produces. This lets sampler.py and
    postprocess.py be fully implemented and unit-tested on day one.
"""

import os
from typing import Tuple

import torch

from shared.config import DATA_CONFIG, MODEL_CONFIG, CHECKPOINT_SCHEMA
from developer_a.model import MusicSequenceModel


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device = None
) -> Tuple[torch.nn.Module, dict]:
    """
    Loads a trained MusicSequenceModel from a checkpoint file.

    Returns:
        (model, metadata) where metadata contains epoch, global_step, val_loss
        as saved by developer_a/train.py's save_checkpoint().
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # weights_only=False is required because the checkpoint stores DataConfig /
    # ModelConfig dataclass instances alongside the tensors (see
    # shared.config.CHECKPOINT_SCHEMA). Only load checkpoints produced by
    # developer_a/train.py or developer_b/checkpoint_loader.py in this codebase.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    required_keys = [
        CHECKPOINT_SCHEMA.model_state_dict_key,
        CHECKPOINT_SCHEMA.epoch_key,
        CHECKPOINT_SCHEMA.step_key,
        CHECKPOINT_SCHEMA.val_loss_key,
    ]
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} is missing expected keys: {missing}. "
            f"Confirm shared/config.py CHECKPOINT_SCHEMA matches the version used "
            f"to save this checkpoint."
        )

    model = MusicSequenceModel(
        vocab_size=DATA_CONFIG.vocab_size,
        d_model=MODEL_CONFIG.d_model,
        n_heads=MODEL_CONFIG.n_heads,
        n_layers=MODEL_CONFIG.n_layers,
        d_feedforward=MODEL_CONFIG.d_feedforward,
        dropout=MODEL_CONFIG.dropout,
        max_sequence_length=DATA_CONFIG.max_sequence_length,
        pad_token_id=DATA_CONFIG.pad_token_id,
    )
    model.load_state_dict(checkpoint[CHECKPOINT_SCHEMA.model_state_dict_key])
    model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint[CHECKPOINT_SCHEMA.epoch_key],
        "global_step": checkpoint[CHECKPOINT_SCHEMA.step_key],
        "val_loss": checkpoint[CHECKPOINT_SCHEMA.val_loss_key],
        "checkpoint_path": checkpoint_path,
    }
    return model, metadata


def build_dummy_checkpoint(output_path: str) -> str:
    """
    Saves a randomly initialized MusicSequenceModel in the exact schema Developer
    A's train.py produces. For Developer B use only, to unblock inference/sampling
    development before a real trained checkpoint exists. Never use this checkpoint
    for actual music generation quality assessment.
    """
    model = MusicSequenceModel()
    dummy_optimizer_state = {}  # optimizer state is not required for inference-only loading

    checkpoint = {
        CHECKPOINT_SCHEMA.model_state_dict_key: model.state_dict(),
        CHECKPOINT_SCHEMA.optimizer_state_dict_key: dummy_optimizer_state,
        CHECKPOINT_SCHEMA.epoch_key: 0,
        CHECKPOINT_SCHEMA.step_key: 0,
        CHECKPOINT_SCHEMA.val_loss_key: float("nan"),
        CHECKPOINT_SCHEMA.data_config_key: DATA_CONFIG,
        CHECKPOINT_SCHEMA.model_config_key: MODEL_CONFIG,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(checkpoint, output_path)
    return output_path


if __name__ == "__main__":
    # Quick manual smoke test: build a dummy checkpoint and reload it.
    dummy_path = build_dummy_checkpoint("checkpoints/dummy_checkpoint.pt")
    loaded_model, meta = load_model_from_checkpoint(dummy_path)
    print(f"Loaded dummy model successfully. Metadata: {meta}")

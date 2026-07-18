"""
Shared configuration contract.

This module is the single source of truth for anything that must remain identical
between the training pipeline (Developer A) and the inference pipeline (Developer B).
Neither developer package should hardcode these values locally. If a value here needs
to change, it must be changed here and communicated to both developers, since it
affects checkpoint compatibility.

Do not import anything from developer_a/ or developer_b/ into this file.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    # Maximum number of tokens per training/inference sequence.
    # TODO(robert-dataset): revisit once average sequence length in Robert's dataset
    # is known. 512 is a reasonable default for event-based symbolic tokenization.
    max_sequence_length: int = 512

    # Vocabulary size. Must match shared.vocabulary.VOCAB_SIZE exactly.
    vocab_size: int = 512

    # Special token ids. Fixed here so both pipelines agree without importing
    # each other.
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass(frozen=True)
class ModelConfig:
    # Baseline lightweight Transformer decoder dimensions.
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_feedforward: int = 1024
    dropout: float = 0.1

    # Class name is part of the contract: Developer B's loader looks up this
    # exact class from developer_a.model. Do not rename without updating both sides.
    model_class_name: str = "MusicSequenceModel"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    num_epochs: int = 20
    grad_clip_norm: float = 1.0
    validation_split: float = 0.1
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 50


@dataclass(frozen=True)
class CheckpointSchema:
    """
    Defines the exact keys expected in a saved checkpoint dict. This is the
    integration point between Developer A (writer) and Developer B (reader).
    Both sides must agree on this without needing to read each other's code.
    """
    model_state_dict_key: str = "model_state_dict"
    optimizer_state_dict_key: str = "optimizer_state_dict"
    epoch_key: str = "epoch"
    step_key: str = "global_step"
    val_loss_key: str = "val_loss"
    data_config_key: str = "data_config"
    model_config_key: str = "model_config"


DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
TRAIN_CONFIG = TrainConfig()
CHECKPOINT_SCHEMA = CheckpointSchema()

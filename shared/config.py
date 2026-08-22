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
class ConditioningConfig:
    """
    Documents the physiological conditioning interface for the Week 2+ closed-loop
    system. This does NOT change how the Week 1 baseline behaves -- conditioning is
    opt-in per model instance via MusicSequenceModel(conditioning_dim=...), and
    defaults to disabled (0) everywhere it is constructed today, so existing
    checkpoints (trained with no conditioning) remain fully loadable and valid.

    feature_names is the placeholder set implied by the professor's coding task
    (HRV, EDA, respiration, movement quality). Extend/reorder this once real
    feature extraction from a physiological dataset (e.g. WESAD) is implemented --
    the order here must match the order features are concatenated into the
    conditioning_vector tensor passed to MusicSequenceModel.forward().
    """
    enabled: bool = False
    feature_names: tuple = ("hrv", "eda", "respiration_rate", "movement_quality")

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class DiscreteConditionConfig:
    """
    Mirrors the discrete condition token ids defined in shared/vocabulary.py
    (Task 2: tempo/complexity condition tokens). Duplicated intentionally,
    following the same pattern already used by DataConfig's pad/bos/eos token
    ids mirroring shared.vocabulary.PAD/BOS/EOS -- both sides must stay in
    sync if either changes.

    This is distinct from ConditioningConfig above: that class documents the
    separate, still-unused *continuous* conditioning pathway in
    developer_a/model.py (conditioning_dim). This class instead documents the
    discrete, token-based conditioning path actually wired up in Task 2, via
    developer_a/dataset.py prepending these token ids to each training window.
    """
    tempo_slow_token_id: int = 451
    tempo_fast_token_id: int = 452
    complexity_low_token_id: int = 453
    complexity_high_token_id: int = 454
    # Number of condition tokens prepended per training example (one tempo
    # token + one complexity token).
    num_condition_tokens_prepended: int = 2


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
CONDITIONING_CONFIG = ConditioningConfig()
DISCRETE_CONDITION_CONFIG = DiscreteConditionConfig()
TRAIN_CONFIG = TrainConfig()
CHECKPOINT_SCHEMA = CheckpointSchema()
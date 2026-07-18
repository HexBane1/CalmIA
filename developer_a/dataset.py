"""
Developer A work package: Data pipeline.

Provides a PyTorch Dataset and a DataLoader factory for tokenized symbolic music
sequences. This module depends only on shared/config.py and shared/vocabulary.py.
It does not import anything from developer_b/.

Expected on-disk layout (adapt via _load_index below):
    data_root/
        train/
            *.pt   (each file is a 1D torch.LongTensor of token ids, or)
            *.mid  (raw MIDI, tokenized on the fly via shared.vocabulary)
        val/
            ... same structure

TODO(robert-dataset): confirm the actual file layout and extension used by Robert's
dataset, then adjust _load_index and _load_sequence accordingly. Everything else
in this file (padding, batching, collation) is layout-agnostic and should not need
changes.
"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from shared.config import DATA_CONFIG, TRAIN_CONFIG
from shared.vocabulary import load_midi_as_tokens, PAD, BOS, EOS


class MusicSequenceDataset(Dataset):
    """
    Loads tokenized symbolic music sequences from disk.

    Each item returned is a tuple (input_ids, target_ids) where target_ids is
    input_ids shifted by one position, standard for autoregressive next-token
    training.
    """

    def __init__(self, data_root: str, split: str, max_sequence_length: int = None):
        self.data_root = data_root
        self.split = split
        self.max_sequence_length = max_sequence_length or DATA_CONFIG.max_sequence_length
        self.file_paths = self._load_index()

    def _load_index(self) -> List[str]:
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected split directory at {split_dir}. "
                f"TODO(robert-dataset): update _load_index if the directory "
                f"layout differs."
            )
        valid_extensions = (".pt", ".mid", ".midi")
        return sorted(
            os.path.join(split_dir, fname)
            for fname in os.listdir(split_dir)
            if fname.lower().endswith(valid_extensions)
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_sequence(self, path: str) -> torch.Tensor:
        if path.endswith(".pt"):
            tokens = torch.load(path)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            return tokens.long()
        # Fall back to on-the-fly MIDI tokenization.
        token_list = load_midi_as_tokens(path)
        return torch.tensor(token_list, dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._load_sequence(self.file_paths[index])

        # Truncate or pad to a fixed window. For sequences longer than the max
        # length, take a random contiguous crop to expose the model to varied
        # positions across epochs rather than always training on the prefix.
        seq_len = tokens.size(0)
        window = self.max_sequence_length + 1  # +1 because we shift for targets

        if seq_len >= window:
            start = torch.randint(0, seq_len - window + 1, (1,)).item()
            tokens = tokens[start:start + window]
        else:
            pad_amount = window - seq_len
            tokens = torch.cat([tokens, torch.full((pad_amount,), PAD, dtype=torch.long)])

        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stacks a list of (input_ids, target_ids) pairs into padded batch tensors and
    produces a padding mask (True where padded) for use in attention.
    """
    input_batch = torch.stack([item[0] for item in batch], dim=0)
    target_batch = torch.stack([item[1] for item in batch], dim=0)
    padding_mask = input_batch.eq(PAD)
    return input_batch, target_batch, padding_mask


def build_dataloaders(data_root: str) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and validation DataLoaders using the shared training
    configuration (batch size, etc.).
    """
    train_dataset = MusicSequenceDataset(data_root, split="train")
    val_dataset = MusicSequenceDataset(data_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        drop_last=False,
    )
    return train_loader, val_loader

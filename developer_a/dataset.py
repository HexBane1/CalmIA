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

import csv
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from shared.config import DATA_CONFIG, TRAIN_CONFIG, DISCRETE_CONDITION_CONFIG
from shared.vocabulary import (
    load_midi_as_tokens,
    PAD,
    BOS,
    EOS,
    tempo_condition_token,
    complexity_condition_token,
)


class MusicSequenceDataset(Dataset):
    """
    Loads tokenized symbolic music sequences from disk.

    Each item returned is a tuple (input_ids, target_ids) where target_ids is
    input_ids shifted by one position, standard for autoregressive next-token
    training.

    Task 2 (discrete condition tokens): if labels_csv_path is provided (the
    output of label_midi_features.py), every returned window has its
    corresponding tempo/complexity condition tokens prepended at position 0/1,
    and the model learns to associate that condition-token pair with the
    musical content that follows. If labels_csv_path is None (the default),
    this class behaves exactly as it did before Task 2 -- fully backward
    compatible with the Week 1 baseline.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        max_sequence_length: int = None,
        labels_csv_path: Optional[str] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.max_sequence_length = max_sequence_length or DATA_CONFIG.max_sequence_length
        self.file_paths = self._load_index()

        self.condition_lookup: Optional[Dict[str, Tuple[int, int]]] = None
        if labels_csv_path is not None:
            self.condition_lookup = self._load_condition_labels(labels_csv_path)
            # Only keep files that actually have a label. Files skipped by
            # label_midi_features.py (e.g. unparseable, no notes) would
            # otherwise crash __getitem__ with no condition tokens to prepend.
            before_count = len(self.file_paths)
            self.file_paths = [
                p for p in self.file_paths if os.path.normpath(p) in self.condition_lookup
            ]
            dropped = before_count - len(self.file_paths)
            if dropped:
                print(
                    f"MusicSequenceDataset({split}): dropped {dropped} file(s) with "
                    f"no entry in {labels_csv_path} (see label_midi_features.py output "
                    f"for why they were skipped)."
                )

    def _load_condition_labels(self, labels_csv_path: str) -> Dict[str, Tuple[int, int]]:
        """
        Loads label_midi_features.py's output CSV into a lookup from
        normalized file path -> (tempo_token_id, complexity_token_id).
        """
        lookup = {}
        with open(labels_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = os.path.normpath(row["file_path"])
                lookup[key] = (
                    tempo_condition_token(row["tempo_bin"]),
                    complexity_condition_token(row["complexity_bin"]),
                )
        if not lookup:
            raise RuntimeError(f"No labels found in {labels_csv_path} -- is the file empty?")
        return lookup

    def _load_index(self) -> List[str]:
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected split directory at {split_dir}. "
                f"TODO(new_dataset): update _load_index if the directory "
                f"layout differs."
            )
        valid_extensions = (".pt", ".mid", ".midi")
        # Recurse into subfolders (e.g. train/ambient/, train/classical/, ...)
        # so genre-labeled subdirectories are picked up automatically. Genre is
        # not used as a training signal in this baseline -- it is purely a
        # folder-organization convenience.
        file_paths = []
        for current_dir, _subdirs, filenames in os.walk(split_dir):
            for fname in filenames:
                if fname.lower().endswith(valid_extensions):
                    file_paths.append(os.path.join(current_dir, fname))
        return sorted(file_paths)

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
        path = self.file_paths[index]
        tokens = self._load_sequence(path)

        # Number of condition tokens to reserve room for at the front of every
        # window. Zero when conditioning is disabled -- window size and
        # cropping behavior then match the pre-Task-2 baseline exactly.
        num_condition_tokens = (
            DISCRETE_CONDITION_CONFIG.num_condition_tokens_prepended
            if self.condition_lookup is not None
            else 0
        )

        # Truncate or pad to a fixed window. For sequences longer than the max
        # length, take a random contiguous crop to expose the model to varied
        # positions across epochs rather than always training on the prefix.
        # The crop window is shrunk by num_condition_tokens so that, after the
        # condition tokens are prepended below, the TOTAL length is still
        # exactly max_sequence_length + 1 -- condition tokens land at position
        # 0/1 of every window regardless of where in the file the crop lands,
        # which is what lets the model learn the association from ANY part of
        # a piece, not just its true beginning.
        seq_len = tokens.size(0)
        window = self.max_sequence_length + 1 - num_condition_tokens

        if seq_len >= window:
            start = torch.randint(0, seq_len - window + 1, (1,)).item()
            tokens = tokens[start:start + window]
        else:
            pad_amount = window - seq_len
            tokens = torch.cat([tokens, torch.full((pad_amount,), PAD, dtype=torch.long)])

        if self.condition_lookup is not None:
            tempo_token_id, complexity_token_id = self.condition_lookup[os.path.normpath(path)]
            condition_prefix = torch.tensor([tempo_token_id, complexity_token_id], dtype=torch.long)
            tokens = torch.cat([condition_prefix, tokens])

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


def build_dataloaders(
    data_root: str, labels_csv_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and validation DataLoaders using the shared training
    configuration (batch size, etc.). Pass labels_csv_path (the output of
    label_midi_features.py) to enable Task 2 discrete condition-token
    conditioning for both splits; omit it to train the plain, unconditioned
    baseline exactly as before.
    """
    train_dataset = MusicSequenceDataset(data_root, split="train", labels_csv_path=labels_csv_path)
    val_dataset = MusicSequenceDataset(data_root, split="val", labels_csv_path=labels_csv_path)

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
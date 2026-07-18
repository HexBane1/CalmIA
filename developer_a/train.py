"""
Developer A work package: Training loop.

Trains MusicSequenceModel (developer_a/model.py) on tokenized sequences produced
by MusicSequenceDataset (developer_a/dataset.py). Tracks training and validation
loss, and saves checkpoints in the schema defined by shared.config.CHECKPOINT_SCHEMA
so that Developer B's loader can consume them without any coordination beyond that
shared schema.

Usage:
    python -m developer_a.train --data_root /path/to/roberts_dataset

TODO(robert-dataset): confirm the data_root path and directory layout expected by
MusicSequenceDataset before the first real run.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

from shared.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT_SCHEMA
from developer_a.dataset import build_dataloaders
from developer_a.model import MusicSequenceModel


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Cross-entropy loss over the vocabulary, ignoring padded positions so that
    padding does not artificially deflate the loss.
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for input_ids, target_ids, padding_mask in val_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        padding_mask = padding_mask.to(device)

        logits = model(input_ids, padding_mask=padding_mask)
        loss = compute_loss(logits, target_ids, DATA_CONFIG.pad_token_id)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        CHECKPOINT_SCHEMA.model_state_dict_key: model.state_dict(),
        CHECKPOINT_SCHEMA.optimizer_state_dict_key: optimizer.state_dict(),
        CHECKPOINT_SCHEMA.epoch_key: epoch,
        CHECKPOINT_SCHEMA.step_key: global_step,
        CHECKPOINT_SCHEMA.val_loss_key: val_loss,
        CHECKPOINT_SCHEMA.data_config_key: DATA_CONFIG,
        CHECKPOINT_SCHEMA.model_config_key: MODEL_CONFIG,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)

    return checkpoint_path


def train(data_root: str, checkpoint_dir: str = None, num_epochs: int = None) -> None:
    checkpoint_dir = checkpoint_dir or TRAIN_CONFIG.checkpoint_dir
    num_epochs = num_epochs or TRAIN_CONFIG.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(data_root)
    print(f"Train batches per epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = MusicSequenceModel().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {num_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG.learning_rate,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        running_loss = 0.0

        for batch_index, (input_ids, target_ids, padding_mask) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, padding_mask=padding_mask)
            loss = compute_loss(logits, target_ids, DATA_CONFIG.pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG.grad_clip_norm)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if batch_index % TRAIN_CONFIG.log_every_n_steps == 0:
                avg_loss = running_loss / batch_index
                print(
                    f"Epoch {epoch} | Step {batch_index}/{len(train_loader)} "
                    f"| Global step {global_step} | Train loss: {avg_loss:.4f}"
                )

        val_loss = evaluate(model, val_loader, device)
        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = running_loss / max(len(train_loader), 1)

        print(
            f"Epoch {epoch} complete in {epoch_duration:.1f}s | "
            f"Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, global_step, val_loss, checkpoint_dir, is_best
        )
        print(f"Saved checkpoint: {checkpoint_path}" + (" (new best)" if is_best else ""))


def main():
    parser = argparse.ArgumentParser(description="Train the Week 1 baseline music generation model.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Robert's dataset root.")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    train(data_root=args.data_root, checkpoint_dir=args.checkpoint_dir, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()

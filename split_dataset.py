"""
One-time dataset preparation script.

Takes a dataset organized as:
    <source_root>/
        ambient/*.mid
        classical/*.mid
        jazz/*.mid
        pop/*.mid
        soundtracks/*.mid

and produces the layout MusicSequenceDataset expects:
    <output_root>/
        train/
            ambient/...
            classical/...
            jazz/...
            pop/...
            soundtracks/...
        val/
            ambient/...
            classical/...
            jazz/...
            pop/...
            soundtracks/...

The split is stratified per genre (each genre is split ~90/10 independently),
so both train/ and val/ end up with a proportional mix of all five genres
rather than, for example, an entire genre landing only in val/.

This script only copies files -- it never deletes or modifies your original
dataset folders.

Usage:
    python3 split_dataset.py --source_root /path/to/dataset_with_genre_folders \
                              --output_root data/new_dataset \
                              --val_fraction 0.1
"""

import argparse
import os
import random
import shutil


VALID_EXTENSIONS = (".mid", ".midi")


def split_genre_folder(genre_dir: str, val_fraction: float, seed: int) -> tuple:
    files = sorted(
        f for f in os.listdir(genre_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    )
    rng = random.Random(seed)
    rng.shuffle(files)

    num_val = max(1, int(len(files) * val_fraction)) if files else 0
    val_files = files[:num_val]
    train_files = files[num_val:]
    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="Split genre-labeled MIDI folders into train/val.")
    parser.add_argument("--source_root", type=str, required=True,
                         help="Folder containing ambient/, classical/, jazz/, pop/, soundtracks/.")
    parser.add_argument("--output_root", type=str, required=True,
                         help="Destination folder; will contain train/ and val/ subfolders.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.source_root):
        raise FileNotFoundError(f"Source root not found: {args.source_root}")

    genre_dirs = [
        d for d in sorted(os.listdir(args.source_root))
        if os.path.isdir(os.path.join(args.source_root, d))
    ]
    if not genre_dirs:
        raise RuntimeError(f"No genre subfolders found inside {args.source_root}")

    total_train, total_val = 0, 0

    for genre in genre_dirs:
        genre_dir = os.path.join(args.source_root, genre)
        train_files, val_files = split_genre_folder(genre_dir, args.val_fraction, args.seed)

        train_dest = os.path.join(args.output_root, "train", genre)
        val_dest = os.path.join(args.output_root, "val", genre)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)

        for fname in train_files:
            shutil.copy2(os.path.join(genre_dir, fname), os.path.join(train_dest, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(genre_dir, fname), os.path.join(val_dest, fname))

        print(f"{genre:12s} -> train: {len(train_files):4d} | val: {len(val_files):4d}")
        total_train += len(train_files)
        total_val += len(val_files)

    print(f"\nTotal      -> train: {total_train:4d} | val: {total_val:4d}")
    print(f"\nDone. Point training at: --data_root {args.output_root}")


if __name__ == "__main__":
    main()
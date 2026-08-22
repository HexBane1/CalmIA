"""
Developer B work package: End-to-end example script.

Ties checkpoint_loader.py, sampler.py, and postprocess.py together into a single
command-line entry point for generating the first batch of songs. This is the
Week 1, Task 3 deliverable ("Generate our first batch of songs from this
baseline model").

Usage:
    python -m developer_b.generate_song \
        --checkpoint checkpoints/checkpoint_best.pt \
        --output_dir generated_songs \
        --num_songs 5

If --checkpoint is omitted, a dummy (randomly initialized) checkpoint is built
automatically so this script is runnable standalone before Developer A's first
trained checkpoint is available.
"""

import argparse
import os

from shared.config import DATA_CONFIG
from developer_b.checkpoint_loader import load_model_from_checkpoint, build_dummy_checkpoint
from developer_b.sampler import generate
from developer_b.postprocess import tokens_to_midi


def build_seed_primer() -> list:
    """
    Returns a minimal seed primer (just BOS) to start generation from silence.
    TODO(new_dataset): replace with a short tokenized phrase extracted from a
    real reference piece in the dataset once available, to seed generation
    with a more musically grounded opening.
    """
    return [DATA_CONFIG.bos_token_id]


def generate_batch(
    checkpoint_path: str,
    output_dir: str,
    num_songs: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> list:
    model, metadata = load_model_from_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from epoch {metadata['epoch']} (val_loss={metadata['val_loss']:.4f})")

    output_paths = []
    for song_index in range(1, num_songs + 1):
        primer = build_seed_primer()
        token_ids = generate(
            model,
            primer_ids=primer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        output_path = os.path.join(output_dir, f"baseline_song_{song_index:02d}.mid")
        tokens_to_midi(token_ids, output_path)
        output_paths.append(output_path)
        print(f"Generated song {song_index}/{num_songs}: {output_path} ({len(token_ids)} tokens)")

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Generate a batch of songs from the Week 1 baseline model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained checkpoint.")
    parser.add_argument("--output_dir", type=str, default="generated_songs")
    parser.add_argument("--num_songs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print("No --checkpoint provided. Building a dummy checkpoint for pipeline testing.")
        print("NOTE: output will be musically meaningless until a real trained checkpoint is used.")
        checkpoint_path = build_dummy_checkpoint("checkpoints/dummy_checkpoint.pt")

    generate_batch(
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        num_songs=args.num_songs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()

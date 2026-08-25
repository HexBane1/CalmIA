"""
midi_metrics.py -- Post-Hoc MIDI Quality Metrics (Phase 4 / Phase 5).

Computes: note_density, rhythm_regularity, repetition_rate, pitch_entropy.

Usage:
    python midi_metrics.py generated_songs_classical/ --output midi_metrics.csv
    python midi_metrics.py path/to/file.mid
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pretty_midi


def compute_metrics(midi_path: str) -> dict:
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return {
            "note_density": np.nan, "rhythm_regularity": np.nan,
            "repetition_rate": np.nan, "pitch_entropy": np.nan,
            "error": str(e),
        }

    all_notes = [note for inst in pm.instruments for note in inst.notes]
    if len(all_notes) < 2:
        return {
            "note_density": 0.0, "rhythm_regularity": np.nan,
            "repetition_rate": 0.0, "pitch_entropy": 0.0,
        }

    all_notes.sort(key=lambda n: n.start)
    onsets = np.array([n.start for n in all_notes])
    pitches = [n.pitch for n in all_notes]

    duration = pm.get_end_time()
    note_density = len(all_notes) / max(duration, 1e-6)

    iois = np.diff(onsets)
    iois = iois[iois > 1e-6]
    rhythm_regularity = float(np.std(iois) / (np.mean(iois) + 1e-9)) if len(iois) > 1 else np.nan

    ngram_size = 4
    if len(pitches) >= ngram_size:
        ngrams = [tuple(pitches[i:i + ngram_size]) for i in range(len(pitches) - ngram_size + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        repetition_rate = repeated / len(counts)
    else:
        repetition_rate = 0.0

    pitch_counts = Counter(pitches)
    total = sum(pitch_counts.values())
    probs = np.array([v / total for v in pitch_counts.values()])
    pitch_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    return {
        "note_density": float(note_density),
        "rhythm_regularity": float(rhythm_regularity),
        "repetition_rate": float(repetition_rate),
        "pitch_entropy": float(pitch_entropy),
    }


def scan_directory(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith((".mid", ".midi")):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root_dir)
                yield full, rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a .mid file or directory.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = []

    if os.path.isfile(args.path):
        metrics = compute_metrics(args.path)
        metrics["file"] = os.path.basename(args.path)
        rows.append(metrics)
    elif os.path.isdir(args.path):
        midi_files = list(scan_directory(args.path))
        if not midi_files:
            print(f"No .mid files found under {args.path}")
            sys.exit(1)
        print(f"Found {len(midi_files)} MIDI file(s)")
        for full_path, rel_path in midi_files:
            metrics = compute_metrics(full_path)
            metrics["file"] = rel_path
            rows.append(metrics)
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)

    df = pd.DataFrame(rows)
    col_order = ["file", "note_density", "rhythm_regularity", "repetition_rate", "pitch_entropy"]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]

    print("\n" + "=" * 70)
    print("MIDI QUALITY METRICS")
    print("=" * 70)
    print(df.to_string(index=False))

    if len(df) > 1:
        print("\nAggregate (mean):")
        print(df[["note_density", "rhythm_regularity",
                   "repetition_rate", "pitch_entropy"]].mean().to_string())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
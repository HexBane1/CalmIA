"""
Task 1: Label the training MIDI dataset with discrete tempo/complexity bins.

Scans a directory of MIDI files recursively -- so it works directly against the
train/val split already produced by split_dataset.py -- and computes two
continuous metrics per file:

    - Average tempo (BPM), via pretty_midi's estimate_tempo().
    - Musical complexity, as note onsets per second (drum tracks excluded, to
      stay consistent with how shared/vocabulary.py already treats drums).

These continuous values are then quantized into discrete bins via a median
split across the scanned dataset (TEMPO_SLOW/TEMPO_FAST, COMPLEXITY_LOW/
COMPLEXITY_HIGH). A median split is data-driven and avoids guessing absolute
BPM/density thresholds that may not suit this specific dataset.

IMPORTANT -- flag for Task 2: these bin boundaries are relative to THIS MIDI
dataset's own distribution. Before wiring this up alongside the WESAD-side
calculate_music_targets() output, the two sides need to agree on what
"TEMPO_SLOW" and "Target_Tempo=low" actually mean in comparable terms --
otherwise the model could learn a condition-token vocabulary that does not
line up with what the physiological mapper requests at inference time. The
raw continuous columns are kept in the output CSV specifically so re-binning
against different thresholds later does not require re-scanning every file.

Usage:
    python label_midi_features.py --data_root data/classical_split --output midi_labels.csv
"""

import argparse
import csv
import os
import statistics
from collections import Counter

import pretty_midi

VALID_EXTENSIONS = (".mid", ".midi")


def find_midi_files(data_root: str):
    file_paths = []
    for current_dir, _subdirs, filenames in os.walk(data_root):
        for fname in filenames:
            if fname.lower().endswith(VALID_EXTENSIONS):
                file_paths.append(os.path.join(current_dir, fname))
    return sorted(file_paths)


def compute_file_metrics(midi_path: str):
    """
    Returns (avg_tempo_bpm, complexity_notes_per_second) for a single MIDI
    file, or None if the file cannot be parsed or has no usable content.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as exc:
        print(f"SKIPPED (failed to parse): {midi_path} -> {exc}")
        return None

    try:
        avg_tempo_bpm = pm.estimate_tempo()
    except Exception:
        # estimate_tempo() can be unstable on files with very few notes.
        avg_tempo_bpm = None

    total_notes = sum(
        len(instrument.notes) for instrument in pm.instruments if not instrument.is_drum
    )
    duration_seconds = pm.get_end_time()

    if not avg_tempo_bpm or duration_seconds <= 0 or total_notes == 0:
        print(f"SKIPPED (no usable tempo/notes): {midi_path}")
        return None

    complexity_notes_per_second = total_notes / duration_seconds
    return avg_tempo_bpm, complexity_notes_per_second


def main():
    parser = argparse.ArgumentParser(description="Label MIDI files with discrete tempo/complexity bins.")
    parser.add_argument("--data_root", type=str, required=True,
                         help="Directory to scan recursively for .mid/.midi files, "
                              "e.g. data/classical_split (covers both train/ and val/).")
    parser.add_argument("--output", type=str, default="midi_labels.csv")
    args = parser.parse_args()

    midi_files = find_midi_files(args.data_root)
    print(f"Found {len(midi_files)} MIDI files under {args.data_root}")
    if not midi_files:
        raise RuntimeError(f"No .mid/.midi files found under {args.data_root}")

    results = []
    for path in midi_files:
        metrics = compute_file_metrics(path)
        if metrics is None:
            continue
        avg_tempo_bpm, complexity_notes_per_second = metrics
        results.append({
            "file_path": path,
            "avg_tempo_bpm": avg_tempo_bpm,
            "complexity_notes_per_second": complexity_notes_per_second,
        })

    if not results:
        raise RuntimeError("No files produced usable metrics -- nothing to label.")

    tempo_median = statistics.median(r["avg_tempo_bpm"] for r in results)
    complexity_median = statistics.median(r["complexity_notes_per_second"] for r in results)

    print(f"\nTempo median: {tempo_median:.1f} BPM")
    print(f"Complexity median: {complexity_median:.2f} notes/sec")

    for r in results:
        r["tempo_bin"] = "TEMPO_SLOW" if r["avg_tempo_bpm"] < tempo_median else "TEMPO_FAST"
        r["complexity_bin"] = (
            "COMPLEXITY_LOW" if r["complexity_notes_per_second"] < complexity_median else "COMPLEXITY_HIGH"
        )

    fieldnames = ["file_path", "avg_tempo_bpm", "complexity_notes_per_second", "tempo_bin", "complexity_bin"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLabeled {len(results)} files -> {args.output}")

    bin_counts = Counter((r["tempo_bin"], r["complexity_bin"]) for r in results)
    print("\nBin distribution:")
    for combo, count in sorted(bin_counts.items()):
        print(f"  {combo}: {count}")

    skipped = len(midi_files) - len(results)
    if skipped:
        print(f"\n{skipped} file(s) were skipped -- see SKIPPED lines above for reasons.")


if __name__ == "__main__":
    main()
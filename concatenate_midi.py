"""
concatenate_midi.py -- Concatenates per-step MIDI snippets into a single
continuous piece for listening/demo purposes.

Usage:
    python concatenate_midi.py --input_dir benchmark_output/arm5 \
        --output concatenated_arm5.mid
    python concatenate_midi.py --input_dir generated_songs_classical \
        --output concatenated_classical.mid
"""

import argparse
import os
import glob
import pretty_midi


def concatenate_midi_files(input_dir: str, output_path: str, gap_seconds: float = 0.5):
    """
    Concatenates all .mid files in input_dir (sorted alphabetically)
    into a single MIDI file. An optional gap_seconds silence is inserted
    between each snippet.
    """
    midi_files = sorted(glob.glob(os.path.join(input_dir, "*.mid")))
    if not midi_files:
        print(f"No .mid files found in {input_dir}")
        return

    print(f"Found {len(midi_files)} MIDI file(s) to concatenate...")

    combined = pretty_midi.PrettyMIDI()
    combined_instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    current_offset = 0.0

    for i, midi_path in enumerate(midi_files):
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"  Skipping {os.path.basename(midi_path)}: {e}")
            continue

        # Collect all notes from all instruments
        all_notes = []
        for instrument in pm.instruments:
            all_notes.extend(instrument.notes)

        if not all_notes:
            current_offset += gap_seconds
            continue

        # Find the start of the earliest note (may not be 0.0)
        min_start = min(n.start for n in all_notes)

        # Shift all notes to start at current_offset
        for note in all_notes:
            shifted = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=current_offset + (note.start - min_start),
                end=current_offset + (note.end - min_start),
            )
            combined_instrument.notes.append(shifted)

        # Advance offset by the duration of this snippet + gap
        snippet_duration = pm.get_end_time() - min_start
        current_offset += snippet_duration + gap_seconds

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(midi_files)} files...")

    combined.instruments.append(combined_instrument)
    combined.write(output_path)

    total_duration = current_offset
    print(f"\nDone! Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Concatenate MIDI snippets into one file.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .mid files to concatenate.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .mid file path.")
    parser.add_argument("--gap", type=float, default=0.5,
                        help="Silence gap between snippets in seconds (default: 0.5).")
    args = parser.parse_args()

    concatenate_midi_files(args.input_dir, args.output, gap_seconds=args.gap)


if __name__ == "__main__":
    main()
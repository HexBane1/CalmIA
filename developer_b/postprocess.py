"""
Developer B work package: Post-processing / synthesis.

Converts a generated token sequence (as produced by developer_b/sampler.py) back
into a playable MIDI file, using the event-based vocabulary defined in
shared/vocabulary.py. Requires `pretty_midi` (pip install pretty_midi).

This module depends only on shared/vocabulary.py and shared/config.py.
"""

import os
from typing import List

import pretty_midi

from shared.config import DATA_CONFIG
from shared.vocabulary import (
    is_note_on,
    is_note_off,
    is_time_shift,
    token_to_pitch,
    token_to_time_shift_bin,
    VELOCITY_OFFSET,
    NUM_VELOCITY_BINS,
    TEMPO_OFFSET,
    NUM_TEMPO_BINS,
)

SECONDS_PER_TIME_SHIFT_BIN = 0.03125  # must match shared.vocabulary.load_midi_as_tokens
DEFAULT_VELOCITY = 80
DEFAULT_TEMPO_BPM = 120.0


def tokens_to_midi(
    token_ids: List[int],
    output_path: str,
    instrument_program: int = 0,
    fallback_note_duration: float = 0.25,
) -> str:
    """
    Converts a list of token ids into a MIDI file and writes it to output_path.

    Notes that receive a NOTE_ON but never a matching NOTE_OFF before the
    sequence ends (a plausible artifact of an imperfectly trained baseline model)
    are closed out with `fallback_note_duration` so the output file remains valid
    and playable rather than silently dropping the note.

    Args:
        token_ids: generated sequence, including BOS/EOS/PAD tokens if present
            (these are skipped automatically).
        output_path: destination .mid file path.
        instrument_program: General MIDI program number for the single output
            instrument track (0 = Acoustic Grand Piano).
        fallback_note_duration: duration in seconds used to close any note that
            was opened but never explicitly closed.

    Returns:
        The output_path, for convenience chaining.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=DEFAULT_TEMPO_BPM)
    instrument = pretty_midi.Instrument(program=instrument_program)

    current_time = 0.0
    current_velocity = DEFAULT_VELOCITY
    open_notes = {}  # pitch -> (start_time, velocity)

    for token_id in token_ids:
        if token_id in (DATA_CONFIG.pad_token_id, DATA_CONFIG.bos_token_id, DATA_CONFIG.eos_token_id):
            continue

        if is_time_shift(token_id):
            bin_index = token_to_time_shift_bin(token_id)
            current_time += bin_index * SECONDS_PER_TIME_SHIFT_BIN

        elif VELOCITY_OFFSET <= token_id < VELOCITY_OFFSET + NUM_VELOCITY_BINS:
            velocity_bin = token_id - VELOCITY_OFFSET
            current_velocity = int(min(127, max(1, (velocity_bin + 1) / NUM_VELOCITY_BINS * 127)))

        elif TEMPO_OFFSET <= token_id < TEMPO_OFFSET + NUM_TEMPO_BINS:
            # Baseline model does not yet condition tempo dynamically; tempo
            # tokens are parsed but a single global tempo is used for the
            # Week 1 baseline output. Per-segment tempo changes are a natural
            # extension point for the Safe RL controller in later weeks.
            continue

        elif is_note_on(token_id):
            pitch = token_to_pitch(token_id)
            # If a note-on arrives for a pitch that is already open (model
            # artifact), close the previous instance first so we do not lose it.
            if pitch in open_notes:
                start_time, velocity = open_notes.pop(pitch)
                instrument.notes.append(
                    pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=current_time)
                )
            open_notes[pitch] = (current_time, current_velocity)

        elif is_note_off(token_id):
            pitch = token_to_pitch(token_id)
            if pitch in open_notes:
                start_time, velocity = open_notes.pop(pitch)
                end_time = max(current_time, start_time + 1e-3)
                instrument.notes.append(
                    pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                )
            # A NOTE_OFF with no matching open NOTE_ON is a model artifact and is
            # silently ignored rather than raising, to keep generation robust.

    # Close out any notes still open at the end of the sequence.
    for pitch, (start_time, velocity) in open_notes.items():
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=start_time + fallback_note_duration,
            )
        )

    pm.instruments.append(instrument)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pm.write(output_path)
    return output_path


def tokens_to_wav(midi_path: str, wav_output_path: str, sample_rate: int = 44100) -> str:
    """
    Optional convenience step: synthesizes a .mid file to .wav using pretty_midi's
    built-in fluidsynth-based synthesizer, if fluidsynth is installed on the
    system. Falls back with a clear error message if it is not available, since
    this is an optional listening-convenience step rather than a required
    deliverable (the .mid file alone is sufficient for Week 1).
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.synthesize(fs=sample_rate)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "MIDI-to-WAV synthesis requires fluidsynth and a sound font to be "
            "available in this environment. The .mid file was still generated "
            "successfully and can be played in any standard MIDI player or DAW."
        ) from exc

    import numpy as np
    import scipy.io.wavfile as wavfile

    os.makedirs(os.path.dirname(wav_output_path) or ".", exist_ok=True)
    normalized_audio = np.int16(audio / max(np.max(np.abs(audio)), 1e-8) * 32767)
    wavfile.write(wav_output_path, sample_rate, normalized_audio)
    return wav_output_path

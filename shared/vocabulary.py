"""
Shared token vocabulary and MIDI <-> token conversion utilities.

This module defines an event-based (REMI-style) vocabulary. It was selected over
piano-roll or raw-audio representations based on the Week 1 Kaggle survey criterion
(Point 1 in README.md): event tokens expose tempo, velocity, and timing directly as
discrete tokens, which is the representation most compatible with a downstream
physiology-conditioned controller (Week 2+ work).

TODO(robert-dataset): if Robert's dataset ships as pre-tokenized tensors rather than
raw MIDI files, only `load_midi_as_tokens` needs to change. The vocabulary layout and
VOCAB_SIZE below can stay fixed as long as Robert's token ids follow this schema, or
you add a small remapping function here.
"""

from typing import List

# ---------------------------------------------------------------------------
# Vocabulary layout (event-based / REMI-style)
# ---------------------------------------------------------------------------
# Reserved special tokens: ids 0-2 (must match shared.config.DataConfig)
PAD, BOS, EOS = 0, 1, 2

# Pitch events: 128 MIDI pitches -> NOTE_ON_<pitch>, ids 3..130
NOTE_ON_OFFSET = 3
NUM_PITCHES = 128

# NOTE_OFF events for the same 128 pitches, ids 131..258
NOTE_OFF_OFFSET = NOTE_ON_OFFSET + NUM_PITCHES

# Time-shift bins (quantized to 1/32 note up to 4 bars), ids 259..386
TIME_SHIFT_OFFSET = NOTE_OFF_OFFSET + NUM_PITCHES
NUM_TIME_SHIFT_BINS = 128

# Velocity bins (quantized to 32 levels), ids 387..418
VELOCITY_OFFSET = TIME_SHIFT_OFFSET + NUM_TIME_SHIFT_BINS
NUM_VELOCITY_BINS = 32

# Tempo bins (quantized, e.g. 30-250 BPM in coarse steps), ids 419..450
TEMPO_OFFSET = VELOCITY_OFFSET + NUM_VELOCITY_BINS
NUM_TEMPO_BINS = 32

# Remaining ids reserved for future conditioning tokens (Week 2+: HRV/EDA bins, etc.)
VOCAB_SIZE = 512  # must match shared.config.DataConfig.vocab_size

# ---------------------------------------------------------------------------
# Discrete condition tokens (Task 2: physiology-informed music conditioning)
# ---------------------------------------------------------------------------
# Prepended to the start of every training window (see developer_a/dataset.py)
# so the model learns to associate a condition-token pair with the musical
# content that follows it. These are ordinary vocabulary tokens -- embedding
# them requires no architecture change, since nn.Embedding already covers the
# full VOCAB_SIZE range and the Transformer treats any token id identically.
#
# NOTE: this is a *different* concept from the per-event TEMPO_* bins above
# (ids TEMPO_OFFSET..TEMPO_OFFSET+31), which represent a tempo value occurring
# at a specific point within a piece and are currently unused by
# load_midi_as_tokens. These new tokens instead represent a whole-sequence
# style REQUEST, decided once per training example -- kept in a separate id
# range so the two concepts are never confused.
CONDITION_TOKEN_OFFSET = TEMPO_OFFSET + NUM_TEMPO_BINS  # = 451
TEMPO_SLOW = CONDITION_TOKEN_OFFSET + 0        # 451
TEMPO_FAST = CONDITION_TOKEN_OFFSET + 1        # 452
COMPLEXITY_LOW = CONDITION_TOKEN_OFFSET + 2    # 453
COMPLEXITY_HIGH = CONDITION_TOKEN_OFFSET + 3   # 454
NUM_CONDITION_TOKEN_IDS = 4

CONDITION_TOKEN_NAMES = {
    TEMPO_SLOW: "TEMPO_SLOW",
    TEMPO_FAST: "TEMPO_FAST",
    COMPLEXITY_LOW: "COMPLEXITY_LOW",
    COMPLEXITY_HIGH: "COMPLEXITY_HIGH",
}

_TEMPO_BIN_TO_TOKEN = {"TEMPO_SLOW": TEMPO_SLOW, "TEMPO_FAST": TEMPO_FAST}
_COMPLEXITY_BIN_TO_TOKEN = {"COMPLEXITY_LOW": COMPLEXITY_LOW, "COMPLEXITY_HIGH": COMPLEXITY_HIGH}


def tempo_condition_token(tempo_bin: str) -> int:
    """Maps a label_midi_features.py 'tempo_bin' string to its token id."""
    if tempo_bin not in _TEMPO_BIN_TO_TOKEN:
        raise ValueError(f"Unknown tempo_bin: {tempo_bin!r}")
    return _TEMPO_BIN_TO_TOKEN[tempo_bin]


def complexity_condition_token(complexity_bin: str) -> int:
    """Maps a label_midi_features.py 'complexity_bin' string to its token id."""
    if complexity_bin not in _COMPLEXITY_BIN_TO_TOKEN:
        raise ValueError(f"Unknown complexity_bin: {complexity_bin!r}")
    return _COMPLEXITY_BIN_TO_TOKEN[complexity_bin]


def is_condition_token(token_id: int) -> bool:
    return token_id in CONDITION_TOKEN_NAMES


def note_on_token(pitch: int) -> int:
    assert 0 <= pitch < NUM_PITCHES, f"pitch out of range: {pitch}"
    return NOTE_ON_OFFSET + pitch


def note_off_token(pitch: int) -> int:
    assert 0 <= pitch < NUM_PITCHES, f"pitch out of range: {pitch}"
    return NOTE_OFF_OFFSET + pitch


def time_shift_token(bin_index: int) -> int:
    assert 0 <= bin_index < NUM_TIME_SHIFT_BINS, f"time shift bin out of range: {bin_index}"
    return TIME_SHIFT_OFFSET + bin_index


def velocity_token(bin_index: int) -> int:
    assert 0 <= bin_index < NUM_VELOCITY_BINS, f"velocity bin out of range: {bin_index}"
    return VELOCITY_OFFSET + bin_index


def tempo_token(bin_index: int) -> int:
    assert 0 <= bin_index < NUM_TEMPO_BINS, f"tempo bin out of range: {bin_index}"
    return TEMPO_OFFSET + bin_index


def is_note_on(token_id: int) -> bool:
    return NOTE_ON_OFFSET <= token_id < NOTE_ON_OFFSET + NUM_PITCHES


def is_note_off(token_id: int) -> bool:
    return NOTE_OFF_OFFSET <= token_id < NOTE_OFF_OFFSET + NUM_PITCHES


def is_time_shift(token_id: int) -> bool:
    return TIME_SHIFT_OFFSET <= token_id < TIME_SHIFT_OFFSET + NUM_TIME_SHIFT_BINS


def token_to_pitch(token_id: int) -> int:
    """Recovers the MIDI pitch from a NOTE_ON or NOTE_OFF token id."""
    if is_note_on(token_id):
        return token_id - NOTE_ON_OFFSET
    if is_note_off(token_id):
        return token_id - NOTE_OFF_OFFSET
    raise ValueError(f"Token {token_id} is not a note on/off token.")


def token_to_time_shift_bin(token_id: int) -> int:
    assert is_time_shift(token_id), f"Token {token_id} is not a time-shift token."
    return token_id - TIME_SHIFT_OFFSET


def load_midi_as_tokens(midi_path: str) -> List[int]:
    """
    Converts a MIDI file into a list of token ids using the event-based vocabulary
    above. Requires `pretty_midi` (pip install pretty_midi).

    TODO(robert-dataset): confirm whether Robert's files are single-track or
    multi-track MIDI. This implementation flattens all instruments into a single
    monophonic-friendly event stream, sorted by onset time; extend to a multi-track
    interleaving scheme if Robert's dataset requires preserving separate instruments.
    """
    import pretty_midi  # local import: keeps this an optional dependency

    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []  # list of (time, token_id) prior to time-shift insertion

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            velocity_bin = min(int(note.velocity / 128 * NUM_VELOCITY_BINS), NUM_VELOCITY_BINS - 1)
            events.append((note.start, velocity_token(velocity_bin)))
            events.append((note.start, note_on_token(note.pitch)))
            events.append((note.end, note_off_token(note.pitch)))

    events.sort(key=lambda e: e[0])

    tokens = [BOS]
    prev_time = 0.0
    seconds_per_bin = 0.03125  # ~1/32 note at 120 BPM as a coarse quantization step
    for event_time, token_id in events:
        delta = event_time - prev_time
        if delta > 1e-6:
            bin_index = min(int(delta / seconds_per_bin), NUM_TIME_SHIFT_BINS - 1)
            tokens.append(time_shift_token(bin_index))
        tokens.append(token_id)
        prev_time = event_time
    tokens.append(EOS)
    return tokens
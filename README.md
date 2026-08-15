# Week 1 — Physiology-Conditioned Generative Music: Baseline Establishment

## Status Summary (read this first)

**Where things stand:** the Week 1 baseline is functionally complete and pushed to
GitHub. Here's the play-by-play so we're both on the same page:

- Built the full codebase split around a shared contract (`shared/config.py`,
  `shared/vocabulary.py`) so we can work in parallel without blocking each other —
  see "Parallelization Contract" below for exactly how that's meant to work.
- Wired in our actual dataset: 5 genre folders (`ambient`, `classical`, `jazz`, `pop`,
  `soundtracks`), each a flat folder of raw `.mid` files. `split_dataset.py` (new,
  in the repo root) takes that layout and produces a stratified 90/10 train/val split
  per genre into `data/new_dataset/`. `dataset.py` recurses into genre subfolders
  automatically, so genre is just a folder-organization convenience right now — it is
  **not** used as a conditioning signal in this baseline.
- Trained the baseline Transformer (`developer_a/model.py`) for 20 epochs on the real
  dataset. Best checkpoint by validation loss is saved as `checkpoints/checkpoint_best.pt`
  and is in the repo.
- Generated our first batch of songs (`developer_b/generate_song.py`) —
  `generated_songs/baseline_song_01.mid` through `_05.mid`, also in the repo.
- **Honest quality check:** the 5 generated songs are rough — closer to noise than
  music. This is expected, not a bug: a ~3.3M-parameter model trained for 20 epochs on
  a genre-mixed dataset this size was never going to produce something polished. The
  point of this checkpoint was proving the pipeline works end-to-end (data in, model
  trains, music comes out), which it now does. Quality is the next lever to pull, not
  something that needed fixing before this deadline.

**What's actually left before we move on:**
1. **The 10 Kaggle notebook analysis** (see Section 1 below) — this hasn't been
   started yet and is the other explicit "until next time" deliverable. Let's split
   5 notebooks each.
2. Everything else in this README (novel architecture, physiological conditioning,
   Safe RL controller, the full RCT evaluation suite) is final-report / later-weeks
   scope per the professor's requirements doc — **not** due now, so don't feel
   pressure to start on that yet.

If we want to improve generation quality later (lower temperature at sampling time,
train longer, or split training per-genre instead of pooling all 5 together), that's
a cheap, optional next step — not a blocker for this checkpoint.

---

This package covers the three Week 1 objectives:

1. A structured framework for analyzing the 10 Kaggle reference notebooks.
2. Developer A's work package: data pipeline + baseline training routine.
3. Developer B's work package: inference, sampling, and MIDI synthesis.

The codebase is deliberately split around a **shared contract** (`shared/config.py` and
`shared/vocabulary.py`). Both developers import from this contract only — neither imports
from the other's package directly. This is what allows fully parallel work:

- Developer A can change model internals freely as long as the checkpoint format
  (defined in `shared/config.py`) stays fixed.
- Developer B can build and unit-test the entire inference/sampling/synthesis pipeline
  today, against a randomly initialized model, without waiting for a trained checkpoint.

---

## 1. Kaggle Notebook Analysis Framework

**Status: not started — this is the open item.**

Apply this 5-point rubric identically to all 10 notebooks. Time-box each notebook to
30–40 minutes. The goal is pattern extraction, not reproduction — you are mining for
decisions you can reuse or explicitly reject, with a reason.

### Point 1 — Data Representation / Tokenization Strategy
Identify which of these each notebook uses, and note the trade-off it implies for your
physiological-conditioning use case (you need fine-grained, low-latency control over
tempo and tension, which some representations support poorly):

- **Raw audio (waveform / spectrogram)**: high fidelity, expensive, poor direct control
  over discrete musical parameters like tempo or harmonic tension.
- **Piano-roll matrices** (fixed time-step x pitch grid): simple, fixed quantization,
  good for CNN/RNN baselines, weak on dynamics/velocity and long-range structure.
- **Event-based / REMI-style tokens** (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`, `VELOCITY`,
  `TEMPO`, `BAR`): variable length, compact, directly exposes tempo/velocity as tokens
  you can later condition on — closest fit to a physiology-conditioned controller. This
  is what our baseline already uses (`shared/vocabulary.py`).
- **MIDI-derived symbolic sequences with explicit metadata channels** (chord/tempo/
  instrument tracks parsed via `pretty_midi` or `music21`): good middle ground.

**What to record per notebook**: tokenization scheme name, vocabulary size, max
sequence length used, and whether tempo/dynamics are explicit tokens or implicit in
timing.

### Point 2 — Architecture Pattern
Classify the model family and note training cost vs. quality trade-offs reported:

- RNN/LSTM baselines (fast to train, weaker long-range coherence).
- Transformer decoder / GPT-style autoregressive models (better long-range structure,
  higher compute cost, need relative or learned positional encodings for music).
- VAE / VQ-VAE + prior (useful for latent-space conditioning — relevant later for your
  physiology-conditioned latent, worth flagging even if not chosen for baseline).
- GAN-based (rare for symbolic music, usually raw-audio; note if present).

**What to record**: architecture family, layer count/hidden size if reported,
context window length, any reported training time/hardware.

### Point 3 — Conditioning Mechanism (if any)
Even though most Kaggle baselines are unconditional, note any that condition on genre,
composer, or mood tokens, and *how* they inject the conditioning signal:

- Prepended control token(s) in the input sequence.
- Concatenated conditioning embedding added to token embeddings.
- Cross-attention from a separate conditioning encoder.
- FiLM-style feature-wise modulation of hidden activations.

This directly informs how you will later inject HRV/EDA/respiration features into the
architecture — treat this as reconnaissance for your actual research contribution.

### Point 4 — Evaluation Metrics Used
Record every metric so you can pick a comparable set for your baseline report:

- Loss-based: validation cross-entropy / perplexity.
- Music-theoretic: pitch-class histogram entropy, note density, empty-bar rate,
  scale/key consistency, groove/rhythm consistency (autocorrelation of onsets).
- Human/qualitative: listening test descriptions (even informal ones — note sample size
  and bias, since Kaggle listening tests are rarely rigorous).
- Diversity: unique n-gram rate, self-similarity matrices (to catch repetitive-loop
  failure modes, directly relevant to Developer B's sampling design, and directly
  relevant to why our own 5 generated songs sound rough).

### Point 5 — Reproducibility & Failure Modes
- Is there a public repo, fixed random seed, and pinned dependency versions?
- What failure modes does the author admit to (looping, silence collapse, key drift,
  mode collapse)? This is often the most valuable section — it tells you what NOT to
  spend Week 2+ debugging from scratch.
- Dataset size and preprocessing time reported, so you can sanity-check our own
  dataset against a known-working scale.

### Deliverable
A single shared spreadsheet/table, one row per notebook, one column per point above,
plus a final "Adopt / Reject / Adapt" column with a one-line justification. This
directly feeds the architectural decisions already made in `developer_a/model.py` below
— review it and edit `shared/config.py` if the notebook survey suggests a different
tokenization or context length.

---

## 2. Directory Layout

```
week1_baseline/
  shared/
    config.py         # Single source of truth: vocab size, seq len, model dims,
                       # checkpoint schema. BOTH developers import this, never each other.
    vocabulary.py      # Token <-> id mapping and MIDI parsing.
  developer_a/
    dataset.py         # PyTorch Dataset/DataLoader for symbolic music sequences.
                        # Recurses into genre subfolders automatically.
    model.py           # Baseline lightweight Transformer (LSTM variant included as
                        # a commented-out drop-in alternative).
    train.py           # Training loop, validation, checkpointing.
  developer_b/
    checkpoint_loader.py  # Loads shared config + Dev A's checkpoint, no Dev A import.
    sampler.py             # Temperature / top-k / top-p sampling with repetition guard.
    postprocess.py          # Token sequence -> MIDI file via pretty_midi.
    generate_song.py        # End-to-end script tying the above together.
  split_dataset.py     # One-time script: splits genre folders into train/val.
  dataset/              # Raw MIDI, organized by genre (not committed to git).
  data/new_dataset/     # Output of split_dataset.py: train/ and val/, each with
                        # genre subfolders (not committed to git — regeneratable).
  checkpoints/          # Trained model checkpoints. checkpoint_best.pt is committed
                        # as evidence; the rest are gitignored.
  generated_songs/      # baseline_song_01.mid through _05.mid, committed as evidence.
```

## 3. Parallelization Contract

- Freeze `shared/config.py` and `shared/vocabulary.py` first — this is the interface.
  Any change to vocab size, PAD/BOS/EOS ids, or checkpoint keys must be a discussed,
  versioned change, not a silent edit.
- Developer A owns `developer_a/`. Developer B owns `developer_b/`. Neither imports
  from the other's folder.
- Developer B validates against a dummy checkpoint (instructions in
  `developer_b/checkpoint_loader.py` docstring) so the inference pipeline is fully
  tested before Developer A's first real checkpoint lands.
- Integration point: `shared/config.CHECKPOINT_SCHEMA` and the `MusicSequenceModel`
  class name/signature. As long as Developer A does not rename the class or change its
  `forward()`/`generate_step()` signature, Developer B's code needs zero changes when
  the real checkpoint arrives.

## 4. Adapting to the Dataset (done)

The dataset ships as raw MIDI files organized in 5 genre folders (`ambient`,
`classical`, `jazz`, `pop`, `soundtracks`), each a flat folder of `.mid` files. To
reproduce the train/val split from scratch:

```bash
python3 split_dataset.py --source_root dataset --output_root data/new_dataset --val_fraction 0.1
python3 -m developer_a.train --data_root data/new_dataset
python3 -m developer_b.generate_song --checkpoint checkpoints/checkpoint_best.pt --output_dir generated_songs --num_songs 5
```

Any remaining placeholders for dataset-specific adaptation are marked with
`# TODO(new_dataset): ...` in the code — none are currently blocking, since the
pipeline above has already been run successfully end-to-end.

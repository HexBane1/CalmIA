# Week 1 — Physiology-Conditioned Generative Music: Baseline Establishment

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
  you can later condition on — closest fit to a physiology-conditioned controller.
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
  failure modes, directly relevant to Developer B's sampling design).

### Point 5 — Reproducibility & Failure Modes
- Is there a public repo, fixed random seed, and pinned dependency versions?
- What failure modes does the author admit to (looping, silence collapse, key drift,
  mode collapse)? This is often the most valuable section — it tells you what NOT to
  spend Week 2+ debugging from scratch.
- Dataset size and preprocessing time reported, so you can sanity-check Robert's
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
    vocabulary.py      # Token <-> id mapping and MIDI parsing placeholder.
  developer_a/
    dataset.py         # PyTorch Dataset/DataLoader for symbolic music sequences.
    model.py           # Baseline lightweight Transformer (LSTM variant included as
                       # a commented-out drop-in alternative).
    train.py           # Training loop, validation, checkpointing.
  developer_b/
    checkpoint_loader.py  # Loads shared config + Dev A's checkpoint, no Dev A import.
    sampler.py             # Temperature / top-k / top-p sampling with repetition guard.
    postprocess.py          # Token sequence -> MIDI file via pretty_midi.
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

## 4. Adapting to Robert's Dataset

Every placeholder for Robert's dataset specifics is marked with:
`# TODO(robert-dataset): ...`
Search for that tag across the codebase before running anything. The two most likely
adaptation points are `vocabulary.py` (if Robert's data is raw MIDI files vs. a
pre-tokenized format) and `dataset.py`'s `_load_index()` method (file layout).

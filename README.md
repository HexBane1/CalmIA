# Week 1 — Physiology-Conditioned Generative Music: Baseline Establishment

## Status Summary (read this first)

**Where things stand:** the baseline, the discrete condition-token conditioning pipeline, and the core Safe RL controller are all implemented and pushed to GitHub. Here is the summary of what has been implemented:

- Built the codebase split around a shared contract (`shared/config.py`, `shared/vocabulary.py`) for parallel development.
- Initial multi-genre baseline: Trained on 5 genres (`ambient`, `classical`, `jazz`, `pop`, `soundtracks`) using `split_dataset.py` to produce `data/new_dataset/`. Generated baseline songs are in `generated_songs/`.
- Classical baseline: Curated a single-genre classical dataset (`classical_dataset/`) split into `data/classical_split/` to reduce genre-confusion. Retrained the model and generated the classical baseline batch in `generated_songs_classical/`.
- **Discrete condition-token conditioning:** `label_midi_features.py` scans the classical MIDI dataset and quantizes each file's average tempo and note density into discrete bins (`TEMPO_SLOW`/`TEMPO_FAST`, `COMPLEXITY_LOW`/`COMPLEXITY_HIGH`), output as `midi_labels.csv`. The vocabulary, dataset pipeline, and training script were extended so these tokens are prepended to every training window — **no architecture change was required**, since condition tokens are ordinary vocabulary entries the existing Transformer already handles. Retrained and confirmed the model generates audibly different output per condition (`generated_songs_conditioned/`).
- Model checkpoints: Best weights are tracked in `checkpoints/checkpoint_best.pt`.
- Reference review: Completed the 10 Kaggle reference notebook analysis (`kaggle_analysis.xlsx`).
- Physiological data pipeline: Verified WESAD file integrity (`check_wesad.py`) and verified extraction of HRV (SDNN) and EDA metrics (`extract_features.py`).
- Real-time windowing & target mapping: `extract_features.py` produces a windowed, multi-condition physiological timeline. Originally validated on a single subject (`S2_physiological_timeline.csv`, kept for reference — see Section 4), then **scaled to loop across all available subjects (S2–S17, S12 excluded — known sensor malfunction in the official WESAD release)**, aggregating into a unified `wesad_physiological_timeline.csv`.
- **Safe RL Controller (Phase 2 core):** `rl_env.py` implements a custom `gymnasium.Env` (`PatientPhysiologyEnv`) treating the physiological timeline as an offline-simulated patient — one episode per subject, state = normalized (HRV_SDNN, Mean_EDA), action = a 4-way discrete tempo/complexity choice, reward = a therapeutic bonus for calming choices under high stress plus a flat safety penalty (not stacked) for abrupt or unsafe transitions. `train_rl.py` trains a PPO agent (`stable-baselines3`) on this environment, saving to `checkpoints/rl_controller.zip`. `closed_loop_generate.py` is the full integration: it steps a trained agent through a subject's timeline, converts each chosen action into condition tokens, and generates a matching MIDI snippet per step into `closed_loop_output/`, logging every step to `manifest.csv`.

### Known open items (read before extending this work)

- **Condition-scheme mismatch, not yet reconciled:** `label_midi_features.py`'s MIDI-side bins are a 2-tier scheme (`SLOW`/`FAST`, `LOW`/`HIGH`), while `extract_features.py`'s `calculate_music_targets()` outputs a *3-tier numeric* scheme (`Target_Tempo` in {70, 90, 110}, `Target_Complexity` in {0.2, 0.5, 0.8}). `rl_env.py`'s action space uses its own independent 4-way discrete mapping rather than either of these directly. All three are internally consistent on their own, but the three-way relationship between them hasn't been formally reconciled — worth resolving explicitly before treating any one of them as the sole source of truth downstream.
- **Threshold calibration is still first-pass.** Both `calculate_music_targets()` (HRV_SDNN < 50ms, EDA > 2.0µS) and the RL environment's high-stress detection (same thresholds, reused for consistency) are provisional values, not clinically validated or calibrated against the full aggregated population. `label_midi_features.py`'s bins, by contrast, already use a data-driven median split rather than a fixed threshold — the same approach is worth applying to the physiological side once the full multi-subject dataset is stable.
- **Phase 2's action space is simpler than the original roadmap envisioned.** The roadmap below originally called for controlling "tempo delta, harmonic tension, rhythmic complexity" as separate, more granular levers. What's actually implemented is a single 4-way discrete choice over (tempo, complexity) jointly — a deliberate simplification to get a working closed loop end-to-end first, not an oversight, but worth being explicit about rather than letting the roadmap imply more granularity than currently exists.

## Collaborator Setup & Requirements

**1. Local Environment & Dependencies**
Model checkpoints and datasets are intentionally excluded from version control to save space. To set up your local environment:
* Pull the latest changes from the `main` branch.
* Install required libraries: `pip install -r requirements.txt`
* The physiological and RL pipeline additionally requires: `pip install neurokit2 gymnasium stable-baselines3`

**2. Downloading the Datasets**
You must download the datasets locally and place them in the project root.

*   **WESAD Dataset (~3GB):**
    1. Download from [here:](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset).
    2. Extract the contents into a folder named `wesad_data/` in the project root. 
    3. Verify the file path matches this format: `wesad_data/S2/S2.pkl`.

*   **Classical MIDI Dataset:**
    1. Download from [here:](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi).
    2. Extract the MIDI files into a folder named `classical_dataset/` in the project root.

**3. Generating the Train/Val Split**
Do not manually split the MIDI files. Generate the local splits by running the preprocessing script:
`python split_dataset.py --source_root classical_dataset --output_root data/classical_split --val_fraction 0.1`

**4. Real-Time Windowing & Target Mapping**

**Status: Completed**, then scaled beyond its original single-subject scope (see Section 6 below for the multi-subject version). Original implementation notes, preserved for reference:

- Implemented `get_windows_for_condition()`: slices raw signal into consecutive
  10-second windows (7000 samples at 700Hz), filtered per condition label.
- Extended feature extraction to cover Baseline (1), Stress (2), and Meditation (4),
  not just Stress.
- Implemented `calculate_music_targets(hrv, eda)`: rule-based mapping using
  HRV_SDNN < 50ms and EDA > 2.0 microsiemens as high-arousal thresholds, producing
  lower target tempo/complexity under high stress (safety-first: calm rather than
  excite). Thresholds are a first-pass baseline, not clinically validated.
- Fixed an HRV computation issue: switched from `nk.hrv()` (all HRV domains) to
  `nk.hrv_time()` (time-domain only), since the full computation failed on most
  10s windows (especially Meditation, where low variability broke the
  frequency-domain calculations). This raised valid-window coverage from 16% to 100%.
- Output: `S2_physiological_timeline.csv`, 251 rows (114 Baseline / 61 Stress /
  76 Meditation), columns: `Timestamp, Condition_Label, HRV_SDNN, Mean_EDA,
  Target_Tempo, Target_Complexity`.
- Sanity check: mean HRV_SDNN is lowest under Stress (49.1ms) and highest under
  Meditation (59.3ms), consistent with expected physiology.

**5. MIDI Tempo/Complexity Labeling & Condition-Token Training**

**Status: Completed.** Bridges the classical MIDI dataset and the physiological pipeline by giving the Transformer a vocabulary of style-request tokens it can be conditioned on.

- `label_midi_features.py` scans `data/classical_split` (or any directory recursively), computing per-file average tempo (`pretty_midi.estimate_tempo()`) and note density (onsets/second, drum tracks excluded). Values are quantized via a **median split across the scanned dataset** into `TEMPO_SLOW`/`TEMPO_FAST` and `COMPLEXITY_LOW`/`COMPLEXITY_HIGH`, written to `midi_labels.csv` alongside the raw continuous values (so re-binning against different thresholds later doesn't require re-scanning every file).
- `shared/vocabulary.py` gained 4 new token ids (451-454) for these bins, in a range distinct from the pre-existing (currently unused) per-event tempo bins.
- `developer_a/dataset.py` optionally prepends the matching tempo+complexity token pair to **every training window**, not just the start of the raw file — the crop window is shrunk by 2 tokens so the condition tokens land at position 0/1 regardless of where a random crop lands in a long piece.
- `developer_a/train.py` gained a `--labels_csv` flag to enable this; omitting it trains the original unconditioned baseline unchanged.
- `developer_a/model.py` required **no changes** — condition tokens are embedded exactly like any other vocabulary token.

To reproduce:
```bash
python label_midi_features.py --data_root data/classical_split --output midi_labels.csv
python -m developer_a.train --data_root data/classical_split --labels_csv midi_labels.csv
```

**6. Safe RL Controller**

**Status: Core implementation completed**, pending the calibration/reconciliation work noted above.

- `rl_env.py` — `PatientPhysiologyEnv`, a custom `gymnasium.Env`. One episode = one subject's row sequence from `wesad_physiological_timeline.csv`. State: normalized (HRV_SDNN, Mean_EDA). Action: `Discrete(4)` — `0=Slow/Low, 1=Slow/High, 2=Fast/Low, 3=Fast/High`. Reward: `+1` for choosing Slow/Low under high stress; a flat `-5` safety penalty (applied once per step even if multiple rules trigger) for an abrupt single-step jump between maximally-opposite actions, a 3-step oscillation, or choosing Fast/High specifically under high stress. Verified against `stable-baselines3`'s `check_env`.
- `train_rl.py` — trains PPO on `PatientPhysiologyEnv` (wrapped in `DummyVecEnv` + `Monitor`), with periodic checkpointing during training, saving the final policy to `checkpoints/rl_controller.zip`. Subject selection is intentionally randomized per episode during training (not pinned), so the policy generalizes across the population rather than memorizing one patient.
- `closed_loop_generate.py` — the full integration. Loads a specific subject's timeline, the trained RL agent, and the trained conditioned Transformer checkpoint; steps through the timeline; converts each chosen action into `[tempo_token_id, complexity_token_id]`; generates a MIDI snippet per step via the existing `developer_b` sampler/postprocess modules; writes numbered `.mid` files plus `manifest.csv` (per-step HRV/EDA, action, and output filename) into `closed_loop_output/`.

To reproduce:
```bash
python train_rl.py --csv_path wesad_physiological_timeline.csv --timesteps 75000
python closed_loop_generate.py --subject S2 --physio_csv wesad_physiological_timeline.csv --rl_checkpoint checkpoints/rl_controller.zip --music_checkpoint checkpoints/checkpoint_best.pt --output_dir closed_loop_output
```

---

## Project Completion Roadmap & Technical Requirements

The following phases outline the remaining deliverables required for the final research prototype and evaluation suite:

### Phase 1: Conditioning & Feature Ingestion — Status: Completed
* **Multimodal Integration:** Ingest extracted windowed features (HRV, EDA) alongside tokenized MIDI sequences. *(Respiration not yet integrated as a conditioning feature.)*
* **Conditioning Mechanism:** Implemented as prepended discrete condition tokens (Section 5 above) rather than concatenated embeddings or cross-attention — chosen for simplicity and because it required zero architecture changes to the existing Transformer.

### Phase 2: Safe Reinforcement Learning Controller — Status: Core implementation completed
* **Environment Modeling:** Done — `rl_env.py`, state = physiological indicators, action = a 4-way discrete tempo/complexity choice (see "Known open items" above regarding the gap between this and the originally-envisioned finer-grained lever control).
* **Safety Constraints:** Done — abrupt-transition and oscillation detection, both capped at a flat penalty rather than stacked.
* **Reward Function:** Done — therapeutic reward for calming choices under detected high stress, safety penalty for the converse.

### Phase 3: Benchmark Studies (5-Arm Comparison) — Status: Not started
Conduct comparative evaluations across the required baseline conditions:
1. Silence / No music
2. Fixed static music
3. Therapist-selected music
4. Non-adaptive generative AI music (unconditioned baseline)
5. Closed-loop adaptive AI music (our conditioned model + safe RL)

### Phase 4: Robustness & Systems Evaluation — Status: Not started
* **Fault Tolerance:** Test model behavior under sensor noise, missing data packets, and simulated sensor disconnects.
* **Systems Profiling:** Measure inference latency, memory consumption, and real-time generation throughput.
* **Statistical Rigor:** Run multi-seed experiments with confidence intervals and statistical significance testing.

### Phase 5: Qualitative, Ethical, and Usability Analysis — Status: Not started
* **User & Subject Evaluation:** Assess listener adherence, fatigue levels, and usability metrics.
* **Analysis & Reporting:** Document model explainability, sensor data privacy considerations, and calibration results.

---

## 1. Kaggle Notebook Analysis Framework

**Status: Completed (see `kaggle_analysis.xlsx`).**

The 5-point rubric applied across the 10 reference notebooks:

### Point 1 — Data Representation / Tokenization Strategy
- Tokenization scheme name, vocabulary size, max sequence length used, and handling of explicit vs. implicit tempo/dynamics tokens.

### Point 2 — Architecture Pattern
- Model family (Transformer decoder, LSTM, VAE), parameter counts, context window length, and training compute requirements.

### Point 3 — Conditioning Mechanism
- Method of injecting conditioning signals (prepended tokens, embedding concatenation, cross-attention, FiLM).

### Point 4 — Evaluation Metrics Used
- Loss metrics (perplexity, cross-entropy), music-theoretic metrics (pitch entropy, note density, key consistency), and qualitative/listening tests.

### Point 5 — Reproducibility & Failure Modes
- Identified failure modes (repetition loops, mode collapse, key drift) and preprocessing scale.

---

## 2. Directory Layout

```
week1_baseline/
shared/
config.py                 # Single source of truth: vocab size, seq len, model dims, checkpoint schema,
                           # discrete condition-token ids.
vocabulary.py              # Token <-> id mapping, MIDI parsing, and discrete condition tokens (ids 451-454).
developer_a/
dataset.py                 # PyTorch Dataset/DataLoader for symbolic music sequences; optionally prepends
                            # condition tokens per window when --labels_csv is supplied.
model.py                   # Baseline Transformer decoder architecture (unchanged by condition-token support).
train.py                   # Training loop, validation, checkpointing; --labels_csv flag enables conditioning.
developer_b/
checkpoint_loader.py       # Checkpoint loading routines.
sampler.py                 # Temperature / top-k / top-p sampling with repetition constraints.
postprocess.py             # Token sequence -> MIDI conversion.
generate_song.py           # End-to-end generation script.
split_dataset.py             # Dataset split utility for MIDI directories.
label_midi_features.py       # Scans MIDI dataset, quantizes tempo/complexity into discrete bins -> midi_labels.csv.
check_wesad.py                # WESAD dataset structure verification script.
extract_features.py           # Physiological feature extraction (HRV, EDA), scaled across all WESAD subjects.
rl_env.py                     # PatientPhysiologyEnv -- gymnasium.Env simulating a patient from the physiological timeline.
train_rl.py                   # PPO training loop (stable-baselines3) for the Safe RL controller.
closed_loop_generate.py       # End-to-end integration: RL agent + conditioned Transformer -> generated MIDI per timestep.
kaggle_analysis.xlsx          # 10 Kaggle notebook comparative analysis.
classical_dataset/            # Raw classical MIDI dataset (local only, gitignored).
dataset/                      # Raw multi-genre MIDI dataset (local only, gitignored).
data/                          # Train/val split outputs (local only, gitignored).
wesad_data/                    # Raw WESAD dataset (local only, gitignored).
checkpoints/                   # Model checkpoints, including rl_controller.zip (Safe RL policy).
generated_songs/                # Initial multi-genre generated MIDI outputs.
generated_songs_classical/      # Classical baseline generated MIDI outputs.
generated_songs_conditioned/    # Condition-token-conditioned generated MIDI outputs.
closed_loop_output/              # Per-timestep MIDI generated by closed_loop_generate.py, plus manifest.csv.
midi_labels.csv                  # label_midi_features.py output: per-file tempo/complexity bins.
S2_physiological_timeline.csv    # Original single-subject sanity-check output (see Section 4).
wesad_physiological_timeline.csv # Aggregated multi-subject physiological timeline (see Section 6), used by rl_env.py.
```

## 3. Parallelization Contract

- Freeze `shared/config.py` and `shared/vocabulary.py` as the fixed interface between generation and training modules.
- Developer A owns model training and architecture (`developer_a/`). Developer B owns inference, sampling, and post-processing (`developer_b/`). Neither imports directly from the other's module.
- Integration point remains `shared/config.CHECKPOINT_SCHEMA` and the `MusicSequenceModel` interface.

## 4. Reproducing Baseline Runs

To reproduce the data split, training, and generation runs:

```bash
# 1. Classical Baseline Split & Train (unconditioned)
python3 split_dataset.py --source_root classical_dataset --output_root data/classical_split --val_fraction 0.1
python3 -m developer_a.train --data_root data/classical_split

# 2. Song Generation (unconditioned)
python3 -m developer_b.generate_song --checkpoint checkpoints/checkpoint_best.pt --output_dir generated_songs_classical --num_songs 5

# 3. Physiological Feature Extraction Sanity Check (single subject)
python3 extract_features.py

# 4. Condition-token labeling + conditioned training
python3 label_midi_features.py --data_root data/classical_split --output midi_labels.csv
python3 -m developer_a.train --data_root data/classical_split --labels_csv midi_labels.csv

# 5. Safe RL controller training + closed-loop generation
python3 train_rl.py --csv_path wesad_physiological_timeline.csv --timesteps 75000
python3 closed_loop_generate.py --subject S2 --physio_csv wesad_physiological_timeline.csv \
    --rl_checkpoint checkpoints/rl_controller.zip --music_checkpoint checkpoints/checkpoint_best.pt \
    --output_dir closed_loop_output
```
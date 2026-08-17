# Week 1 — Physiology-Conditioned Generative Music: Baseline Establishment

## Status Summary (read this first)

**Where things stand:** the Week 1 baseline and initial data pipelines are complete and pushed to GitHub. Here is the summary of what has been implemented:

- Built the codebase split around a shared contract (`shared/config.py`, `shared/vocabulary.py`) for parallel development.
- Initial multi-genre baseline: Trained on 5 genres (`ambient`, `classical`, `jazz`, `pop`, `soundtracks`) using `split_dataset.py` to produce `data/new_dataset/`. Generated baseline songs are in `generated_songs/`.
- Classical baseline: Curated a single-genre classical dataset (`classical_dataset/`) split into `data/classical_split/` to reduce genre-confusion. Retrained the model and generated the classical baseline batch in `generated_songs_classical/`.
- Model checkpoints: Best weights are tracked in `checkpoints/checkpoint_best.pt`.
- Reference review: Completed the 10 Kaggle reference notebook analysis (`kaggle_analysis.xlsx`).
- Physiological data pipeline: Verified WESAD file integrity (`check_wesad.py`) and verified extraction of Heart Rate, HRV (SDNN), and EDA (Skin Conductance Level) metrics (`extract_features.py`).

## Collaborator Setup & Requirements

**1. Local Environment & Dependencies**
Model checkpoints and datasets are intentionally excluded from version control to save space. To set up your local environment:
* Pull the latest changes from the `main` branch.
* Install required physiological processing libraries: `pip install -r requirements.txt`

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

**4. Assigned Task: Real-Time Windowing & Target Mapping**
The current `extract_features.py` script extracts static HRV and EDA metrics from the raw WESAD data. Your task is to convert this into a continuous, simulated real-time dataset to serve as the environment for the Reinforcement Learning controller.

**Action Items:**
* **Time-Windowing:** Update `extract_features.py` to iterate through the data in 10-second rolling windows rather than processing the entire condition block at once.
* **Multi-State Extraction:** Extract features for the Baseline (Label 1), Stress (Label 2), and Meditation (Label 4) conditions.
* **Rule-Based Mapper:** Create a function `calculate_music_targets(hrv, eda)` that applies our safety constraints to map physiological states to musical targets (e.g., high EDA/low HRV outputs a lower target tempo and complexity).
* **Data Export:** Save the windowed results to a CSV file (e.g., `S2_physiological_timeline.csv`) containing the following columns: `[Timestamp, Condition_Label, HRV_SDNN, Mean_EDA, Target_Tempo, Target_Complexity]`.

---

## Project Completion Roadmap & Technical Requirements

The following phases outline the remaining deliverables required for the final research prototype and evaluation suite:

### Phase 1: Conditioning & Feature Ingestion
* **Multimodal Integration:** Ingest extracted windowed features (HRV, EDA, Respiration) alongside tokenized MIDI sequences.
* **Conditioning Mechanism:** Implement feature injection into the generative model (e.g., prepended condition tokens, concatenated embeddings, or cross-attention layers).

### Phase 2: Safe Reinforcement Learning Controller
* **Environment Modeling:** Construct a gym-style environment where state represents patient physiological indicators and action space controls musical levers (tempo delta, harmonic tension, rhythmic complexity).
* **Safety Constraints:** Implement hard action bounds and rate limiters to prevent abrupt musical shifts, excessive volume spikes, or destabilizing patterns.
* **Reward Function:** Formulate an optimization objective focused on stress reduction and physiological stabilization.

### Phase 3: Benchmark Studies (5-Arm Comparison)
Conduct comparative evaluations across the required baseline conditions:
1. Silence / No music
2. Fixed static music
3. Therapist-selected music
4. Non-adaptive generative AI music (unconditioned baseline)
5. Closed-loop adaptive AI music (our conditioned model + safe RL)

### Phase 4: Robustness & Systems Evaluation
* **Fault Tolerance:** Test model behavior under sensor noise, missing data packets, and simulated sensor disconnects.
* **Systems Profiling:** Measure inference latency, memory consumption, and real-time generation throughput.
* **Statistical Rigor:** Run multi-seed experiments with confidence intervals and statistical significance testing.

### Phase 5: Qualitative, Ethical, and Usability Analysis
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
config.py                 # Single source of truth: vocab size, seq len, model dims, checkpoint schema.
vocabulary.py             # Token <-> id mapping and MIDI parsing.
developer_a/
dataset.py                # PyTorch Dataset/DataLoader for symbolic music sequences.
model.py                  # Baseline Transformer decoder architecture.
train.py                  # Training loop, validation, checkpointing.
developer_b/
checkpoint_loader.py      # Checkpoint loading routines.
sampler.py                # Temperature / top-k / top-p sampling with repetition constraints.
postprocess.py            # Token sequence -> MIDI conversion.
generate_song.py          # End-to-end generation script.
split_dataset.py            # Dataset split utility for MIDI directories.
check_wesad.py              # WESAD dataset structure verification script.
extract_features.py         # Physiological feature extraction script (HRV, EDA).
kaggle_analysis.xlsx        # 10 Kaggle notebook comparative analysis.
classical_dataset/          # Raw classical MIDI dataset (local only, gitignored).
dataset/                    # Raw multi-genre MIDI dataset (local only, gitignored).
data/                       # Train/val split outputs (local only, gitignored).
wesad_data/                 # Raw WESAD dataset (local only, gitignored).
checkpoints/                # Model checkpoint files.
generated_songs/            # Initial multi-genre generated MIDI outputs.
generated_songs_classical/  # Classical baseline generated MIDI outputs.
```

## 3. Parallelization Contract

- Freeze `shared/config.py` and `shared/vocabulary.py` as the fixed interface between generation and training modules.
- Developer A owns model training and architecture (`developer_a/`). Developer B owns inference, sampling, and post-processing (`developer_b/`). Neither imports directly from the other's module.
- Integration point remains `shared/config.CHECKPOINT_SCHEMA` and the `MusicSequenceModel` interface.

## 4. Reproducing Baseline Runs

To reproduce the data split, training, and generation runs:

```bash
# 1. Classical Baseline Split & Train
python3 split_dataset.py --source_root classical_dataset --output_root data/classical_split --val_fraction 0.1
python3 -m developer_a.train --data_root data/classical_split

# 2. Song Generation
python3 -m developer_b.generate_song --checkpoint checkpoints/checkpoint_best.pt --output_dir generated_songs_classical --num_songs 5

# 3. Physiological Feature Extraction Sanity Check
python3 extract_features.py
```
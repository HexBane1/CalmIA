# Phase 5: Research and Ethics Report — CalmIA

**Physiology-Conditioned Generative Music with Safe Reinforcement Learning for Personalized Rehabilitation**

---

## Document Scope and Evidentiary Basis

This document addresses the three components of the final research rubric requiring narrative analysis rather than code: model explainability and calibration, privacy and ethical analysis, and the formal design of the 5-arm benchmark study. It is written against the system as implemented through Phase 4 (baseline generation, discrete condition-token conditioning, the Safe RL controller, and the robustness/systems evaluation).

Quantitative results cited from `robustness_test.py` and `profile_system.py` reflect a specific documented run, executed by the author on their own hardware against subject S2's full 251-window timeline. The verbatim terminal output from both runs is reproduced in the Appendix and is the source of every figure cited below. These figures are reported as empirical findings from that specific run, not as formally validated guarantees across all subjects, seeds, or hardware configurations — see Section 1.3 for what would be required to strengthen that claim.

---

## 1. Model Explainability & Calibration

### 1.1 Decision Rationale of the RL Controller

The Safe RL controller's decision process is directly interpretable by design, which is itself a deliberate architectural choice: rather than a black-box policy over a high-dimensional continuous action space, `PatientPhysiologyEnv` (`rl_env.py`) uses a 4-way discrete action space (`Slow/Low`, `Slow/High`, `Fast/Low`, `Fast/High`), each action tied to an explicit, human-readable pair of condition tokens (`TEMPO_SLOW`/`TEMPO_FAST`, `COMPLEXITY_LOW`/`COMPLEXITY_HIGH`). This means any decision the trained policy makes can be reported not as an opaque numeric action index, but as a direct statement of intent ("the controller selected the calming configuration").

The policy's incentive to select `Slow/Low` specifically under detected high-stress physiological states (`HRV_SDNN < 50ms` and `Mean_EDA > 2.0µS`) is not emergent or inferred after the fact — it is explicitly engineered into the reward function as a `+1` therapeutic reward, applied only when that exact (state, action) pairing occurs. The mapping from "high EDA / low HRV" to the calming token pair is therefore explainable at the level of the reward specification itself, not merely observable as a statistical correlation in trained behavior.

### 1.2 Reward Structure and the Asymmetric Safety Penalty

The reward function combines two additive components with deliberately asymmetric magnitudes: a `+1` therapeutic reward for the single correct calming choice under stress, against a `-5` safety penalty for any of three defined violations (an abrupt maximally-opposite transition, a 3-step oscillation, or prescribing the most stimulating configuration during detected high stress).

This 5:1 asymmetry is the mechanism by which the reward function is intended to bias the policy toward conservative, risk-averse behavior over aggressive reward-seeking: under standard reinforcement learning optimization, a policy that accumulates reward by chasing the `+1` therapeutic bonus while incurring even a modest rate of `-5` safety violations will, in expectation, underperform a more conservative policy that forgoes some therapeutic reward to avoid violations entirely. The mathematical intent is that "do no harm" dominates "actively help" whenever the two are in tension — consistent with the professor's stated requirement to guard against destabilizing musical patterns.

It is important to state precisely what this mechanism does and does not guarantee. The `-5` penalty is a *learned deterrent applied during training*, not a hard constraint enforced at the level of the action space — the trained policy remains structurally capable of selecting any of the four actions at any state, including ones that would trigger a safety violation. The asymmetric reward shapes the *probability distribution* the trained policy converges toward; it does not architecturally forbid unsafe actions. This distinction matters for how the empirical results below should be interpreted.

### 1.3 Empirical Evidence from Robustness Testing

The reported robustness evaluation (`robustness_test.py`, subject S2, 251 windows, 15% noise probability, 15% dropout probability) found **zero** occurrences of the worst-case failure mode — the trained policy prescribing the most stimulating configuration (`Fast/High`) while the true, uncorrupted physiological state indicated high stress — in both the clean run (0/251) and the corrupted run (0/251). This specific metric is what the reward asymmetry in Section 1.2 is designed to prevent, and it held at zero under the tested corruption level.

A more precise reading of the full result is warranted, however, rather than a blanket "zero failures" claim: the corrupted run did see the dangerous action chosen once overall (1/251, 0.4%, versus 0/251 on clean data), and saw both abrupt transitions (2, versus 0 clean) and 3-step oscillations (1, versus 0 clean) increase. That single dangerous-action instance did not coincide with a true high-stress state — which is why the worst-case metric remained zero — but the increase in transition instability under corruption is itself a real, non-trivial finding: sensor corruption measurably degraded the smoothness of the controller's decisions even though it did not, in this run, cause the specific safety failure the reward function was designed to prevent. Both observations should be reported together, not just the more favorable one.

This finding should be reported as a single-run observation under specific test parameters, not a formally validated bound. Strengthening this claim for a final submission would require: repeating the test across multiple random seeds and multiple subjects (the current WESAD-derived dataset spans up to 14 usable subjects); reporting a violation rate with a confidence interval rather than a raw count; and testing across a range of noise/dropout probabilities rather than a single fixed setting. The codebase's own `robustness_test.py` accepts `--noise_prob`, `--dropout_prob`, and `--seed` as parameters specifically to support this kind of multi-condition sweep.

### 1.4 Known Limitations to Explainability

Two caveats limit how far the explainability claims above can be extended:

- **Threshold provenance.** The `HRV_SDNN < 50ms` / `Mean_EDA > 2.0µS` high-stress definition, used both by the reward function and by `calculate_music_targets()`, is a first-pass value not calibrated against the full aggregated population or validated against any clinical ground truth. The controller's decisions are fully explainable *relative to this definition*, but the definition itself has not been independently validated.
- **Condition-scheme reconciliation.** As documented in the project README, the MIDI-side condition tokens (a 2-tier scheme) and the WESAD-side `calculate_music_targets()` output (a 3-tier numeric scheme) have not been formally reconciled. The RL controller's action space uses its own independent 4-way mapping. All three are internally consistent, but a reader attempting to trace a single physiological reading through the entire pipeline to a specific generated musical parameter should be aware that the three schemes are not yet a single unified mapping.

---

## 2. Privacy & Ethical Analysis

### 2.1 Data Locality and On-Premises Processing

The current implementation has no network dependency anywhere in the physiological-to-music pipeline: `rl_env.py` reads physiological data from a local CSV, `train_rl.py` and the Transformer generation path (`developer_b/checkpoint_loader.py`, `sampler.py`, `postprocess.py`) operate entirely on local files and in-process model inference, and no component of the codebase makes an outbound network call. This is a verifiable property of the code as written, not an inference from the resource-usage measurements.

The profiling results (Phase 4, `profile_system.py`, CPU-only, 50 measured steps) recorded a real-time margin of 3.8x (mean 2604.9ms per simulated window against a 10-second budget) and a total memory footprint of 351.9MB RSS after both models are loaded, with **zero measurable growth** over that baseline across all 50 measured inference steps — indicating stable memory behavior under sustained operation rather than accumulation or a leak. These results support the claim that this pipeline can run entirely on a standard clinical workstation without requiring cloud offload for computational reasons. This is a meaningfully different claim from asserting the architecture is proven suitable for arbitrary low-power or wearable clinical hardware, which was not tested and should not be inferred from workstation-class CPU measurements. For biometric data such as ECG-derived HRV and EDA — both of which could plausibly be considered sensitive health data under frameworks such as GDPR or HIPAA depending on jurisdiction and deployment context — the demonstrated property is: **no architectural requirement exists to transmit raw physiological telemetry off-device for this system to function**, which is a strong starting position for privacy-by-design, subject to future validation on the actual target deployment hardware.

### 2.2 Ethical Guardrails in the Generative Process

The system incorporates two categories of guardrail against generating destabilizing musical output:

1. **Reward-shaped deterrents in the RL controller** (Section 1.2): a learned bias against abrupt transitions and against prescribing high-intensity music during detected patient distress. As established above, this is probabilistic, not absolute.
2. **Bounded action space**: the controller can only select from four pre-defined, pre-validated musical configurations — it cannot request an out-of-distribution or unbounded musical parameter combination, since the action space itself is closed and small. This is a structural (architectural) guarantee, distinct from the learned reward-shaping in point 1, and is a stronger claim: regardless of training quality, the controller is mechanically incapable of requesting a configuration outside the four defined options.

A genuinely hard, architecturally-enforced constraint (e.g., a rate limiter on the generation side that physically rejects a proposed action if it violates a defined safety rule, rather than merely disincentivizing it during training) is not currently implemented. This would be a natural extension: layering a deterministic safety filter between the RL controller's output and the generator's input, which would upgrade the current probabilistic guarantee to an absolute one, directly satisfying the professor's original requirement for "controls to prevent abrupt tempo changes, excessive sound intensity, or destabilizing musical patterns" in the strongest possible sense.

### 2.3 Residual Ethical Considerations

- **Clinical validation gap.** No component of this system has been validated against real patient outcomes. The therapeutic reward function encodes a hypothesis (calming music during physiological stress is beneficial) drawn from the professor's stated safety constraints, not from a validated clinical evidence base specific to this patient population or intervention.
- **Population representativeness.** WESAD's subject pool (15 individuals, non-patient, laboratory-induced stress rather than rehabilitation-context stress) differs from the eventual target population in ways that may limit how directly current threshold calibrations and learned policies transfer.
- **Human-subject research requirements.** Any progression from the current offline-simulated environment to real patient testing (Section 3 below) would require formal ethics committee / institutional review board approval, informed consent procedures, and a defined safety-monitoring and stopping-rule protocol — none of which are addressed by the current codebase, since no human-subject data has yet been collected or used.

---

## 3. The 5-Arm Benchmark Study Design (Future Work)

### 3.1 Study Design Overview

A within-subject randomized crossover design is proposed, in which each participant is exposed to all five conditions in a randomized or counterbalanced order across separate sessions, with a washout period between sessions to minimize carryover effects. A crossover design is appropriate here specifically because it allows each participant to serve as their own control, substantially increasing statistical power relative to a between-subjects design for a study of this likely scale, at the cost of requiring multiple sessions per participant and careful control for order/carryover effects.

### 3.2 The Five Arms

| Arm | Condition | Implementation Basis |
|---|---|---|
| 1 | Silence / no music | No audio stimulus during the rehabilitation task. |
| 2 | Fixed static music | A single, unchanging pre-recorded track, held constant across the session and across participants. |
| 3 | Therapist-selected music | Music manually selected by a clinician/therapist, based on their independent judgment of what suits the patient — not generated or influenced by this system. |
| 4 | Non-adaptive generative AI | The Week 1 unconditioned baseline Transformer (`generated_songs_classical/`), generating music with no physiological input. |
| 5 | Closed-loop adaptive AI | The full Phase 2 system: real-time physiological input, the trained Safe RL controller, and condition-token-conditioned generation (`closed_loop_generate.py`). |

Arms 4 and 5 isolate the specific contribution of physiological adaptivity, holding the underlying generative model constant — this is the comparison most directly relevant to the professor's stated research question ("can physiologically adaptive music improve outcomes compared with fixed music, therapist-selected music, and silence").

### 3.3 Primary and Secondary Endpoints

**Primary endpoints:**
- Patient-reported stress (validated self-report instrument, administered pre/post session).
- Motor-task performance (task-specific quantitative measure, defined by the rehabilitation protocol being studied).

**Secondary endpoints:**
- Task adherence (session completion rate, voluntary early termination rate).
- Physiological stabilization (HRV/EDA trajectory during the session, using the same feature extraction pipeline as `extract_features.py`).
- Fatigue (validated self-report instrument, post-session).
- Usability and therapist assessment (structured questionnaire, administered to both patient and supervising clinician).

### 3.4 Statistical Analysis Plan

Two distinct sources of variability must be addressed, and should not be conflated in the analysis plan:

- **Participant-level variability** (the actual clinical question) should be addressed via a repeated-measures / mixed-effects model appropriate to the crossover design, with participant as a random effect and arm as a fixed effect, reporting effect sizes with 95% confidence intervals rather than p-values alone. Sample size should be determined via a priori power analysis once a minimum clinically meaningful effect size is defined with clinical collaborators, rather than assumed.
- **Model-level stochastic variability** (an AI-system reproducibility concern, distinct from the clinical question above) applies specifically to Arms 4 and 5: since both involve a stochastic generative model, results for these two arms specifically should be reported across multiple random seeds for model training and multiple independent generation samples per seed, to characterize how much of any observed effect is attributable to the model's inherent stochasticity versus a genuine physiological response. This is a machine-learning reproducibility practice and should not be conflated with the participant-level statistical design above — "multiple random seeds" answers "is this AI system's behavior stable," not "does this intervention work on patients," and both questions need to be answered, separately, for the study's conclusions to be defensible.

### 3.5 Ethical Oversight Requirements

Before any human-subject data collection under this protocol, the following are required, none of which are currently in place:

- Institutional Review Board (or equivalent ethics committee) approval, including review of the informed consent procedure.
- A defined safety-monitoring plan and pre-specified stopping rules, particularly relevant given the rehabilitation/patient population and the system's direct control over a sensory stimulus during a physical task.
- A data management and privacy plan for any physiological data collected in the study itself, consistent with the architectural privacy properties described in Section 2.1, but extended to cover study-specific data retention, storage, and eventual disposal.

### 3.6 Sample Size Considerations

A specific target sample size is deliberately not proposed here, since a defensible number requires a formal power analysis conditioned on: the minimum clinically meaningful effect size (to be defined in consultation with clinical collaborators, not assumed by the engineering team), the expected within-subject correlation structure of a crossover design, and the number of arms (5, which increases the number of pairwise comparisons and should be accounted for via appropriate multiple-comparison correction in the analysis plan). Proposing a specific number without this analysis would itself be a methodological weakness in the protocol.

---

## Traceability

| Claim | Source Artifact |
|---|---|
| Reward structure, action space, safety violation definitions | `rl_env.py` |
| Robustness test methodology and results | `robustness_test.py` |
| Latency/memory profiling methodology and results | `profile_system.py` |
| Condition-token conditioning mechanism | `label_midi_features.py`, `shared/vocabulary.py`, `developer_a/dataset.py` |
| Physiological feature extraction and threshold definitions | `extract_features.py` |
| Known open items referenced throughout | `README.md`, Status Summary and "Known open items" sections |

---

## Appendix: Empirical Evaluation Logs

Verbatim terminal output from the runs cited throughout this document, executed on the author's local hardware (Windows, CPU-only) against subject S2's full physiological timeline (251 windows).

### Robustness Test Output

```
Loading RL controller from checkpoints/rl_controller.zip...
Loading reference environment over the FULL dataset for normalization statistics...
  (population stats: HRV_SDNN mean=61.79 std=42.88, Mean_EDA mean=4.634 std=3.399)

--- CLEAN baseline -- subject S2 (251 steps) ---
  Dangerous action (Fast/High) chosen:        0/251 (0.0%)
  Calming action (Slow/Low) chosen:            237/251 (94.4%)
  Dangerous action during TRUE high stress:    0/251  <-- worst-case failure mode
  Abrupt (maximally-opposite) transitions:     0
  3-step oscillations (A->B->A):                0

--- CORRUPTED run -- subject S2 (noise_prob=0.15, dropout_prob=0.15) (251 steps) ---
  Dangerous action (Fast/High) chosen:        1/251 (0.4%)
  Calming action (Slow/Low) chosen:            220/251 (87.6%)
  Dangerous action during TRUE high stress:    0/251  <-- worst-case failure mode
  Abrupt (maximally-opposite) transitions:     2
  3-step oscillations (A->B->A):                1
  Rows with injected noise:                    33
  Rows with dropped packets:                   32
  Dangerous action specifically on a dropped-packet row: 0/32

--- Comparison ---
Actions that differ between clean and corrupted runs: 19/251 (7.6%)
```

### System Profiling Output

```
Loading RL controller from checkpoints/rl_controller.zip...
Loading music generator checkpoint from checkpoints/checkpoint_best.pt...
  (device: cpu, trained through epoch 19)

Running 5 warmup step(s) (excluded from measurements)...
Running 50 measured step(s)...

============================================================
SYSTEMS PROFILING SUMMARY
============================================================
Device:                                  cpu
Measured steps:                          50 (+5 warmup, excluded)

-- Latency (milliseconds) --
  RL decision      : mean=   0.87  p50=   0.83  p95=   1.26
  Music generation : mean=2603.93  p50=2539.68  p95=2963.55
  Total per window : mean=2604.90  p50=2540.51  p95=2964.48

-- Throughput --
  Windows/second:                        0.38
  Tokens/second:                         99.0
  Real-time margin (10s budget): 3.8x (>1x means faster than the 10-second window it needs to keep up with)

-- Memory --
  Baseline RSS (models loaded, idle): 351.9 MB
  Peak RSS during inference:           351.9 MB
```
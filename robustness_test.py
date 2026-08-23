"""
robustness_test.py -- Fault Tolerance & Noise (Phase 4, Script 1).

Tests how the trained PPO agent (checkpoints/rl_controller.zip) behaves when
its physiological input is corrupted, compared against how it behaves on the
clean ground-truth signal for the same subject and timepoints.

DESIGN NOTES (read before trusting the numbers):

1. rl_env.py is treated as read-only here -- its constants (thresholds,
   action definitions, opposite-action pairs) are imported and reused
   directly, never reimplemented, so "dangerous" and "abrupt transition"
   mean exactly the same thing in this test as they do during training.

2. Normalization statistics matter. The agent was trained on observations
   normalized using the FULL aggregated dataset's mean/std. This script
   builds one reference PatientPhysiologyEnv over the whole (uncorrupted)
   CSV purely to obtain those exact statistics, then applies them manually
   to the corrupted test observations -- recomputing fresh statistics from
   just the corrupted subject's data would silently change what the agent is
   being tested against.

3. This is a diagnostic script, not training infrastructure -- it does NOT
   force the agent into a safe action when data is corrupted. The premise
   under test is whether the TRAINED policy already behaves safely when fed
   degraded input; hard-coding a safety override here would make the test
   trivially pass without answering that question.

4. NaN handling: a neural network cannot consume NaN. When a "packet" is
   dropped, this script falls back to holding the last known-good reading
   (a common practical strategy) before it reaches the agent -- this is an
   explicit, flagged design choice, not something intended to be invisible.
   The agent still acts on this held (stale) value entirely on its own.

5. Two distinct failure modes are tracked separately, because they mean
   different things clinically:
     (a) Does the agent choose the single most dangerous action (Fast/High)
         more often under corruption than on clean data at all?
     (b) Specifically, does it choose Fast/High at a moment when the TRUE
         (uncorrupted) state was actually high-stress -- corruption masking
         real patient distress and the agent responding as if things were
         fine. This is the worst-case failure and is reported separately.

Usage:
    python robustness_test.py --subject S2 --csv_path wesad_physiological_timeline.csv \
        --rl_checkpoint checkpoints/rl_controller.zip --noise_prob 0.15 --dropout_prob 0.15
"""

import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_env import (
    PatientPhysiologyEnv,
    HIGH_STRESS_HRV_THRESHOLD,
    HIGH_STRESS_EDA_THRESHOLD,
    OPPOSITE_ACTION,
    DANGEROUS_ACTION,
    CALMING_ACTION,
    ACTION_TO_TOKENS,
)


def is_high_stress(hrv: float, eda: float) -> bool:
    """Mirrors PatientPhysiologyEnv._is_high_stress() exactly -- duplicated
    here only because it's a private method on the env, not because the
    definition differs."""
    return hrv < HIGH_STRESS_HRV_THRESHOLD and eda > HIGH_STRESS_EDA_THRESHOLD


def corrupt_timeline(
    clean_hrv: np.ndarray,
    clean_eda: np.ndarray,
    noise_prob: float,
    dropout_prob: float,
    noise_scale_hrv: float,
    noise_scale_eda: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Builds a corrupted copy of a clean (hrv, eda) sequence. Each row
    independently has probability noise_prob of receiving a Gaussian noise
    spike, and (independently) probability dropout_prob of being dropped
    entirely (both channels set to NaN, simulating a lost packet). A row can
    be hit by both, or neither.

    Returns (corrupted_hrv, corrupted_eda, fault_log) where fault_log[i] is
    "none", "noise", "dropout", or "noise+dropout" for row i.
    """
    n = len(clean_hrv)
    corrupted_hrv = clean_hrv.copy()
    corrupted_eda = clean_eda.copy()
    fault_log = ["none"] * n

    for i in range(n):
        applied = []
        if rng.random() < noise_prob:
            corrupted_hrv[i] += rng.normal(0, noise_scale_hrv)
            corrupted_eda[i] += rng.normal(0, noise_scale_eda)
            applied.append("noise")
        if rng.random() < dropout_prob:
            corrupted_hrv[i] = np.nan
            corrupted_eda[i] = np.nan
            applied.append("dropout")
        if applied:
            fault_log[i] = "+".join(applied)

    return corrupted_hrv, corrupted_eda, fault_log


def apply_hold_last_known_good(hrv: np.ndarray, eda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fills NaN entries by carrying forward the last non-NaN value -- the
    fallback the agent actually receives when a packet is dropped (see
    design note 4 above). The first row cannot be filled this way if it is
    itself NaN; falls back to 0.0 in that edge case only.
    """
    hrv_filled = hrv.copy()
    eda_filled = eda.copy()

    last_good_hrv, last_good_eda = 0.0, 0.0
    for i in range(len(hrv_filled)):
        if np.isnan(hrv_filled[i]):
            hrv_filled[i] = last_good_hrv
        else:
            last_good_hrv = hrv_filled[i]
        if np.isnan(eda_filled[i]):
            eda_filled[i] = last_good_eda
        else:
            last_good_eda = eda_filled[i]

    return hrv_filled, eda_filled


def run_policy_over_sequence(
    model: PPO,
    hrv_sequence: np.ndarray,
    eda_sequence: np.ndarray,
    hrv_mean: float,
    hrv_std: float,
    eda_mean: float,
    eda_std: float,
) -> List[int]:
    """
    Steps the trained policy (deterministically) over a fixed sequence of
    (already NaN-free) hrv/eda pairs, normalizing each with the SAME
    statistics the agent was trained under. Returns the list of chosen
    actions, one per row.
    """
    actions = []
    for hrv, eda in zip(hrv_sequence, eda_sequence):
        obs = np.array(
            [(hrv - hrv_mean) / hrv_std, (eda - eda_mean) / eda_std], dtype=np.float32
        )
        action, _states = model.predict(obs, deterministic=True)
        actions.append(int(action))
    return actions


def count_transition_violations(actions: List[int]) -> Tuple[int, int]:
    """
    Reuses rl_env.py's own OPPOSITE_ACTION definition to count (a) abrupt
    single-step transitions between maximally-opposite actions, and (b)
    3-step oscillations (A -> B -> A), exactly as PatientPhysiologyEnv
    defines them internally.
    """
    abrupt_count = 0
    oscillation_count = 0
    for i in range(1, len(actions)):
        if OPPOSITE_ACTION[actions[i]] == actions[i - 1]:
            abrupt_count += 1
        if i >= 2 and actions[i] == actions[i - 2] and OPPOSITE_ACTION[actions[i - 1]] == actions[i - 2]:
            oscillation_count += 1
    return abrupt_count, oscillation_count


def summarize_run(
    label: str,
    actions: List[int],
    true_hrv: np.ndarray,
    true_eda: np.ndarray,
    fault_log: Optional[List[str]] = None,
) -> None:
    n = len(actions)
    dangerous_count = sum(1 for a in actions if a == DANGEROUS_ACTION)
    calming_count = sum(1 for a in actions if a == CALMING_ACTION)

    # The worst-case failure mode: dangerous action chosen while the TRUE
    # (uncorrupted) state was actually high-stress.
    dangerous_during_true_stress = sum(
        1 for a, hrv, eda in zip(actions, true_hrv, true_eda)
        if a == DANGEROUS_ACTION and is_high_stress(hrv, eda)
    )

    abrupt_count, oscillation_count = count_transition_violations(actions)

    print(f"\n--- {label} ({n} steps) ---")
    print(f"  Dangerous action (Fast/High) chosen:        {dangerous_count}/{n} ({100 * dangerous_count / n:.1f}%)")
    print(f"  Calming action (Slow/Low) chosen:            {calming_count}/{n} ({100 * calming_count / n:.1f}%)")
    print(f"  Dangerous action during TRUE high stress:    {dangerous_during_true_stress}/{n}  <-- worst-case failure mode")
    print(f"  Abrupt (maximally-opposite) transitions:     {abrupt_count}")
    print(f"  3-step oscillations (A->B->A):                {oscillation_count}")

    if fault_log is not None:
        num_noise = sum(1 for f in fault_log if "noise" in f)
        num_dropout = sum(1 for f in fault_log if "dropout" in f)
        dangerous_on_dropout_rows = sum(
            1 for a, f in zip(actions, fault_log) if a == DANGEROUS_ACTION and "dropout" in f
        )
        print(f"  Rows with injected noise:                    {num_noise}")
        print(f"  Rows with dropped packets:                   {num_dropout}")
        print(f"  Dangerous action specifically on a dropped-packet row: "
              f"{dangerous_on_dropout_rows}/{max(num_dropout, 1)}")


def main():
    parser = argparse.ArgumentParser(description="Test the trained RL controller's fault tolerance under corrupted sensor data.")
    parser.add_argument("--subject", type=str, default="S2")
    parser.add_argument("--csv_path", type=str, default="wesad_physiological_timeline.csv")
    parser.add_argument("--rl_checkpoint", type=str, default="checkpoints/rl_controller.zip")
    parser.add_argument("--noise_prob", type=float, default=0.15, help="Per-row probability of a noise spike.")
    parser.add_argument("--dropout_prob", type=float, default=0.15, help="Per-row probability of a dropped packet.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading RL controller from {args.rl_checkpoint}...")
    model = PPO.load(args.rl_checkpoint)

    print(f"Loading reference environment over the FULL dataset for normalization statistics...")
    reference_env = PatientPhysiologyEnv(args.csv_path)
    hrv_mean, hrv_std = reference_env._hrv_mean, reference_env._hrv_std
    eda_mean, eda_std = reference_env._eda_mean, reference_env._eda_std
    print(f"  (population stats: HRV_SDNN mean={hrv_mean:.2f} std={hrv_std:.2f}, "
          f"Mean_EDA mean={eda_mean:.3f} std={eda_std:.3f})")

    if args.subject not in reference_env.subject_blocks:
        raise ValueError(
            f"Subject {args.subject!r} not found among usable subjects: "
            f"{sorted(reference_env.subject_blocks.keys())}"
        )
    subject_block = reference_env.subject_blocks[args.subject]
    true_hrv = subject_block["HRV_SDNN"].to_numpy()
    true_eda = subject_block["Mean_EDA"].to_numpy()

    # Noise spike magnitude: scaled relative to POPULATION std, so a "spike"
    # is genuinely anomalous relative to normal variation, not just typical
    # measurement jitter.
    noise_scale_hrv = 3.0 * hrv_std
    noise_scale_eda = 3.0 * eda_std

    # --- Control run: clean data, same subject ---
    clean_actions = run_policy_over_sequence(model, true_hrv, true_eda, hrv_mean, hrv_std, eda_mean, eda_std)
    summarize_run(f"CLEAN baseline -- subject {args.subject}", clean_actions, true_hrv, true_eda)

    # --- Corrupted run: same subject, same underlying true state, noisy input ---
    corrupted_hrv, corrupted_eda, fault_log = corrupt_timeline(
        true_hrv, true_eda, args.noise_prob, args.dropout_prob, noise_scale_hrv, noise_scale_eda, rng
    )
    fed_hrv, fed_eda = apply_hold_last_known_good(corrupted_hrv, corrupted_eda)
    corrupted_actions = run_policy_over_sequence(model, fed_hrv, fed_eda, hrv_mean, hrv_std, eda_mean, eda_std)
    summarize_run(
        f"CORRUPTED run -- subject {args.subject} "
        f"(noise_prob={args.noise_prob}, dropout_prob={args.dropout_prob})",
        corrupted_actions, true_hrv, true_eda, fault_log=fault_log,
    )

    # --- Direct comparison ---
    changed_actions = sum(1 for c, k in zip(clean_actions, corrupted_actions) if c != k)
    print(f"\n--- Comparison ---")
    print(f"Actions that differ between clean and corrupted runs: "
          f"{changed_actions}/{len(clean_actions)} ({100 * changed_actions / len(clean_actions):.1f}%)")
    print(
        "\nNote: 'dangerous action during TRUE high stress' is the metric that matters most --\n"
        "it isolates cases where sensor corruption caused the agent to act as if a genuinely\n"
        "distressed patient were fine. A high count there, even with low overall action-change\n"
        "rates, would be a meaningful safety finding worth investigating further."
    )


if __name__ == "__main__":
    main()
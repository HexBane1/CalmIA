"""
benchmark_arms.py -- 5-Arm Benchmark Comparison (Phase 3).

Arm 1: Silence
Arm 2: Fixed music
Arm 3: Therapist-selected (simulated)
Arm 4: Non-adaptive AI
Arm 5: Closed-loop adaptive AI

Usage:
    python benchmark_arms.py --subject S2 \
        --physio_csv wesad_physiological_timeline.csv \
        --rl_checkpoint checkpoints/rl_controller.zip \
        --music_checkpoint checkpoints/checkpoint_best.pt \
        --therapist_midi generated_songs_classical/baseline_song_01.mid \
        --output_dir benchmark_output \
        --arms 1 2 3 5
"""

import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import shared.vocabulary as vocab_module

from rl_env import (
    PatientPhysiologyEnv,
    ACTION_TO_TOKENS,
    OPPOSITE_ACTION,
    DANGEROUS_ACTION,
    CALMING_ACTION,
    HIGH_STRESS_HRV_THRESHOLD,
    HIGH_STRESS_EDA_THRESHOLD,
)

ARM_NAMES = {
    1: "Silence",
    2: "Fixed_Music",
    3: "Therapist_Selected",
    4: "NonAdaptive_AI",
    5: "ClosedLoop_Adaptive_AI",
}


def is_high_stress(hrv: float, eda: float) -> bool:
    return hrv < HIGH_STRESS_HRV_THRESHOLD and eda > HIGH_STRESS_EDA_THRESHOLD


def get_subject_timeline(physio_csv: str, subject_id: str) -> pd.DataFrame:
    df = pd.read_csv(physio_csv)
    subject_df = df[df["Subject"] == subject_id].dropna(
        subset=["HRV_SDNN", "Mean_EDA", "Mean_RSP_Rate"]
    ).reset_index(drop=True)
    if subject_df.empty:
        raise ValueError(f"No valid rows for subject {subject_id} in {physio_csv}")
    return subject_df


def run_arm_1_silence(timeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in timeline.iterrows():
        rows.append({
            "Arm": 1, "ArmName": ARM_NAMES[1],
            "Timestamp": row["Timestamp"],
            "Condition_Label": row["Condition_Label"],
            "HRV_SDNN": row["HRV_SDNN"],
            "Mean_EDA": row["Mean_EDA"],
            "Action": None, "ActionName": "silence",
            "IsHighStress": is_high_stress(row["HRV_SDNN"], row["Mean_EDA"]),
            "IsDangerous": False, "IsCalming": False,
        })
    return pd.DataFrame(rows)


def run_arm_2_fixed(timeline: pd.DataFrame, fixed_midi_path: str) -> pd.DataFrame:
    rows = []
    for _, row in timeline.iterrows():
        rows.append({
            "Arm": 2, "ArmName": ARM_NAMES[2],
            "Timestamp": row["Timestamp"],
            "Condition_Label": row["Condition_Label"],
            "HRV_SDNN": row["HRV_SDNN"],
            "Mean_EDA": row["Mean_EDA"],
            "Action": None,
            "ActionName": f"fixed:{os.path.basename(fixed_midi_path)}",
            "IsHighStress": is_high_stress(row["HRV_SDNN"], row["Mean_EDA"]),
            "IsDangerous": False, "IsCalming": False,
        })
    return pd.DataFrame(rows)


def run_arm_3_therapist(timeline: pd.DataFrame, therapist_midi_path: str) -> pd.DataFrame:
    rows = []
    for _, row in timeline.iterrows():
        rows.append({
            "Arm": 3, "ArmName": ARM_NAMES[3],
            "Timestamp": row["Timestamp"],
            "Condition_Label": row["Condition_Label"],
            "HRV_SDNN": row["HRV_SDNN"],
            "Mean_EDA": row["Mean_EDA"],
            "Action": None,
            "ActionName": f"therapist_sim:{os.path.basename(therapist_midi_path)}",
            "IsHighStress": is_high_stress(row["HRV_SDNN"], row["Mean_EDA"]),
            "IsDangerous": False, "IsCalming": False,
        })
    return pd.DataFrame(rows)


def run_arm_4_nonadaptive(
    timeline: pd.DataFrame,
    music_checkpoint: str,
    output_dir: str,
) -> pd.DataFrame:
    from developer_b.checkpoint_loader import load_model_from_checkpoint
    from developer_b.sampler import generate
    from developer_b.postprocess import tokens_to_midi
    from shared.config import DATA_CONFIG

    os.makedirs(output_dir, exist_ok=True)
    model, metadata = load_model_from_checkpoint(music_checkpoint)
    print(f"  [Arm 4] Loaded checkpoint from epoch {metadata['epoch']}")

    rows = []
    for step_idx, (_, row) in enumerate(timeline.iterrows()):
        primer = [DATA_CONFIG.bos_token_id]
        token_ids = generate(model, primer_ids=primer, max_new_tokens=256,
                             temperature=1.0, top_k=40, top_p=0.9,
                             repetition_penalty=1.2)
        midi_path = os.path.join(output_dir, f"arm4_step_{step_idx:04d}.mid")
        tokens_to_midi(token_ids, midi_path)

        rows.append({
            "Arm": 4, "ArmName": ARM_NAMES[4],
            "Timestamp": row["Timestamp"],
            "Condition_Label": row["Condition_Label"],
            "HRV_SDNN": row["HRV_SDNN"],
            "Mean_EDA": row["Mean_EDA"],
            "Action": 0, "ActionName": "unconditioned",
            "IsHighStress": is_high_stress(row["HRV_SDNN"], row["Mean_EDA"]),
            "IsDangerous": False, "IsCalming": False,
            "OutputFile": midi_path,
        })
        print(f"  [Arm 4] Step {step_idx+1}/{len(timeline)}: {midi_path}")

    return pd.DataFrame(rows)


def run_arm_5_closedloop(
    timeline: pd.DataFrame,
    physio_csv: str,
    subject_id: str,
    rl_checkpoint: str,
    music_checkpoint: str,
    output_dir: str,
) -> pd.DataFrame:
    from developer_b.checkpoint_loader import load_model_from_checkpoint
    from developer_b.sampler import generate
    from developer_b.postprocess import tokens_to_midi
    from shared.config import DATA_CONFIG

    os.makedirs(output_dir, exist_ok=True)
    rl_model = PPO.load(rl_checkpoint)
    music_model, metadata = load_model_from_checkpoint(music_checkpoint)
    print(f"  [Arm 5] Loaded music checkpoint from epoch {metadata['epoch']}")

    reference_env = PatientPhysiologyEnv(physio_csv)
    hrv_mean, hrv_std = reference_env._hrv_mean, reference_env._hrv_std
    eda_mean, eda_std = reference_env._eda_mean, reference_env._eda_std
    resp_mean, resp_std = reference_env._resp_mean, reference_env._resp_std

    rows = []
    action_history = []

    for step_idx, (_, row) in enumerate(timeline.iterrows()):
        norm_obs = np.array([
            (row["HRV_SDNN"] - hrv_mean) / hrv_std,
            (row["Mean_EDA"] - eda_mean) / eda_std,
            (row["Mean_RSP_Rate"] - resp_mean) / resp_std,
        ], dtype=np.float32)

        action, _ = rl_model.predict(norm_obs, deterministic=True)
        action = int(action)
        tempo_token_name, complexity_token_name = ACTION_TO_TOKENS[action]

        TOKEN_MAP = {
            "TEMPO_SLOW": vocab_module.TEMPO_SLOW,
            "TEMPO_FAST": vocab_module.TEMPO_FAST,
            "COMPLEXITY_LOW": vocab_module.COMPLEXITY_LOW,
            "COMPLEXITY_HIGH": vocab_module.COMPLEXITY_HIGH,
        }
        try:
            tempo_token_id = TOKEN_MAP[tempo_token_name]
            complexity_token_id = TOKEN_MAP[complexity_token_name]
            primer = [tempo_token_id, complexity_token_id, DATA_CONFIG.bos_token_id]
        except KeyError:
            primer = [DATA_CONFIG.bos_token_id]

        token_ids = generate(music_model, primer_ids=primer, max_new_tokens=256,
                             temperature=1.0, top_k=40, top_p=0.9,
                             repetition_penalty=1.2)
        midi_path = os.path.join(output_dir, f"arm5_step_{step_idx:04d}.mid")
        tokens_to_midi(token_ids, midi_path)

        is_abrupt = bool(action_history and OPPOSITE_ACTION[action] == action_history[-1])
        is_oscillation = (
            len(action_history) >= 2 and
            action == action_history[-2] and
            OPPOSITE_ACTION[action_history[-1]] == action_history[-2]
        )

        action_history.append(action)
        if len(action_history) > 2:
            action_history.pop(0)

        rows.append({
            "Arm": 5, "ArmName": ARM_NAMES[5],
            "Timestamp": row["Timestamp"],
            "Condition_Label": row["Condition_Label"],
            "HRV_SDNN": row["HRV_SDNN"],
            "Mean_EDA": row["Mean_EDA"],
            "Action": action,
            "ActionName": f"{tempo_token_name}/{complexity_token_name}",
            "IsHighStress": is_high_stress(row["HRV_SDNN"], row["Mean_EDA"]),
            "IsDangerous": action == DANGEROUS_ACTION,
            "IsCalming": action == CALMING_ACTION,
            "IsAbruptTransition": is_abrupt,
            "IsOscillation": is_oscillation,
            "OutputFile": midi_path,
        })
        print(f"  [Arm 5] Step {step_idx+1}/{len(timeline)}: action={action} -> {midi_path}")

    return pd.DataFrame(rows)


def print_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for arm_id, arm_name in ARM_NAMES.items():
        arm_df = all_results[all_results["Arm"] == arm_id]
        if arm_df.empty:
            continue
        n = len(arm_df)
        high_stress_steps = int(arm_df["IsHighStress"].sum())
        dangerous = arm_df["IsDangerous"].sum()
        calming = arm_df["IsCalming"].sum()
        dangerous_during_stress = (
            (arm_df["IsDangerous"] & arm_df["IsHighStress"]).sum()
        )
        summary_rows.append({
            "Arm": arm_id, "ArmName": arm_name,
            "TotalSteps": n,
            "HighStressSteps": high_stress_steps,
            "DangerousActionRate": dangerous / n,
            "CalmingActionRate": calming / n,
            "DangerousDuringStressRate": dangerous_during_stress / max(high_stress_steps, 1),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY — 5-ARM COMPARISON")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="5-arm benchmark comparison.")
    parser.add_argument("--subject", type=str, default="S2")
    parser.add_argument("--physio_csv", type=str, default="wesad_physiological_timeline.csv")
    parser.add_argument("--rl_checkpoint", type=str, default="checkpoints/rl_controller.zip")
    parser.add_argument("--music_checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--therapist_midi", type=str,
                        default="generated_songs_classical/baseline_song_01.mid")
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--arms", type=int, nargs="+", default=[1, 2, 3, 5])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timeline = get_subject_timeline(args.physio_csv, args.subject)
    print(f"Subject {args.subject}: {len(timeline)} steps")

    all_frames = []

    if 1 in args.arms:
        print("\nRunning Arm 1: Silence...")
        all_frames.append(run_arm_1_silence(timeline))

    if 2 in args.arms:
        print("\nRunning Arm 2: Fixed music...")
        if not os.path.isfile(args.therapist_midi):
            print(f"  WARNING: MIDI not found at {args.therapist_midi}, skipping arm 2")
        else:
            all_frames.append(run_arm_2_fixed(timeline, args.therapist_midi))

    if 3 in args.arms:
        print("\nRunning Arm 3: Therapist-selected (simulated)...")
        if not os.path.isfile(args.therapist_midi):
            print(f"  WARNING: MIDI not found at {args.therapist_midi}, skipping arm 3")
        else:
            all_frames.append(run_arm_3_therapist(timeline, args.therapist_midi))

    if 4 in args.arms:
        print("\nRunning Arm 4: Non-adaptive AI...")
        arm4_dir = os.path.join(args.output_dir, "arm4")
        all_frames.append(run_arm_4_nonadaptive(timeline, args.music_checkpoint, arm4_dir))

    if 5 in args.arms:
        print("\nRunning Arm 5: Closed-loop adaptive AI...")
        arm5_dir = os.path.join(args.output_dir, "arm5")
        all_frames.append(run_arm_5_closedloop(
            timeline, args.physio_csv, args.subject,
            args.rl_checkpoint, args.music_checkpoint, arm5_dir
        ))

    if not all_frames:
        print("No arms ran successfully.")
        return

    all_results = pd.concat(all_frames, ignore_index=True)
    results_path = os.path.join(args.output_dir, "benchmark_results.csv")
    all_results.to_csv(results_path, index=False)
    print(f"\nPer-step results saved to {results_path}")

    summary_df = print_summary(all_results)
    summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
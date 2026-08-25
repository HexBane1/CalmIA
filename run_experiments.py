"""
run_experiments.py -- Multi-Seed Statistical Testing & Safety Ablation (Phase 4).

Usage:
    python run_experiments.py --csv_path wesad_physiological_timeline.csv \
        --seeds 0 1 2 3 4 --timesteps 20000
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_env import (
    PatientPhysiologyEnv,
    OPPOSITE_ACTION,
    DANGEROUS_ACTION,
    CALMING_ACTION,
    SAFETY_PENALTY as DEFAULT_SAFETY_PENALTY,
)

CONDITIONS = {
    "safe": DEFAULT_SAFETY_PENALTY,
    "unsafe": 0.0,
}


@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    num_steps: int = 0
    dangerous_count: int = 0
    dangerous_during_true_stress_count: int = 0
    calming_count: int = 0
    abrupt_transition_count: int = 0
    oscillation_count: int = 0


def run_deterministic_episode(model: PPO, env: PatientPhysiologyEnv, subject_id: str) -> EpisodeStats:
    obs, info = env.reset(seed=0, options=None)
    assert info["subject"] == subject_id

    stats_ = EpisodeStats()
    action_history: List[int] = []
    terminated = False

    while not terminated:
        raw_hrv, raw_eda = info["raw_hrv"], info["raw_eda"]
        is_true_stress = raw_hrv < 50.0 and raw_eda > 2.0

        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        if action == DANGEROUS_ACTION:
            stats_.dangerous_count += 1
            if is_true_stress:
                stats_.dangerous_during_true_stress_count += 1
        if action == CALMING_ACTION:
            stats_.calming_count += 1

        if action_history and OPPOSITE_ACTION[action] == action_history[-1]:
            stats_.abrupt_transition_count += 1
        if len(action_history) >= 2:
            two_ago, one_ago = action_history[-2], action_history[-1]
            if action == two_ago and OPPOSITE_ACTION[one_ago] == two_ago:
                stats_.oscillation_count += 1

        action_history.append(action)
        if len(action_history) > 2:
            action_history.pop(0)

        obs, reward, terminated, truncated, info = env.step(action)
        stats_.total_reward += reward
        stats_.num_steps += 1

    return stats_


def evaluate_policy(model: PPO, csv_path: str) -> Dict[str, float]:
    reference_env = PatientPhysiologyEnv(csv_path)
    subject_ids = sorted(reference_env.subject_blocks.keys())

    all_stats: List[EpisodeStats] = []
    for subject_id in subject_ids:
        subject_env = PatientPhysiologyEnv(csv_path, subject_id=subject_id)
        all_stats.append(run_deterministic_episode(model, subject_env, subject_id))

    total_steps = sum(s.num_steps for s in all_stats)
    if total_steps == 0:
        raise RuntimeError("No steps evaluated.")

    return {
        "mean_episode_reward": float(np.mean([s.total_reward for s in all_stats])),
        "dangerous_action_rate": sum(s.dangerous_count for s in all_stats) / total_steps,
        "dangerous_during_true_stress_rate": (
            sum(s.dangerous_during_true_stress_count for s in all_stats) / total_steps
        ),
        "calming_action_rate": sum(s.calming_count for s in all_stats) / total_steps,
        "abrupt_transition_rate": sum(s.abrupt_transition_count for s in all_stats) / total_steps,
        "oscillation_rate": sum(s.oscillation_count for s in all_stats) / total_steps,
    }


def train_one_run(csv_path: str, condition_name: str, safety_penalty: float, seed: int, timesteps: int) -> PPO:
    def make_env():
        return Monitor(PatientPhysiologyEnv(csv_path, seed=seed, safety_penalty=safety_penalty))

    vec_env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed)
    print(f"  Training [{condition_name}, seed={seed}] for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    return model


def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95):
    rng = np.random.default_rng(42)
    boots = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boots, (1 - ci) / 2 * 100)
    upper = np.percentile(boots, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description="Multi-seed Safe-vs-Unsafe PPO ablation study.")
    parser.add_argument("--csv_path", type=str, default="wesad_physiological_timeline.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--output_csv", type=str, default="experiment_results.csv")
    args = parser.parse_args()

    results_rows = []

    for condition_name, safety_penalty in CONDITIONS.items():
        for seed in args.seeds:
            model = train_one_run(args.csv_path, condition_name, safety_penalty, seed, args.timesteps)
            metrics = evaluate_policy(model, args.csv_path)
            row = {"condition": condition_name, "seed": seed, **metrics}
            results_rows.append(row)
            print(
                f"    -> reward={metrics['mean_episode_reward']:.2f}  "
                f"dangerous_rate={metrics['dangerous_action_rate']:.3f}  "
                f"dangerous_during_true_stress_rate={metrics['dangerous_during_true_stress_rate']:.4f}"
            )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nPer-run results saved to {args.output_csv}")

    metric_names = [
        "mean_episode_reward",
        "dangerous_action_rate",
        "dangerous_during_true_stress_rate",
        "calming_action_rate",
        "abrupt_transition_rate",
        "oscillation_rate",
    ]

    print("\n" + "=" * 70)
    print("SUMMARY: Safe vs Unsafe (mean +/- std, 95% Bootstrap CI)")
    print("=" * 70)

    summary_rows = []
    for metric in metric_names:
        safe_values = results_df.loc[results_df["condition"] == "safe", metric].to_numpy()
        unsafe_values = results_df.loc[results_df["condition"] == "unsafe", metric].to_numpy()

        try:
            u_stat, p_value = stats.mannwhitneyu(safe_values, unsafe_values, alternative="two-sided")
        except ValueError:
            u_stat, p_value = float("nan"), float("nan")

        safe_ci = bootstrap_ci(safe_values)
        unsafe_ci = bootstrap_ci(unsafe_values)

        print(f"\n{metric}:")
        print(f"  safe:   {safe_values.mean():.4f} +/- {safe_values.std():.4f}  "
              f"95% CI [{safe_ci[0]:.4f}, {safe_ci[1]:.4f}]  (n={len(safe_values)})")
        print(f"  unsafe: {unsafe_values.mean():.4f} +/- {unsafe_values.std():.4f}  "
              f"95% CI [{unsafe_ci[0]:.4f}, {unsafe_ci[1]:.4f}]  (n={len(unsafe_values)})")
        print(f"  Mann-Whitney U p-value: {p_value:.4f}")

        summary_rows.append({
            "metric": metric,
            "safe_mean": safe_values.mean(),
            "safe_std": safe_values.std(),
            "safe_ci_lower": safe_ci[0],
            "safe_ci_upper": safe_ci[1],
            "unsafe_mean": unsafe_values.mean(),
            "unsafe_std": unsafe_values.std(),
            "unsafe_ci_lower": unsafe_ci[0],
            "unsafe_ci_upper": unsafe_ci[1],
            "mannwhitney_p_value": p_value,
        })

    summary_path = os.path.splitext(args.output_csv)[0] + "_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to {summary_path}")
    print(
        f"\nNOTE: with {len(args.seeds)} seed(s) per condition, p-values are "
        f"a first-pass signal. Bootstrap CIs computed with 1000 resamples."
    )


if __name__ == "__main__":
    main()
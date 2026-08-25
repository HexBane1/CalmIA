"""
optimize_hpo.py -- Hyperparameter Optimization for the Safe RL Controller (Phase 4).

Usage:
    python optimize_hpo.py --csv_path wesad_physiological_timeline.csv \
        --n_trials 15 --timesteps_per_trial 15000
"""

import argparse

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_env import PatientPhysiologyEnv
from run_experiments import evaluate_policy

N_STEPS_CHOICES = [64, 128, 256, 512]
BATCH_SIZE_DIVISORS = {
    64: [16, 32, 64],
    128: [32, 64, 128],
    256: [32, 64, 128, 256],
    512: [64, 128, 256, 512],
}


def build_model(env, trial_or_params, seed: int) -> PPO:
    if isinstance(trial_or_params, optuna.Trial):
        trial = trial_or_params
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_categorical("n_steps", N_STEPS_CHOICES)
        batch_size = trial.suggest_categorical(f"batch_size_for_{n_steps}", BATCH_SIZE_DIVISORS[n_steps])
        gamma = trial.suggest_float("gamma", 0.90, 0.9999)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    else:
        params = trial_or_params
        learning_rate = params["learning_rate"]
        n_steps = params["n_steps"]
        batch_size = params[f"batch_size_for_{n_steps}"]
        gamma = params["gamma"]
        ent_coef = params["ent_coef"]
        clip_range = params["clip_range"]

    return PPO(
        "MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
        gamma=gamma, ent_coef=ent_coef, clip_range=clip_range, seed=seed, verbose=0,
    )


def make_objective(csv_path: str, timesteps_per_trial: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        def make_env():
            return Monitor(PatientPhysiologyEnv(csv_path, seed=seed))

        vec_env = DummyVecEnv([make_env])
        model = build_model(vec_env, trial, seed)
        model.learn(total_timesteps=timesteps_per_trial)

        metrics = evaluate_policy(model, csv_path)
        return metrics["mean_episode_reward"]

    return objective


def train_and_evaluate_fixed(csv_path: str, params, timesteps: int, seed: int, label: str) -> float:
    def make_env():
        return Monitor(PatientPhysiologyEnv(csv_path, seed=seed))

    vec_env = DummyVecEnv([make_env])

    if params is None:
        model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
    else:
        model = build_model(vec_env, params, seed)

    print(f"Training [{label}] for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    metrics = evaluate_policy(model, csv_path)
    print(f"  -> mean_episode_reward = {metrics['mean_episode_reward']:.3f}")
    return metrics["mean_episode_reward"]


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for the Safe RL controller.")
    parser.add_argument("--csv_path", type=str, default="wesad_physiological_timeline.csv")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--timesteps_per_trial", type=int, default=15000)
    parser.add_argument("--final_timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="hpo_results.csv")
    args = parser.parse_args()

    print(f"Running Optuna search: {args.n_trials} trials x {args.timesteps_per_trial} timesteps each...")
    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(args.csv_path, args.timesteps_per_trial, args.seed), n_trials=args.n_trials)

    print("\n" + "=" * 70)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best trial value (mean_episode_reward): {study.best_value:.3f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(args.output_csv, index=False)
    print(f"\nFull trial history saved to {args.output_csv}")

    print("\n" + "=" * 70)
    print("FINAL CONFIRMATION RUN: best-found hyperparameters vs. stable-baselines3 defaults")
    print("=" * 70)
    best_reward = train_and_evaluate_fixed(
        args.csv_path, study.best_params, args.final_timesteps, args.seed, label="best-found (Optuna)"
    )
    default_reward = train_and_evaluate_fixed(
        args.csv_path, None, args.final_timesteps, args.seed, label="stable-baselines3 defaults"
    )

    print("\n" + "-" * 70)
    print(f"Best-found (Optuna):        mean_episode_reward = {best_reward:.3f}")
    print(f"stable-baselines3 defaults: mean_episode_reward = {default_reward:.3f}")
    difference = best_reward - default_reward
    if abs(difference) < 0.05 * max(abs(default_reward), 1e-6):
        print(f"\nDifference ({difference:+.3f}) is small -- defaults appear near-optimal.")
    else:
        print(f"\nDifference ({difference:+.3f}) is non-trivial -- consider adopting the found config.")


if __name__ == "__main__":
    main()
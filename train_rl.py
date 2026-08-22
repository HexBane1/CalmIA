"""
train_rl.py -- RL Training Loop (Phase 2, Script 2).

Trains a PPO agent (stable-baselines3) on PatientPhysiologyEnv (rl_env.py) to
learn the safety constraints and therapeutic mapping from physiological state
to musical condition-token action. Saves the trained policy to
checkpoints/rl_controller.zip.

The environment is NOT pinned to a single subject during training --
PatientPhysiologyEnv.reset() randomly picks a different subject's timeline
each episode by default, which is what lets the policy generalize across the
population instead of overfitting to one patient's recording.
closed_loop_generate.py pins a single, specific subject instead, which is the
appropriate place to do that, not here.

Usage:
    python train_rl.py --csv_path wesad_physiological_timeline.csv --timesteps 75000
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env import PatientPhysiologyEnv


def main():
    parser = argparse.ArgumentParser(description="Train the Safe RL controller on the physiological timeline.")
    parser.add_argument("--csv_path", type=str, default="wesad_physiological_timeline.csv")
    parser.add_argument(
        "--timesteps", type=int, default=75000,
        help="Total training timesteps (50,000-100,000 recommended).",
    )
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/rl_controller.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def make_env():
        # Monitor wraps the env so stable-baselines3's own logging can report
        # per-episode reward/length (visible in the training printout as
        # ep_rew_mean) -- the main signal for whether the agent is actually
        # learning the safety constraints and therapeutic mapping, not just
        # running without crashing.
        return Monitor(PatientPhysiologyEnv(args.csv_path, seed=args.seed))

    vec_env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed)

    checkpoint_dir = os.path.dirname(args.checkpoint_path) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Periodic intermediate saves during training, in addition to the final
    # save below -- protects against losing progress if a long training run
    # gets interrupted (a real risk already seen earlier in this project with
    # CPU-only training).
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1),
        save_path=os.path.join(checkpoint_dir, "rl_checkpoints"),
        name_prefix="rl_controller_checkpoint",
    )

    print(f"Training PPO for {args.timesteps} timesteps on {args.csv_path}...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    model.save(args.checkpoint_path)
    print(f"\nTraining complete. Final model saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
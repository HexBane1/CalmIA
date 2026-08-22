"""
closed_loop_generate.py -- End-to-End Closed-Loop System (Phase 2, Script 3).

Bridges the trained Safe RL controller (checkpoints/rl_controller.zip) and the
conditioned Transformer generator (Phase 1) into one working system: steps
through a real subject's physiological timeline, lets the RL agent choose a
musical action at each step, converts that action into the matching condition
tokens, and generates a short MIDI sequence conditioned on those tokens for
that time window.

Reuses developer_b's existing, tested modules directly (checkpoint_loader,
sampler, postprocess) rather than duplicating their logic -- only the primer
construction differs from generate_song.py's default (BOS-only) primer, since
here the primer needs to start with the RL-chosen condition tokens instead.

Usage:
    python closed_loop_generate.py --subject S2 \
        --physio_csv wesad_physiological_timeline.csv \
        --rl_checkpoint checkpoints/rl_controller.zip \
        --music_checkpoint checkpoints/checkpoint_best.pt \
        --output_dir closed_loop_output
"""

import argparse
import csv
import os

from stable_baselines3 import PPO

from rl_env import PatientPhysiologyEnv, ACTION_TO_TOKENS
from shared.vocabulary import tempo_condition_token, complexity_condition_token
from developer_b.checkpoint_loader import load_model_from_checkpoint
from developer_b.sampler import generate
from developer_b.postprocess import tokens_to_midi


def action_to_primer(action: int) -> list:
    """
    Converts an RL action (0-3) into the [tempo_token_id, complexity_token_id]
    primer the conditioned Transformer expects at the start of every sequence
    -- the same layout developer_a/dataset.py prepends during training.
    """
    tempo_bin_name, complexity_bin_name = ACTION_TO_TOKENS[action]
    # tempo_condition_token()/complexity_condition_token() expect the exact
    # bin strings label_midi_features.py produces ("TEMPO_SLOW"/"TEMPO_FAST",
    # "COMPLEXITY_LOW"/"COMPLEXITY_HIGH") -- ACTION_TO_TOKENS in rl_env.py
    # already uses this exact naming, so no translation is needed here.
    tempo_token_id = tempo_condition_token(tempo_bin_name)
    complexity_token_id = complexity_condition_token(complexity_bin_name)
    return [tempo_token_id, complexity_token_id]


def main():
    parser = argparse.ArgumentParser(description="Run the closed-loop physiology -> music system on one subject.")
    parser.add_argument("--subject", type=str, default="S2", help="Subject id to run, e.g. S2.")
    parser.add_argument("--physio_csv", type=str, default="wesad_physiological_timeline.csv")
    parser.add_argument("--rl_checkpoint", type=str, default="checkpoints/rl_controller.zip")
    parser.add_argument("--music_checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--output_dir", type=str, default="closed_loop_output")
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Length of each per-window generated MIDI snippet.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading RL controller from {args.rl_checkpoint}...")
    rl_model = PPO.load(args.rl_checkpoint)

    print(f"Loading music generator checkpoint from {args.music_checkpoint}...")
    music_model, metadata = load_model_from_checkpoint(args.music_checkpoint)
    print(f"  (trained through epoch {metadata['epoch']}, val_loss={metadata['val_loss']:.4f})")

    print(f"Loading physiological timeline for subject {args.subject} from {args.physio_csv}...")
    env = PatientPhysiologyEnv(args.physio_csv, subject_id=args.subject)
    obs, info = env.reset(seed=0)

    step_index = 0
    terminated = False
    manifest_rows = []

    while not terminated:
        action, _states = rl_model.predict(obs, deterministic=True)
        action = int(action)
        tempo_bin_name, complexity_bin_name = ACTION_TO_TOKENS[action]

        raw_hrv = info.get("raw_hrv")
        raw_eda = info.get("raw_eda")
        print(
            f"Step {step_index}: HRV_SDNN={raw_hrv:.2f}, Mean_EDA={raw_eda:.3f} "
            f"-> action {action} ({tempo_bin_name}, {complexity_bin_name})"
        )

        primer = action_to_primer(action)
        token_ids = generate(
            music_model,
            primer_ids=primer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        output_filename = f"step{step_index:03d}_{tempo_bin_name}_{complexity_bin_name}.mid"
        output_path = os.path.join(args.output_dir, output_filename)
        tokens_to_midi(token_ids, output_path)
        print(f"  -> generated {output_path} ({len(token_ids)} tokens)")

        manifest_rows.append({
            "step": step_index,
            "raw_hrv": raw_hrv,
            "raw_eda": raw_eda,
            "action": action,
            "tempo_bin": tempo_bin_name,
            "complexity_bin": complexity_bin_name,
            "output_file": output_filename,
        })

        obs, reward, terminated, truncated, info = env.step(action)
        step_index += 1

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nClosed-loop run complete: {step_index} windows generated for subject {args.subject}.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
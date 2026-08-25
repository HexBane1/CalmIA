"""
profile_system.py -- Latency & Memory Cost (Phase 4, Script 2).

Profiles the full closed-loop pipeline (RL controller decision + Transformer
generation) to characterize whether it can run in a real-time clinical
setting, where each 10-second physiological window needs a decision and a
matching music snippet before the next window arrives.

MEASUREMENT DESIGN NOTES (read before trusting the numbers):

1. Both time-per-stage AND time end-to-end are measured, not just a single
   total. "Which stage is the bottleneck" is a more actionable systems
   finding than a single latency number, and is exactly the kind of
   diagnostic a systems-evaluation section should report.

2. Memory: this script uses BOTH psutil and tracemalloc, deliberately, not
   either/or. tracemalloc only tracks pure-Python object allocations -- it
   MISSES memory held by PyTorch tensors and NumPy buffers (C-extension
   memory), which is where the overwhelming majority of this workload's
   actual memory footprint lives. psutil's process RSS captures the real,
   whole-process memory footprint (including that C-extension memory) and is
   the number that actually matters for "can this run on the target
   hardware." tracemalloc is kept as a secondary signal specifically for
   catching Python-level growth (e.g. an accumulating list) that RSS alone
   wouldn't localize as clearly.

3. GPU/VRAM is only reported if CUDA is actually available. This project has
   been running on CPU throughout -- reporting a VRAM number in that case
   would be actively misleading rather than merely absent.

4. A short WARMUP phase runs first and is excluded from the timed
   measurements. The first few calls into a freshly loaded model include
   one-time costs (lazy initialization, allocator warm-up) that are not
   representative of steady-state per-window latency, and including them
   would understate real throughput.

5. Input is fully synthetic by default (random physiological values in a
   plausible range) so this script runs standalone with no data preparation
   -- matching the task's own framing of "simulated inference steps." A real
   CSV can optionally be supplied instead for extra fidelity.

Usage:
    python profile_system.py --num_steps 50 \
        --rl_checkpoint checkpoints/rl_controller.zip \
        --music_checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import time
import tracemalloc
from typing import List, Optional

import numpy as np
import psutil
import torch
from stable_baselines3 import PPO

from rl_env import ACTION_TO_TOKENS
from shared.vocabulary import tempo_condition_token, complexity_condition_token
from developer_b.checkpoint_loader import load_model_from_checkpoint
from developer_b.sampler import generate


def synthetic_observation(rng: np.random.Generator) -> np.ndarray:
    """
    A plausible-range normalized (HRV, EDA) observation. Since this is a
    z-scored space, values roughly in [-2, 2] cover the range the agent was
    actually trained on; sampling outside that would test generalization
    behavior, not typical steady-state latency, which is not what this
    script is measuring.
    """
    return rng.uniform(-2.0, 2.0, size=3).astype(np.float32)


def action_to_primer(action: int) -> list:
    tempo_bin_name, complexity_bin_name = ACTION_TO_TOKENS[action]
    return [tempo_condition_token(tempo_bin_name), complexity_condition_token(complexity_bin_name)]


def percentile(values: List[float], p: float) -> float:
    return float(np.percentile(values, p))


def main():
    parser = argparse.ArgumentParser(description="Profile latency and memory cost of the closed-loop pipeline.")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--rl_checkpoint", type=str, default="checkpoints/rl_controller.zip")
    parser.add_argument("--music_checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Length of each generated MIDI snippet.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    process = psutil.Process()

    print(f"Loading RL controller from {args.rl_checkpoint}...")
    rl_model = PPO.load(args.rl_checkpoint)

    print(f"Loading music generator checkpoint from {args.music_checkpoint}...")
    music_model, metadata = load_model_from_checkpoint(args.music_checkpoint)
    device = next(music_model.parameters()).device
    print(f"  (device: {device}, trained through epoch {metadata['epoch']})")

    # Baseline RSS after both models are loaded but before any inference --
    # this is the "idle" footprint the timed loop's growth is measured
    # against, so results aren't polluted by one-time model-loading cost.
    baseline_rss_mb = process.memory_info().rss / (1024 ** 2)

    def run_one_window():
        """One full simulated window: RL decision -> condition tokens -> generation."""
        obs = synthetic_observation(rng)

        rl_start = time.perf_counter()
        action, _states = rl_model.predict(obs, deterministic=True)
        rl_elapsed = time.perf_counter() - rl_start

        primer = action_to_primer(int(action))

        gen_start = time.perf_counter()
        token_ids = generate(music_model, primer_ids=primer, max_new_tokens=args.max_new_tokens)
        gen_elapsed = time.perf_counter() - gen_start

        return rl_elapsed, gen_elapsed, len(token_ids)

    print(f"\nRunning {args.warmup_steps} warmup step(s) (excluded from measurements)...")
    for _ in range(args.warmup_steps):
        run_one_window()

    print(f"Running {args.num_steps} measured step(s)...\n")
    rl_latencies_ms: List[float] = []
    gen_latencies_ms: List[float] = []
    total_latencies_ms: List[float] = []
    tokens_generated: List[int] = []

    tracemalloc.start()
    peak_rss_mb = baseline_rss_mb

    pipeline_start = time.perf_counter()
    for step in range(args.num_steps):
        step_start = time.perf_counter()
        rl_elapsed, gen_elapsed, num_tokens = run_one_window()
        step_elapsed = time.perf_counter() - step_start

        rl_latencies_ms.append(rl_elapsed * 1000)
        gen_latencies_ms.append(gen_elapsed * 1000)
        total_latencies_ms.append(step_elapsed * 1000)
        tokens_generated.append(num_tokens)

        current_rss_mb = process.memory_info().rss / (1024 ** 2)
        peak_rss_mb = max(peak_rss_mb, current_rss_mb)

    pipeline_elapsed = time.perf_counter() - pipeline_start
    _current_py_mb, peak_py_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_py_mb = peak_py_mb / (1024 ** 2)

    peak_vram_mb: Optional[float] = None
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    total_tokens = sum(tokens_generated)
    windows_per_second = args.num_steps / pipeline_elapsed
    tokens_per_second = total_tokens / pipeline_elapsed

    print("=" * 60)
    print("SYSTEMS PROFILING SUMMARY")
    print("=" * 60)
    print(f"Device:                          {device}")
    print(f"Measured steps:                  {args.num_steps} (+{args.warmup_steps} warmup, excluded)")
    print()
    print("-- Latency (milliseconds) --")
    print(f"  RL decision      : mean={np.mean(rl_latencies_ms):7.2f}  "
          f"p50={percentile(rl_latencies_ms, 50):7.2f}  p95={percentile(rl_latencies_ms, 95):7.2f}")
    print(f"  Music generation : mean={np.mean(gen_latencies_ms):7.2f}  "
          f"p50={percentile(gen_latencies_ms, 50):7.2f}  p95={percentile(gen_latencies_ms, 95):7.2f}")
    print(f"  Total per window : mean={np.mean(total_latencies_ms):7.2f}  "
          f"p50={percentile(total_latencies_ms, 50):7.2f}  p95={percentile(total_latencies_ms, 95):7.2f}")
    print()
    print("-- Throughput --")
    print(f"  Windows/second:                {windows_per_second:.2f}")
    print(f"  Tokens/second:                 {tokens_per_second:.1f}")
    print(f"  Real-time margin (10s budget): {10000 / np.mean(total_latencies_ms):.1f}x "
          f"(>1x means faster than the 10-second window it needs to keep up with)")
    print()
    print("-- Memory --")
    print(f"  Baseline RSS (models loaded, idle): {baseline_rss_mb:.1f} MB")
    print(f"  Peak RSS during inference:           {peak_rss_mb:.1f} MB")
    print(f"  RSS growth over baseline:            {peak_rss_mb - baseline_rss_mb:.1f} MB")
    print(f"  Peak Python-level allocation (tracemalloc): {peak_py_mb:.1f} MB "
          f"(pure-Python objects only -- see note in this script's docstring "
          f"on why this underestimates true memory use for this workload)")
    if peak_vram_mb is not None:
        print(f"  Peak GPU memory allocated:           {peak_vram_mb:.1f} MB")
    else:
        print(f"  Peak GPU memory allocated:           N/A (no CUDA device available -- running on CPU)")
    print("=" * 60)


if __name__ == "__main__":
    main()
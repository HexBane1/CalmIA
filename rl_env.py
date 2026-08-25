"""
rl_env.py -- Safe Patient Environment (Phase 2, Script 1).

Task 4 update: observation expanded from 2D (HRV, EDA) to 3D (HRV, EDA, RESP).
Safety/reward logic still uses only HRV/EDA (documented decision, see below).
Also added a configurable safety_penalty constructor param for run_experiments.py's
Safe-vs-Unsafe ablation.

Usage as a smoke test:
    python rl_env.py path/to/wesad_physiological_timeline.csv
"""

import sys
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

HIGH_STRESS_HRV_THRESHOLD = 50.0
HIGH_STRESS_EDA_THRESHOLD = 2.0

ACTION_TO_TOKENS = {
    0: ("TEMPO_SLOW", "COMPLEXITY_LOW"),
    1: ("TEMPO_SLOW", "COMPLEXITY_HIGH"),
    2: ("TEMPO_FAST", "COMPLEXITY_LOW"),
    3: ("TEMPO_FAST", "COMPLEXITY_HIGH"),
}

OPPOSITE_ACTION = {0: 3, 3: 0, 1: 2, 2: 1}

THERAPEUTIC_REWARD = 1.0
SAFETY_PENALTY = -5.0
CALMING_ACTION = 0
DANGEROUS_ACTION = 3

REQUIRED_COLUMNS = {"Subject", "HRV_SDNN", "Mean_EDA", "Mean_RSP_Rate"}


class PatientPhysiologyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        csv_path: str,
        subject_id: Optional[str] = None,
        seed: Optional[int] = None,
        safety_penalty: float = SAFETY_PENALTY,
    ):
        super().__init__()

        raw = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(
                f"CSV at {csv_path} is missing required column(s): {missing}. "
                f"Re-run extract_features.py to regenerate it with Mean_RSP_Rate."
            )

        raw = raw.dropna(subset=["HRV_SDNN", "Mean_EDA", "Mean_RSP_Rate"]).reset_index(drop=True)

        self.subject_blocks = {
            sid: group.reset_index(drop=True)
            for sid, group in raw.groupby("Subject", sort=False)
            if len(group) >= 2
        }
        if not self.subject_blocks:
            raise ValueError(f"No subject in {csv_path} has at least 2 valid rows.")

        self._fixed_subject_id = subject_id
        if subject_id is not None and subject_id not in self.subject_blocks:
            raise ValueError(
                f"Requested subject_id={subject_id!r} not found among: {sorted(self.subject_blocks.keys())}"
            )

        all_valid = pd.concat(self.subject_blocks.values(), ignore_index=True)
        self._hrv_mean = all_valid["HRV_SDNN"].mean()
        self._hrv_std = all_valid["HRV_SDNN"].std() or 1.0
        self._eda_mean = all_valid["Mean_EDA"].mean()
        self._eda_std = all_valid["Mean_EDA"].std() or 1.0
        self._resp_mean = all_valid["Mean_RSP_Rate"].mean()
        self._resp_std = all_valid["Mean_RSP_Rate"].std() or 1.0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self._safety_penalty = safety_penalty

        self._rng = np.random.default_rng(seed)
        self._current_block = None
        self._current_index = 0
        self._action_history = []

    def _normalize(self, hrv, eda, resp) -> np.ndarray:
        return np.array(
            [
                (hrv - self._hrv_mean) / self._hrv_std,
                (eda - self._eda_mean) / self._eda_std,
                (resp - self._resp_mean) / self._resp_std,
            ],
            dtype=np.float32,
        )

    def _is_high_stress(self, hrv, eda) -> bool:
        return hrv < HIGH_STRESS_HRV_THRESHOLD and eda > HIGH_STRESS_EDA_THRESHOLD

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._fixed_subject_id is not None:
            chosen_subject = self._fixed_subject_id
        else:
            chosen_subject = self._rng.choice(list(self.subject_blocks.keys()))

        self._current_block = self.subject_blocks[chosen_subject]
        self._current_index = 0
        self._action_history = []

        row = self._current_block.iloc[self._current_index]
        observation = self._normalize(row["HRV_SDNN"], row["Mean_EDA"], row["Mean_RSP_Rate"])
        info = {
            "subject": chosen_subject,
            "raw_hrv": row["HRV_SDNN"],
            "raw_eda": row["Mean_EDA"],
            "raw_resp": row["Mean_RSP_Rate"],
        }
        return observation, info

    def _compute_safety_violation(self, action, hrv, eda) -> bool:
        violated = False

        if self._action_history and OPPOSITE_ACTION[action] == self._action_history[-1]:
            violated = True

        if len(self._action_history) >= 2:
            two_steps_ago, one_step_ago = self._action_history[-2], self._action_history[-1]
            if action == two_steps_ago and OPPOSITE_ACTION[one_step_ago] == two_steps_ago:
                violated = True

        if action == DANGEROUS_ACTION and self._is_high_stress(hrv, eda):
            violated = True

        return violated

    def step(self, action: int):
        if self._current_block is None:
            raise RuntimeError("step() called before reset()")

        row = self._current_block.iloc[self._current_index]
        hrv, eda = float(row["HRV_SDNN"]), float(row["Mean_EDA"])

        reward = 0.0
        if action == CALMING_ACTION and self._is_high_stress(hrv, eda):
            reward += THERAPEUTIC_REWARD
        if self._compute_safety_violation(action, hrv, eda):
            reward += self._safety_penalty

        self._action_history.append(action)
        if len(self._action_history) > 2:
            self._action_history.pop(0)

        self._current_index += 1
        terminated = self._current_index >= len(self._current_block)

        if not terminated:
            next_row = self._current_block.iloc[self._current_index]
            observation = self._normalize(
                next_row["HRV_SDNN"], next_row["Mean_EDA"], next_row["Mean_RSP_Rate"]
            )
            info = {
                "raw_hrv": next_row["HRV_SDNN"],
                "raw_eda": next_row["Mean_EDA"],
                "raw_resp": next_row["Mean_RSP_Rate"],
            }
        else:
            observation = self._normalize(hrv, eda, float(row["Mean_RSP_Rate"]))
            info = {"raw_hrv": hrv, "raw_eda": eda, "raw_resp": float(row["Mean_RSP_Rate"])}

        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self):
        pass


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "wesad_physiological_timeline.csv"
    env = PatientPhysiologyEnv(csv_path)

    obs, info = env.reset(seed=0)
    print(f"Episode started on subject {info['subject']}, {len(env._current_block)} steps available")
    print(f"Initial observation (normalized, HRV/EDA/RESP): {obs}")

    total_reward = 0.0
    terminated = False
    step_count = 0
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Random-policy episode finished after {step_count} steps, total reward: {total_reward}")
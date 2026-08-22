"""
rl_env.py -- Safe Patient Environment (Phase 2, Script 1).

A custom gymnasium.Env that treats the aggregated WESAD physiological timeline
(wesad_physiological_timeline.csv, produced by extract_features.py) as an
offline-simulated patient. One episode = one subject's ordered window
sequence, exactly as extract_features.py wrote it (Baseline windows, then
Stress windows, then Meditation windows -- a designed scenario progression,
not a literal continuous recording, since each condition's windows are
extracted and concatenated separately).

State: normalized (HRV_SDNN, Mean_EDA) -- z-scored using dataset-wide
statistics, for numerically stable RL training. The RAW (non-normalized)
values are kept internally for reward computation, since the therapeutic/
safety thresholds below are defined in raw physical units (ms, microsiemens),
matching extract_features.py's calculate_music_targets() thresholds exactly,
for consistency across the pipeline. Both threshold sets are first-pass, not
clinically validated -- see that function's docstring.

Action: Discrete(4), mapping to the four condition-token combinations added
to the vocabulary in Task 2:
    0 = Slow/Low   (TEMPO_SLOW,  COMPLEXITY_LOW)
    1 = Slow/High  (TEMPO_SLOW,  COMPLEXITY_HIGH)
    2 = Fast/Low   (TEMPO_FAST,  COMPLEXITY_LOW)
    3 = Fast/High  (TEMPO_FAST,  COMPLEXITY_HIGH)

Reward:
    - Therapeutic: +1 for choosing action 0 (Slow/Low) while the CURRENT
      state is high-stress (raw HRV_SDNN < HIGH_STRESS_HRV_THRESHOLD and raw
      Mean_EDA > HIGH_STRESS_EDA_THRESHOLD).
    - Safety penalty: -5, applied ONCE per step even if multiple rules below
      trigger simultaneously (see _compute_safety_violation() docstring for
      why this is a flat penalty rather than a stacked/summed one), for any
      of:
        (a) an abrupt single-step transition to the maximally-opposite
            action (0<->3 or 1<->2 -- both tempo AND complexity flip at
            once), or
        (b) a 3-step oscillation: the action returns to what it was two
            steps ago, having gone through that action's maximal opposite
            in between (A -> B -> A), or
        (c) choosing action 3 (Fast/High) -- the single most stimulating
            option -- while the current state is high-stress.
    All other (state, action) combinations receive 0 reward. This is
    intentional: the reward only actively shapes behavior at the extremes
    (the one clearly-correct calming choice under stress, and the clearly
    dangerous/unstable choices), leaving the agent free to explore actions 1
    and 2 during non-stress states without being pushed by a preference I
    have not validated against real outcomes.

Usage as a smoke test:
    python rl_env.py path/to/wesad_physiological_timeline.csv
"""

import sys
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Matches shared.vocabulary's condition-token naming exactly, and the same
# threshold constants used in extract_features.py's calculate_music_targets().
HIGH_STRESS_HRV_THRESHOLD = 50.0  # ms; below this = low HRV
HIGH_STRESS_EDA_THRESHOLD = 2.0   # microsiemens; above this = high arousal

ACTION_TO_TOKENS = {
    0: ("TEMPO_SLOW", "COMPLEXITY_LOW"),
    1: ("TEMPO_SLOW", "COMPLEXITY_HIGH"),
    2: ("TEMPO_FAST", "COMPLEXITY_LOW"),
    3: ("TEMPO_FAST", "COMPLEXITY_HIGH"),
}

# Action pairs considered "maximally opposite" -- both tempo and complexity
# flip simultaneously. Used to detect abrupt single-step transitions and
# 3-step oscillations.
OPPOSITE_ACTION = {0: 3, 3: 0, 1: 2, 2: 1}

THERAPEUTIC_REWARD = 1.0
SAFETY_PENALTY = -5.0
CALMING_ACTION = 0    # Slow/Low
DANGEROUS_ACTION = 3  # Fast/High


class PatientPhysiologyEnv(gym.Env):
    """
    Gymnasium environment simulating a patient's physiological response
    timeline from a pre-extracted WESAD CSV. One episode = one subject's
    full window sequence.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        csv_path: str,
        subject_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        raw = pd.read_csv(csv_path)
        required_columns = {"Subject", "HRV_SDNN", "Mean_EDA"}
        missing = required_columns - set(raw.columns)
        if missing:
            raise ValueError(f"CSV at {csv_path} is missing required column(s): {missing}")

        # Drop rows where extraction failed (NaN HRV/EDA) -- these cannot be
        # used as valid states.
        raw = raw.dropna(subset=["HRV_SDNN", "Mean_EDA"]).reset_index(drop=True)

        # One ordered row-block per subject. Row order within each subject is
        # preserved exactly as extract_features.py wrote it.
        self.subject_blocks = {
            sid: group.reset_index(drop=True)
            for sid, group in raw.groupby("Subject", sort=False)
            if len(group) >= 2  # need at least 2 rows for a meaningful episode
        }
        if not self.subject_blocks:
            raise ValueError(
                f"No subject in {csv_path} has at least 2 valid (non-NaN) rows -- "
                f"nothing usable for an episode."
            )

        self._fixed_subject_id = subject_id
        if subject_id is not None and subject_id not in self.subject_blocks:
            raise ValueError(
                f"Requested subject_id={subject_id!r} not found among usable "
                f"subjects: {sorted(self.subject_blocks.keys())}"
            )

        # Normalization statistics computed across ALL usable rows (not just
        # one subject), so observation scale is consistent regardless of
        # which subject a given episode picks.
        all_valid = pd.concat(self.subject_blocks.values(), ignore_index=True)
        self._hrv_mean = all_valid["HRV_SDNN"].mean()
        self._hrv_std = all_valid["HRV_SDNN"].std() or 1.0  # avoid divide-by-zero
        self._eda_mean = all_valid["Mean_EDA"].mean()
        self._eda_std = all_valid["Mean_EDA"].std() or 1.0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._current_block = None
        self._current_index = 0
        self._action_history = []  # up to the last 2 actions, for transition/oscillation checks

    def _normalize(self, hrv: float, eda: float) -> np.ndarray:
        return np.array(
            [(hrv - self._hrv_mean) / self._hrv_std, (eda - self._eda_mean) / self._eda_std],
            dtype=np.float32,
        )

    def _is_high_stress(self, hrv: float, eda: float) -> bool:
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
        observation = self._normalize(row["HRV_SDNN"], row["Mean_EDA"])
        info = {"subject": chosen_subject, "raw_hrv": row["HRV_SDNN"], "raw_eda": row["Mean_EDA"]}
        return observation, info

    def _compute_safety_violation(self, action: int, hrv: float, eda: float) -> bool:
        """
        Returns True if ANY safety rule is violated this step. Deliberately a
        single boolean rather than a sum of violations: if multiple rules
        trigger at once, SAFETY_PENALTY is still applied exactly once, not
        stacked. This keeps the reward scale bounded and predictable as more
        safety rules are potentially added later, rather than risking large
        negative-reward spikes that can destabilize training.
        """
        violated = False

        # (a) abrupt single-step transition to the maximally-opposite action
        if self._action_history and OPPOSITE_ACTION[action] == self._action_history[-1]:
            violated = True

        # (b) 3-step oscillation: action returns to what it was two steps
        # ago, having gone through that action's maximal opposite in between.
        if len(self._action_history) >= 2:
            two_steps_ago, one_step_ago = self._action_history[-2], self._action_history[-1]
            if action == two_steps_ago and OPPOSITE_ACTION[one_step_ago] == two_steps_ago:
                violated = True

        # (c) prescribing the single most stimulating option to a
        # high-stress patient
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
            reward += SAFETY_PENALTY

        self._action_history.append(action)
        if len(self._action_history) > 2:
            self._action_history.pop(0)

        self._current_index += 1
        terminated = self._current_index >= len(self._current_block)

        if not terminated:
            next_row = self._current_block.iloc[self._current_index]
            observation = self._normalize(next_row["HRV_SDNN"], next_row["Mean_EDA"])
            info = {"raw_hrv": next_row["HRV_SDNN"], "raw_eda": next_row["Mean_EDA"]}
        else:
            # Episode over -- return the last valid observation again.
            # Gymnasium convention: not meant to be acted on further once
            # terminated=True.
            observation = self._normalize(hrv, eda)
            info = {"raw_hrv": hrv, "raw_eda": eda}

        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self):
        pass  # no visualization needed for this offline-simulated environment


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "wesad_physiological_timeline.csv"
    env = PatientPhysiologyEnv(csv_path)

    obs, info = env.reset(seed=0)
    print(f"Episode started on subject {info['subject']}, {len(env._current_block)} steps available")
    print(f"Initial observation (normalized): {obs}")

    total_reward = 0.0
    terminated = False
    step_count = 0
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Random-policy episode finished after {step_count} steps, total reward: {total_reward}")
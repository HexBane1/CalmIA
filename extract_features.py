"""
Extended physiological feature extraction: real-time windowing + target mapping.

Assigned task (see README): converts the static, whole-block HRV/EDA extraction
into a continuous, simulated real-time dataset for the RL controller environment.

Usage:
    python extract_features.py
"""

import pickle
import numpy as np
import neurokit2 as nk
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

SAMPLING_RATE = 700  # WESAD chest sensors sampling rate (Hz)
WINDOW_SECONDS = 10
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLING_RATE  # 7000 samples per window

# WESAD condition labels we care about (label 3 = Amusement is excluded per task spec)
CONDITIONS = {
    1: "Baseline",
    2: "Stress",
    4: "Meditation",
}


def get_windows_for_condition(signal, labels, condition_label, window_samples):
    """
    Slices `signal` into consecutive, non-overlapping windows of `window_samples`
    length, using only the portion where `labels == condition_label`.
    Leftover samples that don't fill a full window are dropped.
    """
    condition_indices = np.where(labels == condition_label)[0]
    if len(condition_indices) == 0:
        return []

    condition_signal = signal[condition_indices]
    num_windows = len(condition_signal) // window_samples

    windows = []
    for i in range(num_windows):
        start = i * window_samples
        end = start + window_samples
        windows.append(condition_signal[start:end])

    return windows


def extract_window_features(ecg_window, eda_window, sampling_rate):
    """
    Processes a single 10s window of raw ECG + EDA and returns (hrv_sdnn, mean_eda).
    Some windows are too short/noisy for neurokit2 to find enough R-peaks -- in that
    case we return NaN for that metric rather than crashing the whole run, since a
    handful of unusable windows out of many is expected and shouldn't halt processing.
    """
    try:
        ecg_signals, ecg_info = nk.ecg_process(ecg_window, sampling_rate=sampling_rate)
        hrv_df = nk.hrv_time(ecg_info, sampling_rate=sampling_rate)
        hrv_sdnn = hrv_df["HRV_SDNN"].iloc[0]
    except Exception:
        hrv_sdnn = np.nan

    try:
        eda_signals, eda_info = nk.eda_process(eda_window, sampling_rate=sampling_rate)
        mean_eda = eda_signals["EDA_Tonic"].mean()
    except Exception:
        mean_eda = np.nan

    return hrv_sdnn, mean_eda


def calculate_music_targets(hrv, eda):
    """
    Rule-based mapping from physiological state to musical targets, per the
    project's safety constraints: high arousal / low HRV -> calmer music
    (lower tempo, lower complexity), to avoid reinforcing a stressed state.

    Thresholds (HRV_SDNN in ms, EDA in microsiemens) are a first-pass baseline,
    not clinically validated -- meant to be tuned once we have more subjects.
    """
    if np.isnan(hrv) or np.isnan(eda):
        return np.nan, np.nan

    high_stress = hrv < 50 and eda > 2.0
    moderate_stress = (hrv < 50) != (eda > 2.0)  # exactly one condition triggered

    if high_stress:
        target_tempo, target_complexity = 70, 0.2
    elif moderate_stress:
        target_tempo, target_complexity = 90, 0.5
    else:
        target_tempo, target_complexity = 110, 0.8

    return target_tempo, target_complexity


def process_subject(subject_file, output_csv):
    print(f"Loading data from {subject_file}...")
    with open(subject_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg_raw = data["signal"]["chest"]["ECG"].flatten()
    eda_raw = data["signal"]["chest"]["EDA"].flatten()
    labels = data["label"].flatten()

    rows = []
    timestamp = 0.0

    for label_value, condition_name in CONDITIONS.items():
        ecg_windows = get_windows_for_condition(ecg_raw, labels, label_value, WINDOW_SAMPLES)
        eda_windows = get_windows_for_condition(eda_raw, labels, label_value, WINDOW_SAMPLES)
        n_windows = min(len(ecg_windows), len(eda_windows))

        print(f"{condition_name}: {n_windows} windows of {WINDOW_SECONDS}s each")

        for i in range(n_windows):
            hrv_sdnn, mean_eda = extract_window_features(
                ecg_windows[i], eda_windows[i], SAMPLING_RATE
            )
            target_tempo, target_complexity = calculate_music_targets(hrv_sdnn, mean_eda)

            rows.append(
                {
                    "Timestamp": timestamp,
                    "Condition_Label": condition_name,
                    "HRV_SDNN": hrv_sdnn,
                    "Mean_EDA": mean_eda,
                    "Target_Tempo": target_tempo,
                    "Target_Complexity": target_complexity,
                }
            )
            timestamp += WINDOW_SECONDS

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows to {output_csv}")
    print(f"Rows with valid (non-NaN) HRV: {df['HRV_SDNN'].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    process_subject("wesad_data/S2/S2.pkl", "S2_physiological_timeline.csv")
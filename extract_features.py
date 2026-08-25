"""
Extended physiological feature extraction: real-time windowing + target mapping.

Task 3 update: scaled across all available WESAD subjects (S2-S17, S12 excluded).
Task 4 update: added real Respiration (RESP) extraction alongside HRV and EDA.

Usage:
    python extract_features.py
"""

import os
import pickle
import numpy as np
import neurokit2 as nk
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

SAMPLING_RATE = 700
WINDOW_SECONDS = 10
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLING_RATE

CONDITIONS = {
    1: "Baseline",
    2: "Stress",
    4: "Meditation",
}

WESAD_ROOT = "wesad_data"
OUTPUT_CSV = "wesad_physiological_timeline.csv"

ALL_SUBJECT_IDS = [n for n in range(2, 18) if n != 12]


def get_windows_for_condition(signal, labels, condition_label, window_samples):
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


def extract_window_features(ecg_window, eda_window, resp_window, sampling_rate):
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

    try:
        # nk.rsp_process() requires multiple full breath cycles and fails on
        # short 10s windows (only 1-2 cycles at normal breathing rate).
        # Instead, we use scipy to find peaks in the raw signal directly and
        # estimate breaths-per-minute from the inter-peak interval -- more
        # robust on short windows than the full neurokit2 pipeline.
        from scipy.signal import find_peaks
        # Normalize signal before peak detection (removes DC offset)
        resp_norm = resp_window - np.mean(resp_window)
        # Min distance between peaks: at least 2s apart (700Hz * 2s = 1400 samples)
        # This prevents detecting noise spikes as breath cycles
        peaks, _ = find_peaks(resp_norm, distance=int(sampling_rate * 2))
        if len(peaks) >= 2:
            # Average interval between peaks -> convert to BPM
            avg_interval_samples = np.mean(np.diff(peaks))
            mean_resp_rate = 60.0 / (avg_interval_samples / sampling_rate)
        else:
            # Fewer than 2 peaks found -- use RMS amplitude as a fallback proxy
            # for respiratory effort (not a rate, but still informative for RL)
            mean_resp_rate = float(np.sqrt(np.mean(resp_norm ** 2)))
    except Exception:
        mean_resp_rate = np.nan

    return hrv_sdnn, mean_eda, mean_resp_rate


def calculate_music_targets(hrv, eda):
    if np.isnan(hrv) or np.isnan(eda):
        return np.nan, np.nan

    high_stress = hrv < 50 and eda > 2.0
    moderate_stress = (hrv < 50) != (eda > 2.0)

    if high_stress:
        target_tempo, target_complexity = 70, 0.2
    elif moderate_stress:
        target_tempo, target_complexity = 90, 0.5
    else:
        target_tempo, target_complexity = 110, 0.8

    return target_tempo, target_complexity


def process_subject(subject_file, subject_id):
    print(f"Loading data from {subject_file}...")
    with open(subject_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg_raw = data["signal"]["chest"]["ECG"].flatten()
    eda_raw = data["signal"]["chest"]["EDA"].flatten()
    resp_raw = data["signal"]["chest"]["Resp"].flatten()
    labels = data["label"].flatten()

    rows = []
    timestamp = 0.0

    for label_value, condition_name in CONDITIONS.items():
        ecg_windows = get_windows_for_condition(ecg_raw, labels, label_value, WINDOW_SAMPLES)
        eda_windows = get_windows_for_condition(eda_raw, labels, label_value, WINDOW_SAMPLES)
        resp_windows = get_windows_for_condition(resp_raw, labels, label_value, WINDOW_SAMPLES)
        n_windows = min(len(ecg_windows), len(eda_windows), len(resp_windows))

        print(f"  [{subject_id}] {condition_name}: {n_windows} windows of {WINDOW_SECONDS}s each")

        for i in range(n_windows):
            hrv_sdnn, mean_eda, mean_resp_rate = extract_window_features(
                ecg_windows[i], eda_windows[i], resp_windows[i], SAMPLING_RATE
            )
            target_tempo, target_complexity = calculate_music_targets(hrv_sdnn, mean_eda)

            rows.append(
                {
                    "Timestamp": timestamp,
                    "Subject": subject_id,
                    "Condition_Label": condition_name,
                    "HRV_SDNN": hrv_sdnn,
                    "Mean_EDA": mean_eda,
                    "Mean_RSP_Rate": mean_resp_rate,
                    "Target_Tempo": target_tempo,
                    "Target_Complexity": target_complexity,
                }
            )
            timestamp += WINDOW_SECONDS

    return pd.DataFrame(rows)


def main():
    all_frames = []

    for subject_num in ALL_SUBJECT_IDS:
        subject_id = f"S{subject_num}"
        subject_file = os.path.join(WESAD_ROOT, subject_id, f"{subject_id}.pkl")

        if not os.path.isfile(subject_file):
            print(f"SKIPPING {subject_id}: file not found at {subject_file}")
            continue

        df = process_subject(subject_file, subject_id)
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError(f"No subject files found under {WESAD_ROOT}/")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(combined)} total rows across {len(all_frames)} subject(s) -> {OUTPUT_CSV}")
    print(f"Rows with valid HRV:  {combined['HRV_SDNN'].notna().sum()}/{len(combined)}")
    print(f"Rows with valid RESP: {combined['Mean_RSP_Rate'].notna().sum()}/{len(combined)}")
    print("\nRows per subject:")
    print(combined["Subject"].value_counts().sort_index())


if __name__ == "__main__":
    main()
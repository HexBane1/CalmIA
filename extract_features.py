import pickle
import numpy as np
import neurokit2 as nk
import pandas as pd
import warnings

# Suppress pandas/neurokit warnings for cleaner output
warnings.filterwarnings("ignore")

def process_subject_data(subject_file):
    print(f"Loading data from {subject_file}...")
    with open(subject_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # WESAD chest sensors are sampled at 700Hz
    sampling_rate = 700 

    # Extract the raw signals and labels from the chest device
    ecg_raw = data["signal"]["chest"]["ECG"].flatten()
    eda_raw = data["signal"]["chest"]["EDA"].flatten()
    labels = data["label"].flatten()

    # WESAD Labels: 1 = Baseline, 2 = Stress, 3 = Amusement, 4 = Meditation
    # We isolate the data where the subject was officially in the "Stress" condition
    stress_indices = np.where(labels == 2)[0]
    
    if len(stress_indices) == 0:
        print("No stress data found for this subject.")
        return

    # Slice the raw signals to only include the stress period
    ecg_stress = ecg_raw[stress_indices]
    eda_stress = eda_raw[stress_indices]

    duration_seconds = len(ecg_stress) / sampling_rate
    print(f"Processing {duration_seconds:.2f} seconds of 'Stress' condition data...")

    # 1. Process ECG to get Heart Rate Variability (HRV)
    print("Extracting ECG features (cleaning signal & finding R-peaks)...")
    ecg_signals, ecg_info = nk.ecg_process(ecg_stress, sampling_rate=sampling_rate)
    
    # Calculate HRV metrics
    hrv_df = nk.hrv(ecg_info, sampling_rate=sampling_rate)
    
    # 2. Process EDA to get Skin Conductance (Arousal/Stress level)
    print("Extracting EDA features (calculating skin conductance)...")
    eda_signals, eda_info = nk.eda_process(eda_stress, sampling_rate=sampling_rate)
    
    # Calculate mean Skin Conductance Level (SCL) - a prime indicator of physiological arousal
    mean_scl = eda_signals["EDA_Tonic"].mean()

    # Output the actionable conditioning variables
    print("\n" + "="*50)
    print(" EXTRACTED PHYSIOLOGICAL CONDITIONING VARIABLES")
    print("="*50)
    print(f"Mean Heart Rate: {ecg_signals['ECG_Rate'].mean():.2f} BPM")
    print(f"HRV (SDNN):      {hrv_df['HRV_SDNN'].iloc[0]:.2f} ms  <-- (Lower generally means higher stress)")
    print(f"Mean EDA (SCL):  {mean_scl:.4f}          <-- (Higher means higher arousal/stress)")
    print(f"EDA Responses:   {len(eda_info['SCR_Peaks'])} peaks          <-- (Number of sweat gland responses)")
    print("="*50)

if __name__ == "__main__":
    # Point this to your existing WESAD subject folder
    process_subject_data("wesad_data/S2/S2.pkl")
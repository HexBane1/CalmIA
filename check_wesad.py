"""
Quick sanity check: confirms a WESAD subject's .pkl file loads correctly and has
the expected structure before building anything on top of it.

Usage:
    python check_wesad.py
"""

import pickle

with open("wesad_data/S2/S2.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Top-level keys:", data.keys())
print("Chest signal keys:", data["signal"]["chest"].keys())
print("Label array shape:", data["label"].shape)
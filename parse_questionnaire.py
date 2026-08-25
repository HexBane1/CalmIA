"""
parse_questionnaire.py -- Patient-Reported Outcome Extraction (Phase 5).

Parses the WESAD per-subject questionnaire CSV (e.g. wesad_data/S2/S2_quest.csv)
and extracts four validated self-report scales:

    PANAS  -- Positive and Negative Affect Schedule (5 items each)
    STAI   -- State-Trait Anxiety Inventory, state subscale (5 items)
    DIM    -- Dimensional affect ratings (Valence, Arousal) -- Baseline + TSST only
    SSSQ   -- Subjective Stress Scale Questionnaire (1 item)

Usage:
    python parse_questionnaire.py
    python parse_questionnaire.py --subject S2
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WESAD_ROOT = "wesad_data"
OUTPUT_CSV = "wesad_questionnaire_summary.csv"

ALL_SUBJECT_IDS = [n for n in range(2, 18) if n != 12]

CONDITIONS = ["Base", "TSST", "Medi1", "Fun", "Medi2", "sRead", "fRead"]

PANAS_ROWS_RAW = [4, 5, 6, 7, 8]
STAI_ROWS_RAW  = [10, 11, 12, 13, 14]
DIM_ROWS_RAW   = [16, 17, 18, 19, 20]
SSSQ_ROW_RAW   = [22]

PANAS_POSITIVE_IDX = [0, 2, 4]
PANAS_NEGATIVE_IDX = [1, 3]


def parse_subject_questionnaire(subject_id: str) -> pd.DataFrame:
    quest_path = os.path.join(WESAD_ROOT, subject_id, f"{subject_id}_quest.csv")
    if not os.path.isfile(quest_path):
        print(f"  SKIPPING {subject_id}: questionnaire not found at {quest_path}")
        return pd.DataFrame()

    raw = pd.read_csv(quest_path, sep=";", header=None)

    n_cols = raw.shape[1]
    available_conditions = CONDITIONS[:min(len(CONDITIONS), n_cols - 1)]

    def get_rows(row_indices):
        rows = raw.iloc[row_indices, 1:len(available_conditions) + 1].copy()
        rows = rows.apply(pd.to_numeric, errors="coerce")
        return rows.values

    panas_vals = get_rows(PANAS_ROWS_RAW)
    stai_vals  = get_rows(STAI_ROWS_RAW)
    dim_vals   = get_rows(DIM_ROWS_RAW)
    sssq_vals  = get_rows(SSSQ_ROW_RAW)

    rows_out = []
    for ci, condition in enumerate(available_conditions):
        panas_pos = np.nanmean(panas_vals[PANAS_POSITIVE_IDX, ci])
        panas_neg = np.nanmean(panas_vals[PANAS_NEGATIVE_IDX, ci])
        stai_total = np.nansum(stai_vals[:, ci])
        dim_valence = dim_vals[0, ci] if ci < dim_vals.shape[1] else np.nan
        dim_arousal = dim_vals[1, ci] if ci < dim_vals.shape[1] else np.nan
        sssq_stress = sssq_vals[0, ci] if not np.all(np.isnan(sssq_vals[:, ci])) else np.nan

        rows_out.append({
            "Subject": subject_id,
            "Condition": condition,
            "PANAS_Positive": panas_pos,
            "PANAS_Negative": panas_neg,
            "STAI_Total": stai_total,
            "DIM_Valence": dim_valence,
            "DIM_Arousal": dim_arousal,
            "SSSQ_Stress": sssq_stress,
        })

    return pd.DataFrame(rows_out)


def main(subject_filter=None):
    all_frames = []

    subject_nums = [int(subject_filter[1:])] if subject_filter else ALL_SUBJECT_IDS
    for sn in subject_nums:
        sid = f"S{sn}"
        print(f"Parsing questionnaire for {sid}...")
        df = parse_subject_questionnaire(sid)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No questionnaire data found.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(combined)} rows to {OUTPUT_CSV}")
    print("\nSample — S2 scores per condition:")
    s2 = combined[combined["Subject"] == "S2"]
    if not s2.empty:
        print(s2[["Condition", "PANAS_Positive", "PANAS_Negative",
                   "STAI_Total", "SSSQ_Stress"]].to_string(index=False))

    print("\nKey sanity check (TSST should show highest stress scores):")
    agg = combined.groupby("Condition")[["PANAS_Negative", "STAI_Total", "SSSQ_Stress"]].mean()
    print(agg.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    args = parser.parse_args()
    main(subject_filter=args.subject)
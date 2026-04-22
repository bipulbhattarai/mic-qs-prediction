#!/usr/bin/env python3
"""
Script 03: Build clean dataset + engineer QS proxy features
Paper: Predicting MIC Rates from QS Features using XGBoost-SHAP
Author: Bipul Bhattarai, University of South Dakota

Input:  data/raw/llm_extracted_records.jsonl
Output: data/processed/mic_qs_dataset.csv
        data/processed/mic_qs_dataset_info.txt
"""

import json, pathlib, sys
import pandas as pd
import numpy as np

IN_FILE   = pathlib.Path("data/raw/llm_extracted_records.jsonl")
OUT_DIR   = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load extracted records ────────────────────────────────────────────
print("=" * 60)
print("Step 3: Building clean dataset")
print("=" * 60)

records = []
with open(IN_FILE) as f:
    for line in f:
        try:
            records.append(json.loads(line.strip()))
        except:
            pass

df = pd.DataFrame(records)
print(f"\nLoaded: {len(df)} records")
print(f"Columns: {list(df.columns)}")

# ── Step 2: Filter usable records ─────────────────────────────────────────────
print("\n[1/6] Filtering usable records...")
df_usable = df[df["data_usable"] == True].copy()
print(f"  Usable: {len(df_usable)} / {len(df)}")

# ── Step 3: Encode categorical columns ────────────────────────────────────────
print("\n[2/6] Encoding categorical features...")

# Metal type → numeric
metal_map = {
    "carbon steel": 1, "mild steel": 1, "cs": 1,
    "stainless steel": 2, "ss": 2, "304": 2, "316": 2,
    "copper": 3, "cu": 3,
    "aluminum": 4, "al": 4, "aluminium": 4,
    "cast iron": 5, "iron": 5,
    "nickel": 6, "inconel": 6,
}

def encode_metal(val):
    if pd.isna(val) or val is None:
        return np.nan
    for key, code in metal_map.items():
        if key in str(val).lower():
            return code
    return 0  # other/unknown

df_usable["metal_code"] = df_usable["metal_type"].apply(encode_metal)

# Environment type → numeric
env_map = {
    "pipeline": 1, "oil": 1, "gas": 1,
    "seawater": 2, "marine": 2, "ocean": 2,
    "freshwater": 3, "river": 3, "lake": 3,
    "wastewater": 4, "sewage": 4, "wwtp": 4,
    "soil": 5, "ground": 5,
    "lab": 6, "laboratory": 6, "in vitro": 6,
}

def encode_env(val):
    if pd.isna(val) or val is None:
        return np.nan
    for key, code in env_map.items():
        if key in str(val).lower():
            return code
    return 0

df_usable["env_code"] = df_usable["environment"].apply(encode_env)

# Biofilm maturity → ordinal
maturity_map = {"early": 1, "mature": 2, "dispersal": 3}
df_usable["biofilm_maturity_code"] = df_usable["biofilm_maturity"].map(maturity_map)

# Corrosion type → binary features
df_usable["is_pitting"] = (df_usable["corrosion_type"].isin(["pitting","both"])).astype(float)

# Boolean columns → numeric
bool_cols = [
    "srb_present", "iob_present", "apb_present", "methanogen_present",
    "biofilm_present", "eps_mentioned",
    "qs_mentioned", "ahl_mentioned", "qs_inhibitor_used",
]
for col in bool_cols:
    if col in df_usable.columns:
        df_usable[col] = pd.to_numeric(
            df_usable[col].map({True:1, False:0, "true":1, "false":0}),
            errors="coerce"
        )

# Study type → numeric
study_map = {"lab": 0, "field": 1, "model": 2}
df_usable["study_type_code"] = df_usable["study_type"].map(study_map)

# ── Step 4: Engineer QS proxy features ───────────────────────────────────────
print("\n[3/6] Engineering QS proxy features...")

def safe_fillna(series, val):
    return series.fillna(val)

# QS Activity Score: composite of QS signals
df_usable["qs_activity_score"] = (
    safe_fillna(df_usable.get("qs_mentioned", pd.Series(0, index=df_usable.index)), 0) * 0.4 +
    safe_fillna(df_usable.get("ahl_mentioned", pd.Series(0, index=df_usable.index)), 0) * 0.4 +
    safe_fillna(df_usable.get("qs_community_fraction", pd.Series(0, index=df_usable.index)), 0) * 0.2
)

# Biofilm Aggressiveness Score
df_usable["biofilm_aggression_score"] = (
    safe_fillna(df_usable.get("srb_present", pd.Series(0, index=df_usable.index)), 0) * 0.35 +
    safe_fillna(df_usable.get("biofilm_present", pd.Series(0, index=df_usable.index)), 0) * 0.25 +
    safe_fillna(df_usable.get("eps_mentioned", pd.Series(0, index=df_usable.index)), 0) * 0.20 +
    safe_fillna(df_usable.get("apb_present", pd.Series(0, index=df_usable.index)), 0) * 0.10 +
    safe_fillna(df_usable.get("iob_present", pd.Series(0, index=df_usable.index)), 0) * 0.10
)

# Community Complexity Score (diversity proxy)
df_usable["community_complexity"] = (
    safe_fillna(df_usable.get("srb_present", pd.Series(0, index=df_usable.index)), 0) +
    safe_fillna(df_usable.get("iob_present", pd.Series(0, index=df_usable.index)), 0) +
    safe_fillna(df_usable.get("apb_present", pd.Series(0, index=df_usable.index)), 0) +
    safe_fillna(df_usable.get("methanogen_present", pd.Series(0, index=df_usable.index)), 0)
)

# Chemical aggressiveness score
df_usable["chem_stress_score"] = (
    (safe_fillna(df_usable.get("h2s_mg_l", pd.Series(0, index=df_usable.index)), 0) / 100).clip(0,1) * 0.4 +
    (safe_fillna(df_usable.get("sulfate_mg_l", pd.Series(0, index=df_usable.index)), 0) / 2000).clip(0,1) * 0.3 +
    ((7 - safe_fillna(df_usable.get("ph", pd.Series(7, index=df_usable.index)), 7)).clip(0,7) / 7) * 0.3
)

# Synergy score: QS × Biofilm (interaction)
df_usable["qs_biofilm_synergy"] = (
    df_usable["qs_activity_score"] * df_usable["biofilm_aggression_score"]
)

print(f"  Engineered features: qs_activity_score, biofilm_aggression_score, "
      f"community_complexity, chem_stress_score, qs_biofilm_synergy")

# ── Step 5: Target variable — MIC rate ────────────────────────────────────────
print("\n[4/6] Processing target variable (MIC rate)...")

# Convert MIC rate category to ordinal if numeric rate missing
def category_to_rate(row):
    """Fill missing mic_rate_um_per_year from category."""
    val = row.get("mic_rate_um_per_year")
    if pd.notna(val) and val is not None:
        try:
            return float(val)
        except:
            pass
    cat = row.get("mic_rate_category")
    # Midpoint of category range
    cat_map = {"low": 5.0, "medium": 55.0, "high": 200.0}
    return cat_map.get(str(cat).lower() if cat else "", np.nan)

df_usable["mic_rate"] = df_usable.apply(category_to_rate, axis=1)

# Log-transform MIC rate (right-skewed distribution)
df_usable["mic_rate_log"] = np.log1p(df_usable["mic_rate"].clip(lower=0))

print(f"  Records with MIC rate: {df_usable['mic_rate'].notna().sum()}")
if df_usable["mic_rate"].notna().sum() > 0:
    print(f"  MIC rate range: {df_usable['mic_rate'].min():.1f} – "
          f"{df_usable['mic_rate'].max():.1f} μm/yr")
    print(f"  MIC rate mean:  {df_usable['mic_rate'].mean():.1f} μm/yr")

# ── Step 6: Select final feature set ─────────────────────────────────────────
print("\n[5/6] Selecting final features...")

FEATURES = [
    # QS features (the key novelty)
    "qs_mentioned", "ahl_mentioned", "qs_community_fraction",
    "qs_activity_score", "qs_biofilm_synergy",
    # Biofilm features
    "srb_present", "iob_present", "apb_present", "methanogen_present",
    "biofilm_present", "eps_mentioned", "biofilm_maturity_code",
    "biofilm_aggression_score", "community_complexity",
    # Environmental
    "temperature_c", "ph", "sulfate_mg_l", "h2s_mg_l",
    "do_mg_l", "salinity_ppt", "chem_stress_score",
    # Diversity
    "shannon_diversity",
    # Material & context
    "metal_code", "env_code", "is_pitting",
    "study_type_code", "exposure_days",
    # Target
    "mic_rate", "mic_rate_log",
]

# Keep only columns that exist
available = [c for c in FEATURES if c in df_usable.columns]
df_final = df_usable[["pmid","title","year","journal"] + available].copy()

print(f"  Final features: {len(available)-2}")  # -2 for targets
print(f"  Final records:  {len(df_final)}")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
print("\n[6/6] Saving dataset...")

out_csv = OUT_DIR / "mic_qs_dataset.csv"
df_final.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")

# Dataset info
info_file = OUT_DIR / "mic_qs_dataset_info.txt"
with open(info_file, "w") as f:
    f.write(f"MIC-QS Dataset Summary\n")
    f.write(f"Generated: {pd.Timestamp.now()}\n")
    f.write(f"{'='*50}\n\n")
    f.write(f"Total records: {len(df_final)}\n")
    f.write(f"Records with MIC rate: {df_final['mic_rate'].notna().sum()}\n\n")
    f.write(f"Feature coverage:\n")
    for feat in available:
        if feat not in ("mic_rate", "mic_rate_log"):
            pct = df_final[feat].notna().mean()*100
            f.write(f"  {feat:<35} {pct:5.1f}% filled\n")
    f.write(f"\nMIC rate statistics:\n")
    f.write(str(df_final["mic_rate"].describe()))

print(f"  Saved: {info_file}")

# Print summary
print(f"\n{'='*60}")
print(f"✅ Dataset built successfully")
print(f"   Records: {len(df_final)}")
print(f"   Features: {len(available)-2}")
print(f"   Output: {out_csv}")

print(f"\nFeature coverage (top 10):")
for feat in available[:10]:
    if feat not in ("mic_rate", "mic_rate_log"):
        pct = df_final[feat].notna().mean()*100
        print(f"  {feat:<35} {pct:5.1f}%")

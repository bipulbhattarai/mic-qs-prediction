#!/usr/bin/env python3
"""
Script 03d: Verify dataset integrity before model training
Author: Bipul Bhattarai, University of South Dakota
"""

import pandas as pd
import numpy as np

DATA = "data/processed/mic_qs_dataset.csv"

FEATURES = [
    "qs_mentioned","ahl_mentioned","qs_community_fraction",
    "qs_activity_score","qs_biofilm_synergy",
    "srb_present","iob_present","apb_present","methanogen_present",
    "biofilm_present","eps_mentioned","biofilm_maturity_code",
    "biofilm_aggression_score","community_complexity",
    "temperature_c","ph","sulfate_mg_l","h2s_mg_l",
    "do_mg_l","salinity_ppt","chem_stress_score",
    "shannon_diversity","metal_code","env_code","is_pitting",
    "study_type_code","exposure_days",
]
TARGETS = ["mic_rate","mic_rate_log","mic_class"]

print("="*55)
print("Dataset Integrity Check")
print("="*55)

df = pd.read_csv(DATA)
print(f"\n✅ File loaded: {len(df)} rows, {len(df.columns)} columns")

# ── 1. Check all required columns exist ──────────────────
print(f"\n[1/5] Checking required columns...")
missing_cols = [c for c in FEATURES + TARGETS if c not in df.columns]
if missing_cols:
    print(f"  ❌ MISSING columns: {missing_cols}")
else:
    print(f"  ✅ All {len(FEATURES)} feature columns present")
    print(f"  ✅ All {len(TARGETS)} target columns present")

# ── 2. Check target variable ──────────────────────────────
print(f"\n[2/5] Checking target variable (mic_rate)...")
n_with_rate = df["mic_rate"].notna().sum()
n_missing    = df["mic_rate"].isna().sum()
print(f"  Records with MIC rate : {n_with_rate}")
print(f"  Records missing rate  : {n_missing}")
print(f"  Min  : {df['mic_rate'].min():.1f} μm/yr")
print(f"  Max  : {df['mic_rate'].max():.1f} μm/yr")
print(f"  Mean : {df['mic_rate'].mean():.1f} μm/yr")
if n_with_rate < 20:
    print(f"  ⚠️  WARNING: Only {n_with_rate} records — model may be unreliable")
else:
    print(f"  ✅ Sufficient records for training")

# ── 3. Check class balance ────────────────────────────────
print(f"\n[3/5] Checking class balance (mic_class)...")
high = int(df["mic_class"].sum())
low  = len(df) - high
ratio = high/len(df)
print(f"  High MIC (>50 μm/yr) : {high} ({ratio*100:.0f}%)")
print(f"  Low  MIC (<50 μm/yr) : {low}  ({(1-ratio)*100:.0f}%)")
if ratio > 0.85 or ratio < 0.15:
    print(f"  ⚠️  WARNING: Imbalanced classes — consider class_weight")
else:
    print(f"  ✅ Acceptable class balance")

# ── 4. Check feature coverage ────────────────────────────
print(f"\n[4/5] Feature coverage (% non-null)...")
problems = []
for feat in FEATURES:
    if feat not in df.columns:
        continue
    pct = df[feat].notna().mean() * 100
    status = "✅" if pct >= 50 else ("⚠️ " if pct >= 20 else "❌")
    if pct < 20:
        problems.append(feat)
    print(f"  {status} {feat:<35} {pct:5.1f}%")

# ── 5. Check for data type issues ────────────────────────
print(f"\n[5/5] Checking data types and NaN extremes...")
for feat in FEATURES:
    if feat not in df.columns:
        continue
    col = pd.to_numeric(df[feat], errors="coerce")
    n_bad = col.isna().sum() - df[feat].isna().sum()
    if n_bad > 0:
        print(f"  ❌ {feat}: {n_bad} non-numeric values")

# Check for inf
numeric_cols = [c for c in FEATURES if c in df.columns]
df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
n_inf = np.isinf(df_num.values).sum()
n_nan = df_num.isna().sum().sum()
print(f"  Total NaN  in features: {n_nan}")
print(f"  Total Inf  in features: {n_inf}")

if n_inf > 0:
    print(f"  ❌ Inf values found — will cause model errors")
else:
    print(f"  ✅ No Inf values")

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*55}")
if missing_cols or n_inf > 0 or n_with_rate < 10:
    print("❌ ISSUES FOUND — fix before running script 04")
else:
    print("✅ Dataset looks good — safe to run script 04!")
    print(f"\nCommand: python3 scripts/04_xgboost_shap.py")

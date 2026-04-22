#!/usr/bin/env python3
"""
Script 03c: Merge manually curated records into main dataset
Author: Bipul Bhattarai, University of South Dakota
"""

import pathlib, pandas as pd, numpy as np

MANUAL  = pathlib.Path("data/raw/manual_records.txt")
OUT_DIR = pathlib.Path("data/processed")

def yesno(v):
    return 1 if str(v).strip().lower() in ("yes","true","1") else 0

def safe_float(v, default=None):
    try: return float(str(v).split()[0].replace("~",""))
    except: return default

def parse_pipe(line, key):
    if "|" in str(line):
        parts = {p.split(":")[0].strip().lower(): p.split(":")[1].strip().lower()
                 for p in str(line).split("|") if ":" in p}
        return parts.get(key, "no")
    return str(line).lower()

metal_map = {
    "carbon steel":1,"mild steel":1,"x65":1,"x70":1,"x80":1,
    "c1018":1,"q235":1,"api":1,"steel":1,
    "13cr":2,"stainless":2,"ss":2,
    "copper":3,"aluminum":4,"aluminium":4,"cast iron":5,
}
env_map = {
    "pipeline":1,"seawater":2,"marine":2,
    "freshwater":3,"river":3,
    "wastewater":4,"sewage":4,
    "soil":5,"ground":5,"lab":6,
}

# ── Parse manual records ──────────────────────────────────────────────────────
records = []
current = {}
with open(MANUAL) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line == "---":
            if current.get("mic") or current.get("mic_rate"):
                records.append(current.copy())
            current = {}
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            current[key.strip().lower().replace(" ","_")] = val.strip()
if current.get("mic") or current.get("mic_rate"):
    records.append(current)

print(f"Parsed {len(records)} manual records")

# ── Load existing dataset first to get exact columns ─────────────────────────
df_existing = pd.read_csv(OUT_DIR / "mic_qs_dataset.csv")
print(f"Existing records : {len(df_existing)}")
print(f"Existing columns : {list(df_existing.columns)}")

# ── Convert manual records using SAME columns as existing dataset ─────────────
rows = []
for r in records:
    srb_line = r.get("srb","no")
    srb  = yesno(parse_pipe(srb_line,"srb"))
    iob  = yesno(parse_pipe(srb_line,"iob"))
    apb  = yesno(parse_pipe(srb_line,"apb"))
    meth = yesno(parse_pipe(srb_line,"meth"))

    bio_line = r.get("biofilm","no")
    biofilm  = yesno(parse_pipe(bio_line,"biofilm"))
    eps      = yesno(parse_pipe(bio_line,"eps")) if "|" in str(bio_line) else biofilm

    qs_line = r.get("qs","no")
    qs  = yesno(parse_pipe(qs_line,"qs"))
    ahl = yesno(parse_pipe(qs_line,"ahl"))

    pit = yesno(r.get("pitting", r.get("pit","no")))
    mic = safe_float(r.get("mic_rate") or r.get("mic"))
    if not mic:
        continue

    temp = safe_float(r.get("temp") or r.get("temperature_c"), 30)
    ph   = safe_float(r.get("ph"), 7.0)
    sulf = safe_float(r.get("sulfate") or r.get("sulfate_mg_l"), 500)
    h2s  = safe_float(r.get("h2s") or r.get("h2s_mg_l"), 0)
    do   = safe_float(r.get("do") or r.get("do_mg_l"), 0)
    sal  = safe_float(r.get("salinity") or r.get("sal") or r.get("salinity_ppt"), 0)
    days = safe_float(r.get("days") or r.get("exposure_days"), 14)

    metal_str = str(r.get("metal","carbon steel")).lower()
    metal_code = 1
    for k,v in metal_map.items():
        if k in metal_str: metal_code = v; break

    env_str = str(r.get("environment","lab")).lower()
    env_code = 6
    for k,v in env_map.items():
        if k in env_str: env_code = v; break

    study_str  = str(r.get("study","lab")).lower()
    study_code = 0 if "lab" in study_str else (1 if "field" in study_str else 2)

    qs_activity = qs*0.4 + ahl*0.4
    bio_agg     = srb*0.35 + biofilm*0.25 + eps*0.20 + apb*0.10 + iob*0.10
    complexity  = srb + iob + apb + meth
    chem_stress = (h2s/100)*0.4 + (sulf/2000)*0.3 + ((7-ph)/7)*0.3
    synergy     = qs_activity * bio_agg

    # Build row with NaN for any column not in manual data
    row = {col: np.nan for col in df_existing.columns}

    # Fill known columns
    row.update({
        "metal_code"              : metal_code,
        "srb_present"             : srb,
        "iob_present"             : iob,
        "apb_present"             : apb,
        "methanogen_present"      : meth,
        "biofilm_present"         : biofilm,
        "eps_mentioned"           : eps,
        "qs_mentioned"            : qs,
        "ahl_mentioned"           : ahl,
        "temperature_c"           : temp,
        "ph"                      : ph,
        "sulfate_mg_l"            : sulf,
        "h2s_mg_l"                : h2s,
        "do_mg_l"                 : do,
        "salinity_ppt"            : sal,
        "env_code"                : env_code,
        "study_type_code"         : study_code,
        "exposure_days"           : days,
        "is_pitting"              : pit,
        "qs_activity_score"       : qs_activity,
        "biofilm_aggression_score": bio_agg,
        "community_complexity"    : complexity,
        "chem_stress_score"       : chem_stress,
        "qs_biofilm_synergy"      : synergy,
        "mic_rate"                : mic,
        "mic_rate_log"            : np.log1p(mic),
        "mic_class"               : int(mic > 50),
    })
    rows.append(row)

df_manual = pd.DataFrame(rows, columns=df_existing.columns)
print(f"\nManual records converted : {len(df_manual)}")

# ── Merge and deduplicate ─────────────────────────────────────────────────────
df_combined = pd.concat([df_existing, df_manual], ignore_index=True)
before = len(df_combined)
df_combined = df_combined.drop_duplicates(
    subset=["mic_rate","metal_code","srb_present","env_code","exposure_days"]
)
print(f"Duplicates removed       : {before - len(df_combined)}")

df_combined.to_csv(OUT_DIR / "mic_qs_dataset.csv", index=False)

print(f"\n{'='*50}")
print(f"✅ Dataset saved: {OUT_DIR}/mic_qs_dataset.csv")
print(f"   Total records : {len(df_combined)}")
print(f"   High MIC >50  : {int(df_combined['mic_class'].sum())}")
print(f"   qs_mentioned  : {int(df_combined['qs_mentioned'].sum())}/{len(df_combined)}")
print(f"   srb_present   : {int(df_combined['srb_present'].sum())}/{len(df_combined)}")
print(f"\nMIC rate stats:")
print(f"   Min  : {df_combined['mic_rate'].min():.1f} μm/yr")
print(f"   Max  : {df_combined['mic_rate'].max():.1f} μm/yr")
print(f"   Mean : {df_combined['mic_rate'].mean():.1f} μm/yr")

# MIC-QS Prediction: XGBoost-SHAP Framework

**Paper:** "Predicting Microbiologically Influenced Corrosion Rates from  
Quorum Sensing Biofilm Community Features: An XGBoost-SHAP Framework"

**Author:** Bipul Bhattarai, Dept. of Biomedical Engineering, University of South Dakota  
**Target venue:** IEEE BIBM 2026

---

## Quick Start (macOS M4)

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
chmod +x run_all.sh
./run_all.sh
```

**Estimated total cost:** ~$0.10–0.20 USD (Claude Haiku API calls)  
**Estimated runtime:** ~15–30 min

---

## Pipeline Steps

| Script | What it does | Runtime |
|--------|-------------|---------|
| `01_fetch_pubmed.py` | Fetch ~300 MIC paper abstracts from PubMed | 2–3 min |
| `02_llm_extract.py` | Extract structured features using Claude API | 5–10 min |
| `03_build_dataset.py` | Clean + engineer QS proxy features | <1 min |
| `04_xgboost_shap.py` | Train models + generate all 5 figures | 2–5 min |

---

## Key Novelty

**First paper to use Quorum Sensing (QS) signals as features for MIC rate prediction.**

Prior ML papers for MIC used only:
- Physical params (resistivity, moisture, pH)
- Chemical params (sulfate, H₂S, organic carbon)
- No microbial community / QS features

This paper adds:
- QS activity score (AHL proxy, QS fraction)
- QS-biofilm synergy feature
- SRB/IOB/APB community composition
- SHAP explainability showing QS as top predictor

---

## Feature Categories

| Category | Features | Color in figures |
|----------|---------|-----------------|
| **QS (novelty)** | qs_mentioned, ahl_mentioned, qs_activity_score, qs_biofilm_synergy | 🔵 Blue |
| **Biofilm** | srb_present, biofilm_present, eps_mentioned, biofilm_aggression_score | 🟠 Orange |
| **Environmental** | temperature, pH, sulfate, H₂S, DO | 🔷 Light blue |
| **Material** | metal_type, environment, exposure_days | ⬜ Light |

---

## Expected Results

| Model | Regression R² | Classification F1 |
|-------|-------------|-----------------|
| **XGBoost** | **~0.78–0.85** | **~0.82–0.88** |
| Random Forest | ~0.72–0.80 | ~0.78–0.84 |
| SVM | ~0.60–0.70 | ~0.70–0.78 |
| MLP | ~0.65–0.75 | ~0.72–0.80 |

SHAP analysis expected to rank `qs_activity_score` and `srb_present` in top 3.

---

## Figures Generated

1. `Fig1_model_comparison.png` — XGBoost vs RF vs SVM vs MLP
2. `Fig2_shap_beeswarm_reg.png` — SHAP beeswarm (all features)
3. `Fig3_shap_importance_reg.png` — Mean |SHAP| bar chart
4. `Fig4_shap_qs_dependence.png` — QS activity score dependence plot
5. `Fig5_predicted_vs_actual.png` — Predicted vs actual MIC rates

---

## Connection to Prior Work

| Paper | Connection |
|-------|-----------|
| Paper 1 (QS biofilm DT, IEEE BIBM 2025) | QS features and dataset source |
| Paper 2 (LSTM QS community DT) | QS community fraction features |
| XGBoost corrosion paper (IEEE BIBM 2023) | Same XGBoost framework |
| LLM domain terms paper | LLM extraction pipeline (Script 02) |

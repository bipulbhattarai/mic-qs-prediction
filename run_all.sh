#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full pipeline for MIC-QS XGBoost paper
# Paper: Predicting MIC Rates from QS Features using XGBoost-SHAP
# Author: Bipul Bhattarai, University of South Dakota
#
# Tested on: macOS Apple Silicon (M4)
# Usage:
#   chmod +x run_all.sh
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   ./run_all.sh
# =============================================================================
set -e

echo "=============================================="
echo "MIC-QS XGBoost Paper Pipeline"
echo "Bipul Bhattarai | University of South Dakota"
echo "=============================================="
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
python3 --version || { echo "ERROR: python3 not found"; exit 1; }

# ── Check API key ─────────────────────────────────────────────────────────────
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    echo "  Run: export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi
echo "✅ API key set"

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo ">>> Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# ── Step 1: Fetch PubMed abstracts ───────────────────────────────────────────
echo ""
echo ">>> Step 1: Fetching PubMed abstracts..."
python3 scripts/01_fetch_pubmed.py
echo "✅ Step 1 complete"

# ── Step 2: LLM extraction ───────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Extracting features with Claude API..."
echo "    (This uses claude-haiku-4-5, ~$0.05-0.20 USD total)"
python3 scripts/02_llm_extract.py
echo "✅ Step 2 complete"

# ── Step 3: Build dataset ────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Building and cleaning dataset..."
python3 scripts/03_build_dataset.py
echo "✅ Step 3 complete"

# ── Step 4: XGBoost + SHAP ───────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Training XGBoost + SHAP analysis..."
python3 scripts/04_xgboost_shap.py
echo "✅ Step 4 complete"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Pipeline complete!"
echo ""
echo "Outputs:"
echo "  data/processed/mic_qs_dataset.csv    ← Final dataset"
echo "  data/processed/model_results.json    ← All metrics"
echo "  models/xgb_mic_regression.json       ← Saved model"
echo "  figures/Fig1_model_comparison.png"
echo "  figures/Fig2_shap_beeswarm_reg.png"
echo "  figures/Fig3_shap_importance_reg.png"
echo "  figures/Fig4_shap_qs_dependence.png"
echo "  figures/Fig5_predicted_vs_actual.png"
echo "=============================================="

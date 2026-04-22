#!/usr/bin/env python3
"""
Script 04: Train XGBoost model + SHAP analysis for MIC prediction
Paper: Predicting MIC Rates from QS Features using XGBoost-SHAP
Author: Bipul Bhattarai, University of South Dakota

Optimized for macOS Apple Silicon (M4) — uses device='cpu' for XGBoost.
Input:  data/processed/mic_qs_dataset.csv
Output: models/, figures/, data/processed/model_results.json
"""

import json, pathlib, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import shap

DATA_FILE  = pathlib.Path("data/processed/mic_qs_dataset.csv")
MODEL_DIR  = pathlib.Path("models")
FIG_DIR    = pathlib.Path("figures")
PROC_DIR   = pathlib.Path("data/processed")
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── Feature groups ────────────────────────────────────────────────────────────
QS_FEATURES = [
    "qs_mentioned", "ahl_mentioned", "qs_community_fraction",
    "qs_activity_score", "qs_biofilm_synergy",
]
BIOFILM_FEATURES = [
    "srb_present", "iob_present", "apb_present", "methanogen_present",
    "biofilm_present", "eps_mentioned", "biofilm_maturity_code",
    "biofilm_aggression_score", "community_complexity",
]
ENV_FEATURES = [
    "temperature_c", "ph", "sulfate_mg_l", "h2s_mg_l",
    "do_mg_l", "salinity_ppt", "chem_stress_score",
    "shannon_diversity",
]
MATERIAL_FEATURES = [
    "metal_code", "env_code", "is_pitting",
    "study_type_code", "exposure_days",
]
ALL_FEATURES = QS_FEATURES + BIOFILM_FEATURES + ENV_FEATURES + MATERIAL_FEATURES

COLORS = {
    "xgboost"       : "#2166ac",
    "random_forest" : "#4dac26",
    "svm"           : "#d01c8b",
    "mlp"           : "#f1a340",
    "qs"            : "#2166ac",
    "biofilm"       : "#f4a582",
    "environmental" : "#92c5de",
    "material"      : "#d1e5f0",
}

def load_data():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded: {len(df)} records, {len(df.columns)} columns")
    return df

def prepare_regression(df):
    """Prepare data for MIC rate regression."""
    df_reg = df[df["mic_rate"].notna()].copy()
    print(f"\nRegression subset: {len(df_reg)} records with MIC rate")

    X = df_reg[[c for c in ALL_FEATURES if c in df_reg.columns]].copy()
    y = df_reg["mic_rate_log"].values

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imp, y, df_reg

def prepare_classification(df):
    """Prepare data for high/low MIC classification."""
    df_cls = df[df["mic_rate"].notna()].copy()

    # Binary: high MIC (>50 μm/yr) vs low/medium (<50)
    df_cls["mic_class"] = (df_cls["mic_rate"] > 50).astype(int)
    print(f"Classification subset: {len(df_cls)} records")
    print(f"  High MIC (>50 μm/yr): {df_cls['mic_class'].sum()}")
    print(f"  Low MIC  (<50 μm/yr): {(~df_cls['mic_class'].astype(bool)).sum()}")

    X = df_cls[[c for c in ALL_FEATURES if c in df_cls.columns]].copy()
    y = df_cls["mic_class"].values

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imp, y

def train_and_compare_regression(X, y):
    """Train XGBoost vs baselines for regression."""
    print("\n" + "="*55)
    print("REGRESSION: Predicting MIC rate (log μm/yr)")
    print("="*55)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            device="cpu", verbosity=0
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "SVM": SVR(kernel="rbf", C=10, epsilon=0.1),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128,64,32), max_iter=500,
            random_state=42, early_stopping=True
        ),
    }

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # 5-fold CV
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)

        # Train/test
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)

        results[name] = {
            "r2"     : round(r2, 4),
            "rmse"   : round(rmse, 4),
            "mae"    : round(mae, 4),
            "cv_r2"  : round(cv_scores.mean(), 4),
            "cv_std" : round(cv_scores.std(), 4),
        }
        print(f"\n{name}:")
        print(f"  Test R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
        print(f"  CV R²={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return models["XGBoost"], X_tr, X_te, y_tr, y_te, results

def train_and_compare_classification(X, y):
    """Train XGBoost vs baselines for classification."""
    print("\n" + "="*55)
    print("CLASSIFICATION: High vs Low MIC (>50 μm/yr threshold)")
    print("="*55)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
            device="cpu", verbosity=0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", C=10, probability=True, random_state=42),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128,64,32), max_iter=500,
            random_state=42, early_stopping=True
        ),
    }

    cls_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=skf,
                                    scoring="f1_weighted", n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:,1] if hasattr(model,"predict_proba") else y_pred

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="weighted")
        try:
            auc = roc_auc_score(y_te, y_prob)
        except:
            auc = 0.0

        cls_results[name] = {
            "accuracy": round(acc, 4),
            "f1"      : round(f1, 4),
            "auc"     : round(auc, 4),
            "cv_f1"   : round(cv_scores.mean(), 4),
            "cv_std"  : round(cv_scores.std(), 4),
        }
        print(f"\n{name}:")
        print(f"  Acc={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
        print(f"  CV F1={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return models["XGBoost"], X_tr, X_te, y_tr, y_te, cls_results

def plot_model_comparison(reg_results, cls_results):
    """Figure 1: Model comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(reg_results.keys())
    colors = [COLORS.get(n.lower().replace(" ","_"),"#999") for n in names]

    # Regression: R²
    ax = axes[0]
    r2_vals = [reg_results[n]["cv_r2"] for n in names]
    errs    = [reg_results[n]["cv_std"] for n in names]
    bars = ax.bar(names, r2_vals, yerr=errs, color=colors,
                  edgecolor="black", linewidth=0.7, capsize=5)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("R² (5-fold CV)", fontsize=12)
    ax.set_title("Regression: MIC Rate Prediction\n(5-fold Cross-Validation)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)

    # Classification: F1
    ax = axes[1]
    f1_vals = [cls_results[n]["cv_f1"] for n in names]
    errs    = [cls_results[n]["cv_std"] for n in names]
    bars = ax.bar(names, f1_vals, yerr=errs, color=colors,
                  edgecolor="black", linewidth=0.7, capsize=5)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Weighted F1 (5-fold CV)", fontsize=12)
    ax.set_title("Classification: High vs Low MIC\n(5-fold Cross-Validation)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model Performance Comparison\n(XGBoost-SHAP Framework for MIC Prediction)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"Fig1_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig1: Model comparison")

def plot_shap_beeswarm(model, X_te, feature_names, tag="reg"):
    """Figure 2: SHAP beeswarm plot."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_te)

    # Color features by category
    def feat_color(name):
        if name in QS_FEATURES:      return COLORS["qs"]
        if name in BIOFILM_FEATURES: return COLORS["biofilm"]
        if name in ENV_FEATURES:     return COLORS["environmental"]
        return COLORS["material"]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_te, feature_names=feature_names,
                      show=False, plot_size=None, max_display=15)
    plt.title(f"SHAP Feature Importance — MIC {'Rate' if tag=='reg' else 'Classification'}\n"
              f"(XGBoost, n={len(X_te)} test samples)",
              fontsize=12, fontweight="bold")

    # Add legend for feature categories
    patches = [
        mpatches.Patch(color=COLORS["qs"],          label="QS features"),
        mpatches.Patch(color=COLORS["biofilm"],      label="Biofilm features"),
        mpatches.Patch(color=COLORS["environmental"],label="Environmental"),
        mpatches.Patch(color=COLORS["material"],     label="Material/context"),
    ]
    plt.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"Fig2_shap_beeswarm_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Fig2: SHAP beeswarm ({tag})")
    return shap_vals, explainer

def plot_shap_bar(shap_vals, feature_names, tag="reg"):
    """Figure 3: SHAP mean absolute importance bar chart."""
    mean_shap = np.abs(shap_vals).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_shap
    }).sort_values("importance", ascending=True).tail(15)

    # Color by category
    def cat_color(name):
        if name in QS_FEATURES:      return COLORS["qs"]
        if name in BIOFILM_FEATURES: return COLORS["biofilm"]
        if name in ENV_FEATURES:     return COLORS["environmental"]
        return COLORS["material"]

    colors = [cat_color(f) for f in feat_imp["feature"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(feat_imp["feature"], feat_imp["importance"],
                   color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, feat_imp["importance"]):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    patches = [
        mpatches.Patch(color=COLORS["qs"],          label="QS features ★"),
        mpatches.Patch(color=COLORS["biofilm"],      label="Biofilm features"),
        mpatches.Patch(color=COLORS["environmental"],label="Environmental"),
        mpatches.Patch(color=COLORS["material"],     label="Material/context"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Feature Importance for MIC Prediction\n(SHAP mean absolute values)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"Fig3_shap_importance_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Fig3: SHAP importance bar ({tag})")
    return feat_imp

def plot_qs_dependence(shap_vals, X_te, feature_names):
    """Figure 4: SHAP dependence plot for QS activity score."""
    qs_feats = [f for f in QS_FEATURES if f in feature_names]
    if not qs_feats:
        print("  No QS features found for dependence plot")
        return

    target_feat = "qs_activity_score" if "qs_activity_score" in feature_names else qs_feats[0]
    feat_idx = list(feature_names).index(target_feat)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        X_te.iloc[:,feat_idx],
        shap_vals[:,feat_idx],
        c=X_te.iloc[:,feat_idx],
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="none",
        s=40,
    )
    plt.colorbar(sc, ax=ax, label=target_feat)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(target_feat.replace("_"," ").title(), fontsize=12)
    ax.set_ylabel("SHAP value (impact on MIC rate)", fontsize=12)
    ax.set_title(f"SHAP Dependence: {target_feat}\n"
                 f"(Higher QS activity → Higher MIC rate?)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"Fig4_shap_qs_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig4: SHAP QS dependence plot")

def plot_predicted_vs_actual(model, X_te, y_te):
    """Figure 5: Predicted vs actual MIC rate."""
    y_pred = model.predict(X_te)

    # Convert back from log scale
    y_te_orig   = np.expm1(y_te)
    y_pred_orig = np.expm1(y_pred)

    r2 = r2_score(y_te, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Log scale
    ax = axes[0]
    ax.scatter(y_te, y_pred, alpha=0.6, color=COLORS["xgboost"],
               edgecolors="white", linewidth=0.3, s=50)
    mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
    ax.plot([mn,mx],[mn,mx],"r--",linewidth=1.5,label="Perfect prediction")
    ax.set_xlabel("Actual MIC rate (log μm/yr)", fontsize=11)
    ax.set_ylabel("Predicted MIC rate (log μm/yr)", fontsize=11)
    ax.set_title(f"Log Scale\nR² = {r2:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Original scale
    ax = axes[1]
    ax.scatter(y_te_orig, y_pred_orig, alpha=0.6, color=COLORS["xgboost"],
               edgecolors="white", linewidth=0.3, s=50)
    mn2 = min(y_te_orig.min(), y_pred_orig.min())
    mx2 = max(y_te_orig.max(), y_pred_orig.max())
    ax.plot([mn2,mx2],[mn2,mx2],"r--",linewidth=1.5,label="Perfect prediction")
    ax.set_xlabel("Actual MIC rate (μm/yr)", fontsize=11)
    ax.set_ylabel("Predicted MIC rate (μm/yr)", fontsize=11)
    ax.set_title("Original Scale", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("XGBoost: Predicted vs Actual MIC Rates\n(Test Set)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"Fig5_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig5: Predicted vs actual")

def main():
    print("="*60)
    print("Step 4: XGBoost + SHAP MIC Prediction")
    print("="*60)

    df = load_data()

    # ── Regression ────────────────────────────────────────────────────────────
    X_reg, y_reg, df_reg = prepare_regression(df)

    if len(X_reg) < 10:
        print("\n⚠️  WARNING: Very few records with MIC rate.")
        print("   Run scripts 01 and 02 first to build a larger dataset.")
        print("   Demonstrating with synthetic data for pipeline testing...\n")
        # Generate synthetic demo data for pipeline testing
        np.random.seed(42)
        n = 200
        X_reg = pd.DataFrame({
            "qs_mentioned"           : np.random.binomial(1,0.6,n),
            "ahl_mentioned"          : np.random.binomial(1,0.4,n),
            "qs_community_fraction"  : np.random.beta(2,5,n),
            "qs_activity_score"      : np.random.beta(2,3,n),
            "qs_biofilm_synergy"     : np.random.beta(1,4,n),
            "srb_present"            : np.random.binomial(1,0.7,n),
            "iob_present"            : np.random.binomial(1,0.3,n),
            "apb_present"            : np.random.binomial(1,0.2,n),
            "methanogen_present"     : np.random.binomial(1,0.15,n),
            "biofilm_present"        : np.random.binomial(1,0.85,n),
            "eps_mentioned"          : np.random.binomial(1,0.5,n),
            "biofilm_maturity_code"  : np.random.randint(1,4,n).astype(float),
            "biofilm_aggression_score": np.random.beta(3,2,n),
            "community_complexity"   : np.random.randint(0,5,n).astype(float),
            "temperature_c"          : np.random.normal(30,10,n).clip(4,80),
            "ph"                     : np.random.normal(7,1,n).clip(4,9),
            "sulfate_mg_l"           : np.random.exponential(500,n).clip(0,3000),
            "h2s_mg_l"               : np.random.exponential(10,n).clip(0,100),
            "do_mg_l"                : np.random.exponential(2,n).clip(0,15),
            "salinity_ppt"           : np.random.exponential(5,n).clip(0,40),
            "chem_stress_score"      : np.random.beta(2,3,n),
            "shannon_diversity"      : np.random.normal(2.5,0.8,n).clip(0,5),
            "metal_code"             : np.random.choice([1,2,3,4],n).astype(float),
            "env_code"               : np.random.choice([1,2,3,4,5,6],n).astype(float),
            "is_pitting"             : np.random.binomial(1,0.4,n).astype(float),
            "study_type_code"        : np.random.choice([0,1,2],n).astype(float),
            "exposure_days"          : np.random.exponential(30,n).clip(1,365),
        })
        # Synthetic MIC rate: driven by QS and SRB (true relationship)
        mic_true = (
            50 * X_reg["srb_present"] +
            30 * X_reg["qs_activity_score"] +
            25 * X_reg["biofilm_aggression_score"] +
            15 * X_reg["chem_stress_score"] +
            10 * (1 - X_reg["ph"]/14) +
            np.random.normal(0, 15, n)
        ).clip(0.1, 500)
        y_reg = np.log1p(mic_true)
        print("  [DEMO] Using synthetic dataset (200 samples)")
        print("  [DEMO] Replace with real extracted data after running scripts 01-02\n")

    xgb_reg, X_tr, X_te, y_tr, y_te, reg_results = \
        train_and_compare_regression(X_reg, y_reg)

    # ── Classification ────────────────────────────────────────────────────────
    X_cls, y_cls = prepare_classification(df) if len(df[df["mic_rate"].notna()]) >= 10 \
                   else (X_reg.copy(), (np.expm1(y_reg) > 50).astype(int))

    xgb_cls, X_tr_c, X_te_c, y_tr_c, y_te_c, cls_results = \
        train_and_compare_classification(X_cls, y_cls)

    # ── Save models ───────────────────────────────────────────────────────────
    xgb_reg.save_model(str(MODEL_DIR/"xgb_mic_regression.json"))
    xgb_cls.save_model(str(MODEL_DIR/"xgb_mic_classification.json"))
    print(f"\n✅ Models saved to {MODEL_DIR}/")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_model_comparison(reg_results, cls_results)

    feature_names = list(X_reg.columns)
    shap_vals, explainer = plot_shap_beeswarm(xgb_reg, X_te, feature_names, tag="reg")
    feat_imp = plot_shap_bar(shap_vals, feature_names, tag="reg")
    plot_qs_dependence(shap_vals, X_te, feature_names)
    plot_predicted_vs_actual(xgb_reg, X_te, y_te)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "regression"     : reg_results,
        "classification" : cls_results,
        "top_features"   : feat_imp.tail(10)[["feature","importance"]].to_dict("records"),
        "n_samples"      : len(X_reg),
        "n_features"     : len(feature_names),
        "qs_features"    : [f for f in QS_FEATURES if f in feature_names],
    }
    with open(PROC_DIR/"model_results.json","w") as f:
        json.dump(results, f, indent=2)

    # ── Print key results ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("KEY RESULTS SUMMARY")
    print("="*55)
    print(f"\nRegression (MIC rate prediction):")
    for name, res in reg_results.items():
        print(f"  {name:<18} R²={res['cv_r2']:.3f}±{res['cv_std']:.3f}")

    print(f"\nClassification (High vs Low MIC):")
    for name, res in cls_results.items():
        print(f"  {name:<18} F1={res['cv_f1']:.3f}±{res['cv_std']:.3f} "
              f"| AUC={res['auc']:.3f}")

    print(f"\nTop 5 features by SHAP importance:")
    for _, row in feat_imp.tail(5).iloc[::-1].iterrows():
        cat = "★ QS" if row["feature"] in QS_FEATURES else "   "
        print(f"  {cat} {row['feature']:<35} {row['importance']:.4f}")

    print(f"\n✅ All figures saved to: {FIG_DIR}/")
    print(f"✅ Results saved to: {PROC_DIR}/model_results.json")

if __name__ == "__main__":
    main()

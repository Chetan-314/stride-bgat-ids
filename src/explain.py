from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.features import STRIDEFeaturePipeline
from src.stride_map import STRIDE_CODES, STRIDE_LABELS

SEED = 42


def _compute_dim_shap(clf, X_bg: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Computes SHAP matrix (N_eval, F) with automatic fallback for XGBoost >=3.0 JSON base_score."""
    try:
        clf.set_param({"device": "cpu"})
    except Exception:
        pass

    try:
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(X_eval)
        if isinstance(vals, list):
            vals = vals[1]
        return np.asarray(vals, dtype=np.float32)
    except Exception:
        pred_fn = lambda x: clf.predict_proba(x)[:, 1]
        explainer = shap.Explainer(pred_fn, X_bg, max_evals=450)
        exp = explainer(X_eval[:250])
        return np.asarray(exp.values, dtype=np.float32)


def run_shap_analysis(root_dir: Path = Path(".")):
    rng = np.random.default_rng(SEED)
    proc_dir = root_dir / "data" / "processed"
    art_dir = root_dir / "outputs" / "artifacts"
    fig_dir = root_dir / "outputs" / "figures"
    tab_dir = root_dir / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    setattr(sys.modules[__name__], "STRIDEFeaturePipeline", STRIDEFeaturePipeline)
    X_test = np.load(proc_dir / "X_test.npy")
    pipeline = joblib.load(proc_dir / "feature_pipeline.joblib")
    models = joblib.load(art_dir / "xgb_multilabel_models.joblib")
    feature_names = pipeline.get_feature_names()

    bg_idx = rng.choice(len(X_test), size=min(100, len(X_test)), replace=False)
    eval_idx = rng.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    X_bg = X_test[bg_idx]
    X_eval = X_test[eval_idx]

    mean_abs_shap = np.zeros((len(feature_names), 6), dtype=np.float32)
    top_rows = []

    print("Computing cross-STRIDE SHAP attributions...")
    for j, (code, label) in enumerate(zip(STRIDE_CODES, STRIDE_LABELS)):
        shap_vals = _compute_dim_shap(models[j], X_bg, X_eval)
        col_imp = np.mean(np.abs(shap_vals), axis=0)
        mean_abs_shap[:, j] = col_imp

        top_idx = int(np.argmax(col_imp))
        top_feat = feature_names[top_idx]
        top_rows.append(
            {
                "Dimension": f"{label} ({code})",
                "Top_Feature": top_feat,
                "Engineered": "Yes" if top_feat.startswith("feat_") else "No",
                "Mean_Abs_SHAP": round(float(col_imp[top_idx]), 4),
            }
        )
        print(f"  [{code}] Top feature: {top_feat} (mean |SHAP| = {col_imp[top_idx]:.4f})")

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(tab_dir / "shap_top_features.csv", index=False)

    # Select top 18 overall features across all 6 STRIDE dimensions for heatmap
    total_imp = mean_abs_shap.sum(axis=1)
    top18_idx = np.argsort(total_imp)[::-1][:18]
    display_names = [
        f"* {feature_names[i]}" if feature_names[i].startswith("feat_") else feature_names[i]
        for i in top18_idx
    ]
    heatmap_df = pd.DataFrame(mean_abs_shap[top18_idx], index=display_names, columns=STRIDE_LABELS)

    plt.figure(figsize=(9.5, 7.5), dpi=300)
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": "Mean |SHAP value|"})
    plt.title("Cross-STRIDE Feature Importance Heatmap (* = Engineered Behavioural Signal)", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved SHAP heatmap to {fig_dir / 'shap_heatmap.png'} and rankings to {tab_dir / 'shap_top_features.csv'}")


if __name__ == "__main__":
    run_shap_analysis()
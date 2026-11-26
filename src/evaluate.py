import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    roc_auc_score,
    roc_curve,
)

from src.stride_map import STRIDE_CODES, STRIDE_LABELS


def mcnemar_pvalue(y_true_1d: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    """Computes continuity-corrected McNemar chi2 and p-value comparing model B against baseline A."""
    a_correct = pred_a == y_true_1d
    b_correct = pred_b == y_true_1d
    b_wins = int(np.sum((~a_correct) & b_correct))
    a_wins = int(np.sum(a_correct & (~b_correct)))
    denom = b_wins + a_wins
    if denom == 0:
        return 0.0, 1.0
    stat = ((abs(b_wins - a_wins) - 1.0) ** 2) / float(denom)
    pval = float(chi2.sf(stat, df=1))
    return stat, pval


def _eval_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict:
    return {
        "Configuration": name,
        "Macro_F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Micro_F1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "Hamming_Loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "Exact_Match": round(float(accuracy_score(y_true, y_pred)), 4),
        "Macro_AUROC": round(float(roc_auc_score(y_true, y_prob, average="macro")), 4) if y_prob is not None else "N/A",
    }


def run_full_evaluation(root_dir: Path = Path(".")):
    proc_dir = root_dir / "data" / "processed"
    art_dir = root_dir / "outputs" / "artifacts"
    fig_dir = root_dir / "outputs" / "figures"
    tab_dir = root_dir / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    y_test = np.load(proc_dir / "y_test.npy")
    P_single = np.load(art_dir / "P_single_xgb_test.npy")
    P_xgb = np.load(art_dir / "P_xgb_test.npy")
    P_gat = np.load(art_dir / "P_gat_test.npy")
    P_fused = np.load(art_dir / "P_fused_test.npy")

    with open(art_dir / "fusion_params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    tau_xgb = np.asarray(params["tau_xgb"], dtype=np.float32)
    tau_gat = np.asarray(params["tau_gat"], dtype=np.float32)
    tau_fused = np.asarray(params["tau_fused"], dtype=np.float32)

    P_late_avg = 0.5 * (P_xgb + P_gat)

    Y_xgb_def = (P_xgb >= 0.5).astype(np.float32)
    Y_xgb_cal = (P_xgb >= tau_xgb).astype(np.float32)
    Y_gat_def = (P_gat >= 0.5).astype(np.float32)
    Y_gat_cal = (P_gat >= tau_gat).astype(np.float32)
    Y_late = (P_late_avg >= 0.5).astype(np.float32)
    Y_fused_def = (P_fused >= 0.5).astype(np.float32)
    Y_fused_cal = (P_fused >= tau_fused).astype(np.float32)

    # 1. Ablation Comparison Table
    ablation_rows = [
        _eval_row("Single-Label XGBoost (Baseline)", y_test, P_single, None),
        _eval_row("Static Multi-Label XGBoost (tau=0.50)", y_test, Y_xgb_def, P_xgb),
        _eval_row("Static Multi-Label XGBoost (Calibrated tau*)", y_test, Y_xgb_cal, P_xgb),
        _eval_row("Standalone Behavioural k-NN GAT (tau=0.50)", y_test, Y_gat_def, P_gat),
        _eval_row("Standalone Behavioural k-NN GAT (Calibrated tau*)", y_test, Y_gat_cal, P_gat),
        _eval_row("Naive Late Averaging (XGB + B-GAT)", y_test, Y_late, P_late_avg),
        _eval_row("Logit-Gated B-GAT Fusion (tau=0.50)", y_test, Y_fused_def, P_fused),
        _eval_row("Logit-Gated B-GAT Fusion + Calibrated tau* (Ours)", y_test, Y_fused_cal, P_fused),
    ]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(tab_dir / "ablation_results.csv", index=False)

    # 2. Per-STRIDE Dimension Breakdown & McNemar's Test
    dim_rows = []
    for j, (code, label) in enumerate(zip(STRIDE_CODES, STRIDE_LABELS)):
        auc_j = roc_auc_score(y_test[:, j], P_fused[:, j])
        f1_single = f1_score(y_test[:, j], P_single[:, j], zero_division=0)
        f1_xgb_c = f1_score(y_test[:, j], Y_xgb_cal[:, j], zero_division=0)
        f1_ours = f1_score(y_test[:, j], Y_fused_cal[:, j], zero_division=0)
        _, p_vs_single = mcnemar_pvalue(y_test[:, j], P_single[:, j], Y_fused_cal[:, j])
        _, p_vs_xgb = mcnemar_pvalue(y_test[:, j], Y_xgb_cal[:, j], Y_fused_cal[:, j])

        dim_rows.append(
            {
                "STRIDE_Dim": f"{label} ({code})",
                "Gate_w_XGB": params["xgb_gate_weights"][j],
                "Tau_Star": float(tau_fused[j]),
                "AUROC": round(float(auc_j), 4),
                "F1_Single_Base": round(float(f1_single), 4),
                "F1_Static_XGB": round(float(f1_xgb_c), 4),
                "F1_Ours": round(float(f1_ours), 4),
                "p_vs_Single": "<0.0001" if p_vs_single < 1e-4 else f"{p_vs_single:.4f}",
                "p_vs_XGB": "<0.0001" if p_vs_xgb < 1e-4 else f"{p_vs_xgb:.4f}",
            }
        )
    dim_df = pd.DataFrame(dim_rows)
    dim_df.to_csv(tab_dir / "per_stride_results.csv", index=False)

    # Export LaTeX tables for paper
    with open(tab_dir / "paper_tables.tex", "w", encoding="utf-8") as f:
        f.write("% Table II: Ablation Results\n")
        f.write(ablation_df.to_latex(index=False))
        f.write("\n\n% Table III: Per-STRIDE Dimension Performance & McNemar's Test\n")
        f.write(dim_df.to_latex(index=False))

    # 3. Plot 6-Panel ROC Curves
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=300)
    for j, (ax, label) in enumerate(zip(axes.flat, STRIDE_LABELS)):
        fpr, tpr, _ = roc_curve(y_test[:, j], P_fused[:, j])
        auc_val = roc_auc_score(y_test[:, j], P_fused[:, j])
        ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle(f"Per-STRIDE ROC Curves — Logit-Gated B-GAT Fusion (Mean AUROC = {roc_auc_score(y_test, P_fused, average='macro'):.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curves_per_dimension.png", dpi=300)
    plt.close()

    # 4. Plot 6-Panel Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=300)
    for j, (ax, label) in enumerate(zip(axes.flat, STRIDE_LABELS)):
        cm = confusion_matrix(y_test[:, j], Y_fused_cal[:, j], labels=[0, 1])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                color = "white" if cm_norm[r, c] > 0.5 else "black"
                ax.text(c, r, f"{cm_norm[r, c]*100:.1f}%\n({cm[r, c]:,})", ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
        ax.set_title(f"{label} (tau*={tau_fused[j]:.2f})", fontsize=10, fontweight="bold")
    plt.suptitle("Threshold-Calibrated Confusion Matrices per STRIDE Dimension", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrices.png", dpi=300)
    plt.close()

    print("\n=== ABLATION COMPARISON TABLE ===")
    print(ablation_df.to_string(index=False))
    print("\n=== PER-STRIDE DIMENSION & MCNEMAR TABLE ===")
    print(dim_df.to_string(index=False))


if __name__ == "__main__":
    run_full_evaluation()
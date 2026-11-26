import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score

from src.stride_map import STRIDE_CODES

SEED = 42
EPS = 1e-6


def _to_logit(p: np.ndarray) -> torch.Tensor:
    t = torch.clamp(torch.tensor(p, dtype=torch.float32), EPS, 1.0 - EPS)
    return torch.logit(t)


def tune_thresholds_val(P_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Grid-searches optimal per-STRIDE decision thresholds strictly on validation data."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_taus = np.full(6, 0.50, dtype=np.float32)
    for j in range(6):
        best_f1 = -1.0
        for tau in grid:
            preds = (P_val[:, j] >= tau).astype(np.float32)
            score = f1_score(y_val[:, j], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_taus[j] = float(tau)
    return best_taus


def run_fusion_and_calibration(root_dir: Path = Path(".")):
    torch.manual_seed(SEED)
    proc_dir = root_dir / "data" / "processed"
    art_dir = root_dir / "outputs" / "artifacts"

    y_val = np.load(proc_dir / "y_val.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    P_xgb_val = np.load(art_dir / "P_xgb_val.npy")
    P_xgb_test = np.load(art_dir / "P_xgb_test.npy")
    P_gat_val = np.load(art_dir / "P_gat_val.npy")
    P_gat_test = np.load(art_dir / "P_gat_test.npy")

    # Learn per-STRIDE logit gate weights + calibration offset strictly on validation split
    lx_v, lg_v = _to_logit(P_xgb_val), _to_logit(P_gat_val)
    yv_t = torch.tensor(y_val, dtype=torch.float32)

    alpha = nn.Parameter(torch.zeros(6))
    beta = nn.Parameter(torch.zeros(6))
    opt = torch.optim.Adam([alpha, beta], lr=0.05)
    for _ in range(500):
        w = torch.sigmoid(alpha)
        fused_logits = w * lx_v + (1.0 - w) * lg_v + beta
        loss = F.binary_cross_entropy_with_logits(fused_logits, yv_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w_opt = torch.sigmoid(alpha)
        P_fused_val = torch.sigmoid(w_opt * lx_v + (1.0 - w_opt) * lg_v + beta).numpy().astype(np.float32)
        lx_t, lg_t = _to_logit(P_xgb_test), _to_logit(P_gat_test)
        P_fused_test = torch.sigmoid(w_opt * lx_t + (1.0 - w_opt) * lg_t + beta).numpy().astype(np.float32)

    # Validation-only threshold calibration for each model family
    tau_xgb = tune_thresholds_val(P_xgb_val, y_val)
    tau_gat = tune_thresholds_val(P_gat_val, y_val)
    tau_fused = tune_thresholds_val(P_fused_val, y_val)

    Y_pred_fused_test = (P_fused_test >= tau_fused).astype(np.float32)

    np.save(art_dir / "P_fused_val.npy", P_fused_val)
    np.save(art_dir / "P_fused_test.npy", P_fused_test)
    np.save(art_dir / "Y_pred_fused_test.npy", Y_pred_fused_test)

    params = {
        "stride_codes": STRIDE_CODES,
        "xgb_gate_weights": [round(float(x), 4) for x in w_opt.detach().numpy()],
        "bgat_gate_weights": [round(float(1.0 - x), 4) for x in w_opt.detach().numpy()],
        "logit_bias": [round(float(x), 4) for x in beta.detach().numpy()],
        "tau_xgb": [round(float(x), 2) for x in tau_xgb],
        "tau_gat": [round(float(x), 2) for x in tau_gat],
        "tau_fused": [round(float(x), 2) for x in tau_fused],
    }
    with open(art_dir / "fusion_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    macro_f1 = f1_score(y_test, Y_pred_fused_test, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, Y_pred_fused_test, average="micro", zero_division=0)
    h_loss = hamming_loss(y_test, Y_pred_fused_test)
    macro_auc = roc_auc_score(y_test, P_fused_test, average="macro")

    print("Logit-Gated Fusion & Validation Calibration Complete:")
    print(f"  Learned XGBoost gate weights {STRIDE_CODES}: {params['xgb_gate_weights']}")
    print(f"  Learned logit bias offsets   {STRIDE_CODES}: {params['logit_bias']}")
    print(f"  Calibrated thresholds tau*   {STRIDE_CODES}: {params['tau_fused']}")
    print(f"  Test Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Hamming: {h_loss:.4f} | Macro AUROC: {macro_auc:.4f}")


if __name__ == "__main__":
    run_fusion_and_calibration()
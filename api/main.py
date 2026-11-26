from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import BEHAVIOURAL_COLS, STRIDEFeaturePipeline
from src.models.bgat import STRIDEBehaviouralGAT
from src.stride_map import STRIDE_CODES, STRIDE_LABELS

ROOT_DIR = Path(".")
PROC_DIR = ROOT_DIR / "data" / "processed"
ART_DIR = ROOT_DIR / "outputs" / "artifacts"
EPS = 1e-6

state: dict[str, Any] = {}


def _to_logit(p: torch.Tensor) -> torch.Tensor:
    p_clamped = torch.clamp(p, EPS, 1.0 - EPS)
    return torch.logit(p_clamped)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setattr(sys.modules["__main__"], "STRIDEFeaturePipeline", STRIDEFeaturePipeline)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline: STRIDEFeaturePipeline = joblib.load(PROC_DIR / "feature_pipeline.joblib")
    xgb_models = joblib.load(ART_DIR / "xgb_multilabel_models.joblib")
    for clf in xgb_models:
        try:
            clf.set_params(device="cpu")
        except Exception:
            pass

    ckpt = torch.load(ART_DIR / "bgat.pt", map_location=device, weights_only=True)
    bgat = STRIDEBehaviouralGAT(
        in_dim=ckpt["in_dim"],
        d_model=ckpt["d_model"],
        num_labels=ckpt["num_labels"],
        k=ckpt["k"],
    ).to(device)
    bgat.load_state_dict(ckpt["state_dict"])
    bgat.eval()

    with open(ART_DIR / "fusion_params.json", "r", encoding="utf-8") as f:
        fusion_params = json.load(f)

    # Keep 256 validation flows as background graph context for small-batch/single-flow queries
    X_val = np.load(PROC_DIR / "X_val.npy")
    ref_context = torch.from_numpy(X_val[:256]).to(device)

    state["device"] = device
    state["pipeline"] = pipeline
    state["feature_names"] = pipeline.get_feature_names()
    state["xgb_models"] = xgb_models
    state["bgat"] = bgat
    state["fusion_params"] = fusion_params
    state["ref_context"] = ref_context
    yield
    state.clear()


app = FastAPI(
    title="Adaptive STRIDE B-GAT Intrusion Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


class FlowBatchRequest(BaseModel):
    flows: list[dict[str, Any]]


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(state["device"]),
        "stride_dimensions": STRIDE_LABELS,
        "xgb_gate_weights": state["fusion_params"]["xgb_gate_weights"],
        "bgat_gate_weights": state["fusion_params"]["bgat_gate_weights"],
        "calibrated_thresholds": state["fusion_params"]["tau_fused"],
    }


@app.post("/predict")
def predict_flows(payload: FlowBatchRequest):
    if not payload.flows:
        raise HTTPException(status_code=400, detail="Request 'flows' list cannot be empty.")

    t0 = time.perf_counter()
    pipeline: STRIDEFeaturePipeline = state["pipeline"]
    xgb_models = state["xgb_models"]
    bgat: STRIDEBehaviouralGAT = state["bgat"]
    device: torch.device = state["device"]
    params = state["fusion_params"]
    feature_names: list[str] = state["feature_names"]

    raw_df = pd.DataFrame(payload.flows)
    X = pipeline.transform(raw_df)
    n_flows = len(X)

    # 1. Static XGBoost probabilities (CPU)
    P_xgb = np.zeros((n_flows, 6), dtype=np.float32)
    for j, clf in enumerate(xgb_models):
        P_xgb[:, j] = clf.predict_proba(X)[:, 1]

    # 2. Behavioural k-NN GAT probabilities (GPU with 256-flow reference neighborhood)
    with torch.no_grad():
        x_t = torch.from_numpy(X).to(device)
        x_graph = torch.cat([x_t, state["ref_context"]], dim=0)
        P_gat = torch.sigmoid(bgat(x_graph)[:n_flows]).cpu().numpy().astype(np.float32)

    # 3. Bias-Calibrated Logit-Gated Fusion + Validation Thresholding
    w_xgb = torch.tensor(params["xgb_gate_weights"], dtype=torch.float32)
    bias = torch.tensor(params.get("logit_bias", [0.0] * 6), dtype=torch.float32)
    tau = np.asarray(params["tau_fused"], dtype=np.float32)

    fused_logits = (
        w_xgb * _to_logit(torch.from_numpy(P_xgb))
        + (1.0 - w_xgb) * _to_logit(torch.from_numpy(P_gat))
        + bias
    )
    P_fused = torch.sigmoid(fused_logits).numpy().astype(np.float32)
    active_mask = (P_fused >= tau).astype(int)

    # 4. Exact TreeSHAP local feature attributions per flow
    dmat = xgb.DMatrix(X)
    shap_per_dim = []
    for clf in xgb_models:
        contribs = clf.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
        shap_per_dim.append(contribs)
    shap_stack = np.stack(shap_per_dim, axis=1)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    results = []
    for i in range(n_flows):
        probs_i = P_fused[i]
        active_i = active_mask[i]
        primary_dim = int(np.argmax(probs_i))

        dim_shap = shap_stack[i, primary_dim]
        top_idx = np.argsort(np.abs(dim_shap))[::-1][:5]
        top_drivers = [
            {
                "feature": feature_names[idx],
                "engineered": feature_names[idx] in BEHAVIOURAL_COLS,
                "shap_value": round(float(dim_shap[idx]), 4),
            }
            for idx in top_idx
        ]

        results.append(
            {
                "flow_index": i,
                "is_threat": bool(active_i.sum() > 0),
                "active_stride_labels": [STRIDE_LABELS[j] for j in range(6) if active_i[j] == 1],
                "stride_vector": active_i.tolist(),
                "fused_probabilities": {STRIDE_CODES[j]: round(float(probs_i[j]), 4) for j in range(6)},
                "xgb_probabilities": {STRIDE_CODES[j]: round(float(P_xgb[i, j]), 4) for j in range(6)},
                "bgat_probabilities": {STRIDE_CODES[j]: round(float(P_gat[i, j]), 4) for j in range(6)},
                "primary_threat": STRIDE_LABELS[primary_dim],
                "top_shap_drivers": top_drivers,
                "engineered_signals": {
                    col: round(float(X[i, -8 + k]), 4) for k, col in enumerate(BEHAVIOURAL_COLS)
                },
            }
        )

    return {
        "batch_size": n_flows,
        "total_latency_ms": round(latency_ms, 2),
        "ms_per_flow": round(latency_ms / n_flows, 3),
        "predictions": results,
    }
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
SAMPLE_CSV = Path("data/processed/demo_sample_flows.csv")

STRIDE_FULL = [
    ("S", "Spoofing"),
    ("T", "Tampering"),
    ("R", "Repudiation"),
    ("I", "Info Disclosure"),
    ("D", "Denial of Service"),
    ("E", "Elevation of Privilege"),
]

st.set_page_config(page_title="STRIDE B-GAT SOC Triage", layout="wide")
st.title("Adaptive Multi-Label Network Intrusion Detection (STRIDE + B-GAT + XAI)")
st.caption("Hybrid Static XGBoost + Behavioural k-NN Graph Attention Network with Logit-Gated Fusion & TreeSHAP")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
except Exception as exc:
    st.error(
        f"FastAPI backend is offline at {API_URL}. Start it first with: "
        f"`uvicorn api.main:app --port 8000`\n\nDetails: {exc}"
    )
    st.stop()

with st.sidebar:
    st.header("System Telemetry")
    st.success(f"Inference Backend Online ({health['device'].upper()})")
    gate_df = pd.DataFrame(
        {
            "Dim": [c for c, _ in STRIDE_FULL],
            "XGB Weight": health["xgb_gate_weights"],
            "B-GAT Weight": health["bgat_gate_weights"],
            "Tau*": health["calibrated_thresholds"],
        }
    )
    st.dataframe(gate_df, hide_index=True)

    st.header("Traffic Source")
    mode = st.radio("Select Input Mode", ["Preset Attack Scenarios", "Upload CSV Batch"])

if mode == "Preset Attack Scenarios":
    df_samples = pd.read_csv(SAMPLE_CSV)
    df_samples["attack_cat"] = df_samples["attack_cat"].fillna("Normal").astype(str).str.strip()
    categories = sorted(df_samples["attack_cat"].unique().tolist())

    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_cat = st.selectbox("Filter by Ground-Truth Attack Category", categories, index=0)
    subset = df_samples[df_samples["attack_cat"] == selected_cat].reset_index(drop=True)
    with col_sel2:
        if len(subset) > 1:
            row_idx = st.slider("Select Flow Sample Index", 0, len(subset) - 1, 0)
        else:
            row_idx = 0
            st.info("Only 1 sample flow available for this category (Index 0 selected).")

    selected_row = subset.iloc[[row_idx]]
    st.markdown(
        f"**Ground Truth Label:** `{selected_cat}` | "
        f"**Protocol:** `{selected_row['proto'].iloc[0]}` | "
        f"**Service:** `{selected_row['service'].iloc[0]}`"
    )

    if st.button("Analyze Flow with B-GAT + XGBoost", type="primary"):
        records = (
            selected_row.drop(columns=["id", "label", "attack_cat"], errors="ignore")
            .fillna(0)
            .to_dict(orient="records")
        )
        res = requests.post(f"{API_URL}/predict", json={"flows": records}, timeout=15)
        if not res.ok:
            st.error(f"API Error ({res.status_code}): {res.text}")
            st.stop()
        resp = res.json()
        pred = resp["predictions"][0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Verdict", "THREAT" if pred["is_threat"] else "BENIGN")
        m2.metric("Primary Threat", pred["primary_threat"] if pred["is_threat"] else "N/A")
        m3.metric("End-to-End Latency", f"{resp['ms_per_flow']} ms")
        active_str = ", ".join(pred["active_stride_labels"]) if pred["active_stride_labels"] else "None"
        st.info(f"**Active STRIDE Dimensions:** {active_str}")

        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("6-Dimensional STRIDE Threat Profile")
            labels = [f"{name} ({code})" for code, name in STRIDE_FULL]
            codes = [code for code, _ in STRIDE_FULL]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Logit-Gated Fusion", x=labels, y=[pred["fused_probabilities"][c] for c in codes], marker_color="#1f77b4"))
            fig.add_trace(go.Bar(name="Static XGBoost", x=labels, y=[pred["xgb_probabilities"][c] for c in codes], marker_color="#aec7e8"))
            fig.add_trace(go.Bar(name="Behavioural k-NN GAT", x=labels, y=[pred["bgat_probabilities"][c] for c in codes], marker_color="#ff7f0e"))
            fig.add_trace(go.Scatter(name="Calibrated Threshold tau*", x=labels, y=health["calibrated_thresholds"], mode="markers+lines", line=dict(color="red", dash="dash")))
            fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], title="Probability"), height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")

        with c_right:
            st.subheader(f"Top TreeSHAP Drivers ({pred['primary_threat']})")
            shap_df = pd.DataFrame(pred["top_shap_drivers"])
            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["shap_value"][::-1],
                    y=shap_df["feature"][::-1],
                    orientation="h",
                    marker_color=["#d62728" if v > 0 else "#2ca02c" for v in shap_df["shap_value"][::-1]],
                )
            )
            fig_shap.update_layout(xaxis_title="SHAP Contribution (Log-Odds)", height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig_shap, width="stretch")

        with st.expander("Inspect Standardized 8 STRIDE Behavioural Signals"):
            st.json(pred["engineered_signals"])

else:
    uploaded = st.file_uploader("Upload UNSW-NB15 Format CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded).head(100)
        clean_df = batch_df.drop(columns=["id", "label", "attack_cat"], errors="ignore").fillna(0)
        res = requests.post(f"{API_URL}/predict", json={"flows": clean_df.to_dict(orient="records")}, timeout=30)
        if not res.ok:
            st.error(f"API Error ({res.status_code}): {res.text}")
            st.stop()
        resp = res.json()
        st.success(f"Scored {resp['batch_size']} flows in {resp['total_latency_ms']} ms ({resp['ms_per_flow']} ms/flow)")
        summary = pd.DataFrame(
            [
                {
                    "Flow": p["flow_index"],
                    "Threat": p["is_threat"],
                    "Active STRIDE": ", ".join(p["active_stride_labels"]) or "Normal",
                    **p["fused_probabilities"],
                }
                for p in resp["predictions"]
            ]
        )
        st.dataframe(summary)
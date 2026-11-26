from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.stride_map import encode_stride_targets, normalize_attack_cat

SEED = 42
EPS = 1e-6

CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "label", "attack_cat"]

BEHAVIOURAL_COLS = [
    "feat_resource_exhaustion",
    "feat_conn_burst_rate",
    "feat_asymmetry_ratio",
    "feat_session_entropy",
    "feat_privesc_score",
    "feat_stealth_probe",
    "feat_payload_anomaly",
    "feat_repudiation_risk",
]


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    s = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def compute_behavioural_signals(df: pd.DataFrame) -> pd.DataFrame:
    dur = np.maximum(_col(df, "dur"), 0.0)
    sload = np.maximum(_col(df, "sload"), 0.0)
    spkts = np.maximum(_col(df, "spkts"), 0.0)
    dpkts = np.maximum(_col(df, "dpkts"), 0.0)
    sbytes = np.maximum(_col(df, "sbytes"), 0.0)
    dbytes = np.maximum(_col(df, "dbytes"), 0.0)
    ct_srv_src = np.maximum(_col(df, "ct_srv_src"), 0.0)
    ct_srv_dst = np.maximum(_col(df, "ct_srv_dst"), 0.0)
    sjit = np.maximum(_col(df, "sjit"), 0.0)
    djit = np.maximum(_col(df, "djit"), 0.0)
    smean = np.maximum(_col(df, "smean"), 0.0)
    dmean = np.maximum(_col(df, "dmean"), 0.0)
    is_ftp = (_col(df, "is_ftp_login") > 0).astype(np.float64)
    trans_depth = np.maximum(_col(df, "trans_depth"), 0.0)
    res_body_len = np.maximum(_col(df, "res_body_len"), 0.0)
    sttl = _col(df, "sttl")
    dttl = _col(df, "dttl")
    ct_ftp = np.maximum(_col(df, "ct_ftp_cmd"), 0.0)

    out = pd.DataFrame(index=df.index)
    out["feat_resource_exhaustion"] = np.log1p(sload) + (np.log1p(spkts) / (dur + EPS))
    out["feat_conn_burst_rate"] = (ct_srv_src + ct_srv_dst) / (dur + EPS)
    out["feat_asymmetry_ratio"] = np.abs(sbytes - dbytes) / (sbytes + dbytes + 1.0)
    out["feat_session_entropy"] = np.log1p(sjit + djit) + np.log1p(np.abs(smean - dmean))
    out["feat_privesc_score"] = (2.0 * is_ftp) + np.log1p(trans_depth) + np.log1p(res_body_len)
    out["feat_stealth_probe"] = np.abs(sttl - dttl) / ((dur + EPS) * np.log1p(sbytes) + EPS)
    out["feat_payload_anomaly"] = (np.log1p(smean) * np.log1p(dmean)) / (np.log1p(spkts + dpkts) + EPS)
    out["feat_repudiation_risk"] = (1.5 * ct_ftp) + (np.log1p(trans_depth) * np.log1p(sload))

    for c in BEHAVIOURAL_COLS:
        out[c] = np.nan_to_num(out[c].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return out


class STRIDEFeaturePipeline:
    def __init__(self):
        self.num_cols: list[str] = []
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.scaler = StandardScaler()

    def _prepare_parts(self, df: pd.DataFrame):
        behav_df = compute_behavioural_signals(df)
        num_df = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            num_df[c] = _col(df, c).astype(np.float32)
        cat_df = df[CATEGORICAL_COLS].fillna("unknown").astype(str).apply(lambda s: s.str.strip().str.lower())
        return num_df, cat_df, behav_df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.num_cols = [c for c in df.columns if c not in DROP_COLS + CATEGORICAL_COLS + BEHAVIOURAL_COLS]
        num_df, cat_df, behav_df = self._prepare_parts(df)
        num_and_behav = np.hstack([num_df.to_numpy(dtype=np.float32), behav_df.to_numpy(dtype=np.float32)])
        scaled_nb = self.scaler.fit_transform(num_and_behav).astype(np.float32)
        encoded_cat = self.ohe.fit_transform(cat_df)
        # Contract: [scaled_raw_numeric | one_hot_categorical | scaled_8_behavioural_signals]
        X = np.hstack([scaled_nb[:, :-8], encoded_cat, scaled_nb[:, -8:]]).astype(np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num_df, cat_df, behav_df = self._prepare_parts(df)
        num_and_behav = np.hstack([num_df.to_numpy(dtype=np.float32), behav_df.to_numpy(dtype=np.float32)])
        scaled_nb = self.scaler.transform(num_and_behav).astype(np.float32)
        encoded_cat = self.ohe.transform(cat_df)
        X = np.hstack([scaled_nb[:, :-8], encoded_cat, scaled_nb[:, -8:]]).astype(np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names(self) -> list[str]:
        cat_names = self.ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
        return self.num_cols + cat_names + BEHAVIOURAL_COLS


def build_datasets(root_dir: Path = Path(".")):
    raw_dir = root_dir / "data" / "raw"
    proc_dir = root_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_csv = raw_dir / "UNSW_NB15_training-set.csv"
    test_csv = raw_dir / "UNSW_NB15_testing-set.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Place UNSW_NB15 CSVs inside {raw_dir.resolve()}")

    df = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)
    y_stride = encode_stride_targets(df["attack_cat"])
    y_binary = (y_stride.sum(axis=1) > 0).astype(np.int64)
    attack_names = normalize_attack_cat(df["attack_cat"]).to_numpy()

    # 80/20 stratified train/test split, then 85/15 stratified train/val split
    idx_all = np.arange(len(df))
    idx_train_full, idx_test = train_test_split(
        idx_all, test_size=0.20, random_state=SEED, stratify=y_binary
    )
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=0.15, random_state=SEED, stratify=y_binary[idx_train_full]
    )

    pipeline = STRIDEFeaturePipeline()
    X_train = pipeline.fit_transform(df.iloc[idx_train])
    X_val = pipeline.transform(df.iloc[idx_val])
    X_test = pipeline.transform(df.iloc[idx_test])

    feature_names = pipeline.get_feature_names()
    assert feature_names[-8:] == BEHAVIOURAL_COLS, "Contract violation: last 8 features must be BEHAVIOURAL_COLS"

    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "y_train.npy", y_stride[idx_train])
    np.save(proc_dir / "y_val.npy", y_stride[idx_val])
    np.save(proc_dir / "y_test.npy", y_stride[idx_test])
    np.save(proc_dir / "attack_train.npy", attack_names[idx_train])
    np.save(proc_dir / "attack_val.npy", attack_names[idx_val])
    np.save(proc_dir / "attack_test.npy", attack_names[idx_test])

    # Save 200 raw test rows for the Streamlit live demo
    df.iloc[idx_test[:200]].to_csv(proc_dir / "demo_sample_flows.csv", index=False)
    joblib.dump(pipeline, proc_dir / "feature_pipeline.joblib")

    print(f"Saved processed datasets to {proc_dir}:")
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    print(f"  Tail 8 features verified: {feature_names[-8:]}")


if __name__ == "__main__":
    build_datasets()
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from xgboost import XGBClassifier

from src.stride_map import STRIDE_CODES

SEED = 42


def _make_xgb(params: dict) -> XGBClassifier:
    try:
        return XGBClassifier(**params, device="cuda")
    except Exception:
        return XGBClassifier(**params, device="cpu")


def _to_single_label_target(y_multi: np.ndarray) -> np.ndarray:
    """Maps 6D multi-label matrix to 7-class single label: 0=Normal, 1..6=first active STRIDE dim."""
    has_attack = y_multi.sum(axis=1) > 0
    primary_dim = np.argmax(y_multi, axis=1) + 1
    return np.where(has_attack, primary_dim, 0).astype(np.int64)


def _single_pred_to_6d(pred_cls: np.ndarray) -> np.ndarray:
    out = np.zeros((len(pred_cls), 6), dtype=np.float32)
    mask = pred_cls > 0
    out[mask, pred_cls[mask] - 1] = 1.0
    return out


def train_xgb_models(root_dir: Path = Path(".")):
    proc_dir = root_dir / "data" / "processed"
    art_dir = root_dir / "outputs" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    y_val = np.load(proc_dir / "y_val.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    # 1. Single-Label STRIDE Baseline (contiguous label remapping for XGBoost >= 2.0)
    print("Training Single-Label STRIDE XGBoost baseline...")
    y_tr_single = _to_single_label_target(y_train)
    unique_classes, y_tr_contig = np.unique(y_tr_single, return_inverse=True)

    single_clf = _make_xgb(
        dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softprob",
            num_class=len(unique_classes),
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=-1,
        )
    )
    single_clf.fit(X_train, y_tr_contig)
    single_test_contig = single_clf.predict(X_test).astype(np.int64)
    single_test_cls = unique_classes[single_test_contig]
    P_single_test = _single_pred_to_6d(single_test_cls)
    np.save(art_dir / "P_single_xgb_test.npy", P_single_test)
    single_macro_f1 = f1_score(y_test, P_single_test, average="macro", zero_division=0)
    print(f"  Single-Label Baseline Test Macro F1: {single_macro_f1:.4f}")

    # 2. Multi-Label Static XGBoost (6 parallel binary classifiers)
    print("Training 6 Multi-Label Static XGBoost classifiers...")
    t0 = time.perf_counter()
    models = []
    P_val = np.zeros((len(X_val), 6), dtype=np.float32)
    P_test = np.zeros((len(X_test), 6), dtype=np.float32)

    for j, code in enumerate(STRIDE_CODES):
        pos = float(y_train[:, j].sum())
        neg = float(len(y_train) - pos)
        spw = max(neg / max(pos, 1.0), 1.0)

        clf = _make_xgb(
            dict(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                scale_pos_weight=spw,
                random_state=SEED + j,
                n_jobs=-1,
            )
        )
        clf.fit(X_train, y_train[:, j])
        P_val[:, j] = clf.predict_proba(X_val)[:, 1]
        P_test[:, j] = clf.predict_proba(X_test)[:, 1]
        models.append(clf)
        dim_auc = roc_auc_score(y_test[:, j], P_test[:, j])
        print(f"  [{code}] scale_pos_weight={spw:.2f} | Test AUROC={dim_auc:.4f}")

    train_sec = time.perf_counter() - t0
    t_inf0 = time.perf_counter()
    _ = models[0].predict_proba(X_test[:1000])
    ms_per_flow = ((time.perf_counter() - t_inf0) * 6 / 1000.0) * 1000.0

    np.save(art_dir / "P_xgb_val.npy", P_val)
    np.save(art_dir / "P_xgb_test.npy", P_test)
    joblib.dump(models, art_dir / "xgb_multilabel_models.joblib")

    y_pred_default = (P_test >= 0.5).astype(np.float32)
    macro_f1 = f1_score(y_test, y_pred_default, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred_default, average="micro", zero_division=0)
    h_loss = hamming_loss(y_test, y_pred_default)
    macro_auc = roc_auc_score(y_test, P_test, average="macro")

    print(f"\nStatic Multi-Label XGBoost Summary (Train: {train_sec:.1f}s | Inference: {ms_per_flow:.3f} ms/flow):")
    print(f"  Test Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Hamming: {h_loss:.4f} | Macro AUROC: {macro_auc:.4f}")


if __name__ == "__main__":
    train_xgb_models()
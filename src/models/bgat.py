from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score

SEED = 42


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BehaviouralGATLayer(nn.Module):
    """Dynamic cosine k-NN Graph Attention over the 8 STRIDE behavioural signals."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, k: int = 8, dropout: float = 0.2):
        super().__init__()
        self.k = k
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, x_behav: torch.Tensor) -> torch.Tensor:
        b_norm = F.normalize(x_behav, p=2, dim=-1)
        sim = torch.matmul(b_norm, b_norm.transpose(0, 1))
        k_eff = min(self.k, h.size(0))
        _, nbr_idx = torch.topk(sim, k=k_eff, dim=-1)

        nbr_h = h[nbr_idx]
        attn_out, _ = self.mha(h.unsqueeze(1), nbr_h, nbr_h)
        h = self.norm1(h + self.drop(attn_out.squeeze(1)))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h


class STRIDEBehaviouralGAT(nn.Module):
    def __init__(self, in_dim: int = 202, d_model: int = 128, num_labels: int = 6, k: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_labels = num_labels
        self.k = k

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.gat1 = BehaviouralGATLayer(d_model=d_model, num_heads=4, k=k, dropout=0.2)
        self.gat2 = BehaviouralGATLayer(d_model=d_model, num_heads=4, k=k, dropout=0.2)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_behav = x[:, -8:]
        h = self.input_proj(x)
        h = self.gat1(h, x_behav)
        h = self.gat2(h, x_behav)
        return self.head(h)


@torch.no_grad()
def predict_bgat(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    preds = []
    for (xb,) in loader:
        probs = torch.sigmoid(model(xb.to(device, non_blocking=True)))
        preds.append(probs.cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def train_bgat_model(root_dir: Path = Path("."), epochs: int = 15, batch_size: int = 512, lr: float = 1e-3):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training STRIDEBehaviouralGAT on {device}...")

    proc_dir = root_dir / "data" / "processed"
    art_dir = root_dir / "outputs" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    y_val = np.load(proc_dir / "y_val.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    pos_counts = np.clip(y_train.sum(axis=0), 1.0, None)
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.tensor(np.sqrt(neg_counts / pos_counts), dtype=torch.float32, device=device)

    model = STRIDEBehaviouralGAT(in_dim=X_train.shape[1], d_model=128, num_labels=6, k=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val_auc = -1.0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()

        P_val_ep = predict_bgat(model, X_val, device, batch_size=batch_size)
        val_auc = roc_auc_score(y_val, P_val_ep, average="macro")
        val_f1 = f1_score(y_val, (P_val_ep >= 0.5).astype(np.float32), average="macro", zero_division=0)
        print(
            f"  Epoch {epoch:02d}/{epochs} | Loss: {running_loss / len(X_train):.4f} "
            f"| Val AUROC: {val_auc:.4f} | Val Macro F1: {val_f1:.4f}"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    train_sec = time.perf_counter() - t0
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    P_val = predict_bgat(model, X_val, device, batch_size=batch_size)
    t_inf0 = time.perf_counter()
    P_test = predict_bgat(model, X_test, device, batch_size=batch_size)
    ms_per_flow = ((time.perf_counter() - t_inf0) / len(X_test)) * 1000.0

    np.save(art_dir / "P_gat_val.npy", P_val)
    np.save(art_dir / "P_gat_test.npy", P_test)
    torch.save(
        {
            "state_dict": best_state,
            "in_dim": X_train.shape[1],
            "d_model": 128,
            "num_labels": 6,
            "k": 8,
        },
        art_dir / "bgat.pt",
    )

    y_pred_default = (P_test >= 0.5).astype(np.float32)
    macro_f1 = f1_score(y_test, y_pred_default, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred_default, average="micro", zero_division=0)
    h_loss = hamming_loss(y_test, y_pred_default)
    macro_auc = roc_auc_score(y_test, P_test, average="macro")

    print(f"\nStandalone B-GAT Summary (Train: {train_sec:.1f}s | Inference: {ms_per_flow:.4f} ms/flow):")
    print(f"  Test Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Hamming: {h_loss:.4f} | Macro AUROC: {macro_auc:.4f}")


if __name__ == "__main__":
    train_bgat_model()
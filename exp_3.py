#!/usr/bin/env python3
"""
==========================================================================
PERMUTATION IMPORTANCE ANALYSIS
Battery Glass Fatigue — GATv2 Feature Importance
==========================================================================
Runs on the EXISTING N=256 snapshot data (battery_snapshots.npy).
No new simulations needed.

WHAT IT DOES:
  1. Trains one GATv2Classifier per fold on the full 8D feature set
     (same architecture + hyperparameters as the main ablation paper).
  2. For each of the 8 features, sets that column to zero across ALL
     nodes in the validation set and re-evaluates accuracy + AUC.
  3. Importance = drop in AUC vs. the intact model.
  4. Produces:
       · permutation_importance_bar.png  — mean ± std AUC drop per feature
       · permutation_importance_table.txt — full numbers for the paper
       · permutation_importance_heatmap.png — per-fold heatmap

WHY ZERO-OUT (not shuffle):
  Shuffling permutes values across graphs, leaking relative ordering.
  Zeroing replaces the feature with its "missing" value, which is
  equivalent to the model receiving no information from that channel.
  Results are conservative (a lower bound on true importance).

RUNTIME: ~25–35 min on T4 (5 folds × ~5 min training each)

USAGE (Kaggle / Colab):
  Set SNAP_PATH to wherever battery_snapshots.npy lives, then run.
==========================================================================
"""

import os, time, gc, warnings
warnings.filterwarnings('ignore')
os.system("pip install -q torch_geometric 2>/dev/null")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

# ══════════════════════════════════════════════════════════════════════════
# CONFIG  —  must match the original paper exactly
# ══════════════════════════════════════════════════════════════════════════

N_ATOMS      = 256
RHO          = 1.2
BOX_L        = float((N_ATOMS / RHO) ** (1 / 3))   # ≈ 5.975 σ
RC_GRAPH     = 1.5
N_CYCLES     = 400
PRISTINE_CYC = {0}
FATIGUED_CYC = {300, 400}

HIDDEN_DIM   = 64
N_HEADS      = 4
BATCH_SIZE   = 32
LR           = 3e-4
MAX_EPOCHS   = 150
PATIENCE     = 25
N_FOLDS      = 5

# ── Feature names (must match extract_features column order) ──────────────
FEATURE_NAMES = [
    r'$\bar{r}$ (mean)',       # 0
    r'$\sigma_r$ (std)',       # 1
    r'$r_{\min}$',             # 2
    r'$r_{\max}$',             # 3
    r'$\tilde{d}$ (coord)',    # 4
    r'skewness $\tilde{\gamma}$',  # 5
    r'$Q_{25}$',               # 6
    r'$Q_{75}$',               # 7
]
N_FEAT = len(FEATURE_NAMES)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEM  = (DEVICE.type == 'cuda')

# ── Paths ─────────────────────────────────────────────────────────────────
# Try common locations; update SNAP_PATH if your file is elsewhere
for _candidate in [
    "/content/test_v2/battery_snapshots.npy",
    "/kaggle/working/battery_snapshots.npy",
    "/kaggle/working/results_N1024/battery_LJN1024_ALL300_merged.npy",
    "battery_snapshots.npy",
]:
    if os.path.exists(_candidate):
        SNAP_PATH = _candidate
        break
else:
    raise FileNotFoundError(
        "battery_snapshots.npy not found. "
        "Set SNAP_PATH manually to your snapshot file."
    )

OUT_DIR = os.path.join(os.path.dirname(SNAP_PATH), "permutation_importance")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device       : {DEVICE}")
print(f"Snapshot file: {SNAP_PATH}")
print(f"Output dir   : {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (identical to paper)
# ══════════════════════════════════════════════════════════════════════════

def extract_features(positions, box_length=BOX_L, rc=RC_GRAPH, n_feat=8):
    """8D bond-length feature extractor — no instance norm (baseline config)."""
    N   = len(positions)
    pos = np.array(positions, dtype=np.float32)

    dr   = pos[:, None, :] - pos[None, :, :]
    dr   = dr - box_length * np.round(dr / box_length)
    dist = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr))

    if not np.isfinite(dist).all():
        return None, None, None

    nbr_mask = (dist > 1e-6) & (dist < rc)
    coords   = nbr_mask.sum(axis=1).astype(np.float32)
    d_max    = coords.max() if coords.max() > 0 else 1.0

    feats = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        nd = dist[i][nbr_mask[i]]
        if len(nd) == 0:
            feats[i] = [rc, 0.0, rc, rc, 0.0]
            continue
        feats[i, 0] = nd.mean()
        feats[i, 1] = nd.std() if len(nd) > 1 else 0.0
        feats[i, 2] = nd.min()
        feats[i, 3] = nd.max()
        feats[i, 4] = coords[i] / d_max

    if n_feat == 8:
        extra = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            nd = dist[i][nbr_mask[i]]
            if len(nd) < 3:
                extra[i] = [0.0, feats[i, 2], feats[i, 3]]
                continue
            std3 = max(float(nd.std()), 1e-8) ** 3
            skew = float(np.mean((nd - nd.mean()) ** 3) / std3)
            extra[i] = [np.clip(skew, -5, 5),
                        np.percentile(nd, 25),
                        np.percentile(nd, 75)]
        feats = np.concatenate([feats, extra], axis=1)

    if not np.isfinite(feats).all():
        return None, None, None

    rows, cols = np.where(nbr_mask)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = dist[rows, cols].reshape(-1, 1).astype(np.float32)
    return feats, edge_index, edge_attr


def build_clf_dataset(snapshots):
    """Build classification PyG dataset with glass_id stored per graph."""
    data_list = []
    skipped   = 0
    for snap in snapshots:
        cyc = snap['cycle']
        if cyc not in PRISTINE_CYC and cyc not in FATIGUED_CYC:
            continue
        nf, ei, ea = extract_features(snap['positions'])
        if nf is None:
            skipped += 1
            continue
        label = 0 if cyc in PRISTINE_CYC else 1
        data_list.append(Data(
            x          = torch.from_numpy(nf).float(),
            edge_index = torch.from_numpy(ei).long(),
            edge_attr  = torch.from_numpy(ea).float(),
            y          = label,
            glass_id   = int(snap['glass_id']),
        ))
    print(f"  Built {len(data_list)} graphs  ({skipped} skipped)")
    return data_list


# ══════════════════════════════════════════════════════════════════════════
# MODEL  (identical to paper)
# ══════════════════════════════════════════════════════════════════════════

class GATv2Classifier(nn.Module):
    def __init__(self, in_dim=8, hidden=HIDDEN_DIM, heads=N_HEADS):
        super().__init__()
        self.enc  = nn.Linear(in_dim, hidden)
        self.gat1 = GATv2Conv(hidden,       hidden, heads=heads,
                              edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(hidden*heads, hidden, heads=heads,
                              edge_dim=1, concat=True)
        self.post = nn.Sequential(
            nn.Linear(hidden*heads, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, data):
        x = F.relu(self.enc(data.x))
        x = F.relu(self.gat1(x, data.edge_index, data.edge_attr))
        x = F.relu(self.gat2(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        return self.head(self.post(x)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=PIN_MEM)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            loss = criterion(model(batch), batch.y.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, zeroed_feat=None):
    """
    Evaluate model on loader.

    zeroed_feat : int or None
        If an int, sets that feature column to 0.0 for every node
        in every graph before the forward pass.
        The original data tensors are NOT modified — we clone per batch.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    logits_all, labels_all = [], []

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=PIN_MEM)

        if zeroed_feat is not None:
            # Clone to avoid corrupting the cached dataset
            x_mod = batch.x.clone()
            x_mod[:, zeroed_feat] = 0.0
            batch = batch.clone()
            batch.x = x_mod

        logits_all.append(model(batch).cpu())
        labels_all.append(batch.y.cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).float()
    loss   = criterion(logits, labels).item()
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)
    labs   = labels.numpy().astype(int)
    auc    = roc_auc_score(labs, probs)
    acc    = (preds == labs).mean() * 100
    f1     = f1_score(labs, preds)
    return loss, acc, auc, f1


def train_model(train_data, val_data):
    """Full training loop; returns best model state dict and baseline metrics."""
    tr_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=2, pin_memory=PIN_MEM)
    va_loader = DataLoader(val_data,   batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2, pin_memory=PIN_MEM)

    model     = GATv2Classifier(in_dim=N_FEAT).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    scaler    = GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_loss  = float('inf')
    best_state = None
    best_auc   = 0.0
    best_acc   = 0.0
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, tr_loader, optimizer, scaler)
        vl, va, vauc, vf1 = evaluate(model, va_loader)
        scheduler.step()

        if vl < best_loss:
            best_loss  = vl
            best_acc   = va
            best_auc   = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 25 == 0:
            print(f"    ep {epoch:3d} | val_loss {vl:.4f} | "
                  f"acc {va:.1f}% | AUC {vauc:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, va_loader, best_acc, best_auc


# ══════════════════════════════════════════════════════════════════════════
# GLASS-LEVEL CV FOLDS  (same leak-free logic as the fixed ablation script)
# ══════════════════════════════════════════════════════════════════════════

def glass_level_folds(data_list, n_folds=N_FOLDS, seed=42):
    """Split by glass_id so no glass straddles train/val."""
    glass_ids   = np.array([d.glass_id for d in data_list])
    unique_gids = np.unique(glass_ids)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_gids, va_gids in kf.split(unique_gids):
        tr_set = set(unique_gids[tr_gids])
        va_set = set(unique_gids[va_gids])
        tr_idx = [i for i, d in enumerate(data_list) if d.glass_id in tr_set]
        va_idx = [i for i, d in enumerate(data_list) if d.glass_id in va_set]
        yield tr_idx, va_idx


# ══════════════════════════════════════════════════════════════════════════
# SHUFFLED-LABEL CONTROL  (sanity check — acc must collapse to ~50%)
# ══════════════════════════════════════════════════════════════════════════

def run_shuffled_label_control(data_list):
    """
    Train on randomly permuted labels.
    A correctly implemented model should achieve ~50% accuracy — confirming
    the real model is not memorising graph structure or indices.
    """
    print(f"\n{'═'*55}")
    print("  SHUFFLED-LABEL CONTROL")
    print("  Expected: acc ≈ 50%, AUC ≈ 0.50  (random)")
    print(f"{'═'*55}")

    # Permute labels in-place on a copy
    import copy
    shuffled = copy.deepcopy(data_list)
    rng      = np.random.default_rng(seed=0)
    perm     = rng.permutation(len(shuffled))
    orig_labels = [d.y for d in shuffled]
    for i, d in enumerate(shuffled):
        d.y = orig_labels[perm[i]]

    accs, aucs = [], []
    for fold_i, (tr_idx, va_idx) in enumerate(glass_level_folds(shuffled)):
        print(f"\n  ── Fold {fold_i+1}/{N_FOLDS} ──")
        model, va_loader, _, _ = train_model(
            [shuffled[i] for i in tr_idx],
            [shuffled[i] for i in va_idx],
        )
        _, acc, auc, _ = evaluate(model, va_loader)
        accs.append(acc); aucs.append(auc)
        print(f"  Fold {fold_i+1}: acc={acc:.1f}%  AUC={auc:.4f}")

    print(f"\n  MEAN  acc={np.mean(accs):.1f}±{np.std(accs):.1f}%  "
          f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")
    print("  (Should be ~50% / ~0.50 — confirms no structural memorisation)")
    return np.mean(accs), np.mean(aucs)


# ══════════════════════════════════════════════════════════════════════════
# PERMUTATION IMPORTANCE  (main experiment)
# ══════════════════════════════════════════════════════════════════════════

def run_permutation_importance(data_list):
    """
    For each fold:
      1. Train GATv2Classifier on train split.
      2. Evaluate on val split with ALL features intact → baseline AUC.
      3. For each feature i, zero out column i and re-evaluate → AUC_i.
      4. importance_i = baseline_AUC − AUC_i  (higher = more important).

    Returns:
      importance_matrix : (N_FOLDS, N_FEAT) array of AUC drops
      baseline_aucs     : (N_FOLDS,) array of baseline AUCs
    """
    print(f"\n{'═'*55}")
    print("  PERMUTATION IMPORTANCE ANALYSIS")
    print(f"  Features: {N_FEAT}D | Folds: {N_FOLDS}")
    print(f"{'═'*55}")

    importance_matrix = np.zeros((N_FOLDS, N_FEAT))
    baseline_aucs     = np.zeros(N_FOLDS)
    baseline_accs     = np.zeros(N_FOLDS)

    for fold_i, (tr_idx, va_idx) in enumerate(glass_level_folds(data_list)):
        print(f"\n  ── Fold {fold_i+1}/{N_FOLDS} ──────────────────────────")
        t0 = time.time()

        train_data = [data_list[i] for i in tr_idx]
        val_data   = [data_list[i] for i in va_idx]

        # Step 1: train
        model, va_loader, base_acc, base_auc = train_model(train_data, val_data)
        baseline_aucs[fold_i] = base_auc
        baseline_accs[fold_i] = base_acc
        print(f"  Baseline — acc={base_acc:.2f}%  AUC={base_auc:.4f}")

        # Step 2: permute each feature
        for feat_i, feat_name in enumerate(FEATURE_NAMES):
            _, acc_z, auc_z, _ = evaluate(model, va_loader,
                                          zeroed_feat=feat_i)
            drop = base_auc - auc_z
            importance_matrix[fold_i, feat_i] = drop
            print(f"    zero {feat_name:<28}  "
                  f"AUC {auc_z:.4f}  drop {drop:+.4f}")

        print(f"  Fold {fold_i+1} done in {(time.time()-t0)/60:.1f} min")
        gc.collect()

    return importance_matrix, baseline_aucs, baseline_accs


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

def plot_importance_bar(importance_matrix, baseline_aucs, out_path):
    """
    Horizontal bar chart: mean ± std AUC drop per feature.
    Features sorted by mean importance (most important at top).
    """
    means = importance_matrix.mean(axis=0)
    stds  = importance_matrix.std(axis=0)
    order = np.argsort(means)[::-1]   # descending

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = np.arange(N_FEAT)
    colors  = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, N_FEAT))

    bars = ax.barh(y_pos, means[order], xerr=stds[order],
                   align='center', color=colors,
                   error_kw=dict(ecolor='black', capsize=4, lw=1.2))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=11)
    ax.set_xlabel('AUC drop when feature zeroed\n'
                  r'(higher $\rightarrow$ more important)', fontsize=11)
    ax.set_title(
        f'GATv2 Permutation Importance\n'
        f'N=256 Battery Glass Fatigue  '
        f'(baseline AUC = {baseline_aucs.mean():.4f} ± {baseline_aucs.std():.4f})',
        fontsize=12
    )
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_importance_heatmap(importance_matrix, out_path):
    """Per-fold heatmap showing stability of importance across folds."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Sort features by mean importance
    order = np.argsort(importance_matrix.mean(axis=0))[::-1]
    data  = importance_matrix[:, order]
    names = [FEATURE_NAMES[i] for i in order]

    vmax = max(np.abs(data).max(), 0.01)
    im   = ax.imshow(data, aspect='auto', cmap='RdYlGn_r',
                     vmin=-vmax/4, vmax=vmax)

    ax.set_xticks(range(N_FEAT))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(N_FOLDS))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(N_FOLDS)], fontsize=10)
    ax.set_title('Permutation Importance per Fold (AUC drop)', fontsize=12)

    plt.colorbar(im, ax=ax, label='AUC drop')

    # Annotate cells
    for r in range(N_FOLDS):
        for c in range(N_FEAT):
            ax.text(c, r, f'{data[r, c]:.3f}',
                    ha='center', va='center', fontsize=7.5,
                    color='white' if abs(data[r, c]) > vmax * 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def save_text_table(importance_matrix, baseline_aucs, baseline_accs,
                    shuffled_acc, shuffled_auc, out_path):
    """Write a plain-text table ready to paste into the paper."""
    means = importance_matrix.mean(axis=0)
    stds  = importance_matrix.std(axis=0)
    order = np.argsort(means)[::-1]

    lines = []
    lines.append("=" * 65)
    lines.append("  PERMUTATION IMPORTANCE — N=256 Battery Glass Fatigue GATv2")
    lines.append("=" * 65)
    lines.append(f"  Baseline  acc = {baseline_accs.mean():.2f} ± {baseline_accs.std():.2f}%")
    lines.append(f"  Baseline  AUC = {baseline_aucs.mean():.4f} ± {baseline_aucs.std():.4f}")
    lines.append("-" * 65)
    lines.append(f"  {'Rank':<5} {'Feature':<30} {'AUC drop (mean±std)'}")
    lines.append("-" * 65)
    for rank, i in enumerate(order, 1):
        lines.append(f"  {rank:<5} {FEATURE_NAMES[i]:<30} "
                     f"{means[i]:+.4f} ± {stds[i]:.4f}")
    lines.append("-" * 65)
    lines.append("\n  Per-fold importance matrix:")
    header = "  " + " ".join(f"{FEATURE_NAMES[i]:>9}" for i in order)
    lines.append(header)
    for fold_i in range(N_FOLDS):
        row = f"  Fold {fold_i+1}" + "".join(
            f" {importance_matrix[fold_i, i]:>+9.4f}" for i in order)
        lines.append(row)
    lines.append("=" * 65)
    lines.append("\n  SHUFFLED-LABEL CONTROL")
    lines.append(f"  Shuffled acc = {shuffled_acc:.1f}%  (expected ~50%)")
    lines.append(f"  Shuffled AUC = {shuffled_auc:.4f}  (expected ~0.50)")
    lines.append("=" * 65)

    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, 'w') as f:
        f.write(text)
    print(f"\n  Table saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_wall = time.time()

    # ── Load snapshots ─────────────────────────────────────────────────────
    print(f"\nLoading snapshots from {SNAP_PATH} ...")
    snapshots = list(np.load(SNAP_PATH, allow_pickle=True))
    print(f"  Loaded {len(snapshots)} snapshots")
    glass_ids = sorted(set(s['glass_id'] for s in snapshots))
    print(f"  Glasses: {len(glass_ids)}  "
          f"(ids {glass_ids[0]}–{glass_ids[-1]})")

    # ── Build classification dataset ───────────────────────────────────────
    print("\nBuilding classification graphs (8D, no instance norm) ...")
    data_list = build_clf_dataset(snapshots)
    print(f"  {sum(1 for d in data_list if d.y==0)} pristine  "
          f"+ {sum(1 for d in data_list if d.y==1)} fatigued")

    # ── Shuffled-label control ─────────────────────────────────────────────
    shuffled_acc, shuffled_auc = run_shuffled_label_control(data_list)
    gc.collect()

    # ── Permutation importance ─────────────────────────────────────────────
    importance_matrix, baseline_aucs, baseline_accs = \
        run_permutation_importance(data_list)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_importance_bar(
        importance_matrix, baseline_aucs,
        os.path.join(OUT_DIR, "permutation_importance_bar.png"))
    plot_importance_heatmap(
        importance_matrix,
        os.path.join(OUT_DIR, "permutation_importance_heatmap.png"))
    save_text_table(
        importance_matrix, baseline_aucs, baseline_accs,
        shuffled_acc, shuffled_auc,
        os.path.join(OUT_DIR, "permutation_importance_table.txt"))

    # ── Final summary ──────────────────────────────────────────────────────
    means = importance_matrix.mean(axis=0)
    order = np.argsort(means)[::-1]

    print(f"\n{'═'*55}")
    print("  FINAL SUMMARY")
    print(f"{'═'*55}")
    print(f"  Baseline AUC : {baseline_aucs.mean():.4f} ± "
          f"{baseline_aucs.std():.4f}")
    print(f"  Shuffled AUC : {shuffled_auc:.4f}  (sanity ✓)")
    print(f"\n  Feature ranking by AUC drop:")
    for rank, i in enumerate(order, 1):
        bar = "█" * int(means[i] / (means[order[0]] + 1e-8) * 20)
        print(f"  {rank}. {FEATURE_NAMES[i]:<28} {means[i]:+.4f}  {bar}")
    print(f"{'═'*55}")
    print(f"\n  Total runtime : {(time.time()-t_wall)/60:.1f} min")
    print(f"  Output dir    : {OUT_DIR}")


if __name__ == "__main__":
    main()
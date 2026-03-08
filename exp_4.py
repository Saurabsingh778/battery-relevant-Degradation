#!/usr/bin/env python3
"""
==========================================================================
MINIMAL 4D FEATURE ABLATION (N=256)
Battery Glass Fatigue — Minimal Sufficient Descriptor Set
==========================================================================
Hypothesis: The fatigue signature is fully characterised by just 4 features:
  1. r_max (maximum bond length)
  2. r_mean (mean bond length)
  3. skewness (distributional asymmetry)
  4. r_std (distributional spread)

This script drops the other 4 features (r_min, coordination, Q25, Q75) 
and evaluates if the GATv2 model preserves the ~98% accuracy and ~0.31 R^2.

RUNTIME: ~10-15 min on T4 GPU
==========================================================================
"""

import os, time, gc, warnings
warnings.filterwarnings('ignore')
os.system("pip install -q torch_geometric 2>/dev/null")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, r2_score

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

N_ATOMS      = 256
RHO          = 1.2
BOX_L        = float((N_ATOMS / RHO) ** (1 / 3))
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

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEM = (DEVICE.type == 'cuda')

# Target exactly the 4 features identified in the permutation importance
# Original 8D indices: 0:mean, 1:std, 2:min, 3:max, 4:coord, 5:skew, 6:Q25, 7:Q75
KEEP_INDICES = [0, 1, 3, 5]
FEATURE_NAMES = [r'$\bar{r}$ (mean)', r'$\sigma_r$ (std)', r'$r_{\max}$', r'skewness $\tilde{\gamma}$']
N_FEAT = len(KEEP_INDICES)

# Resolve path
for _candidate in [
    "/content/test_v2/battery_snapshots.npy",
    "/kaggle/working/battery_snapshots.npy",
    "battery_snapshots.npy",
]:
    if os.path.exists(_candidate):
        SNAP_PATH = _candidate
        break
else:
    raise FileNotFoundError("battery_snapshots.npy not found. Set SNAP_PATH.")

print(f"Device       : {DEVICE}")
print(f"Features (4D): {FEATURE_NAMES}")
print(f"Snapshot file: {SNAP_PATH}")

# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (4D Minimal Set)
# ══════════════════════════════════════════════════════════════════════════

def extract_4d_features(positions, box_length=BOX_L, rc=RC_GRAPH, instance_norm=False):
    """Extracts 8D features and strictly slices it down to the targeted 4D set."""
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

    extra = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        nd = dist[i][nbr_mask[i]]
        if len(nd) < 3:
            extra[i] = [0.0, feats[i, 2], feats[i, 3]]
            continue
        std3 = max(float(nd.std()), 1e-8) ** 3
        skew = float(np.mean((nd - nd.mean()) ** 3) / std3)
        extra[i] = [np.clip(skew, -5, 5), np.percentile(nd, 25), np.percentile(nd, 75)]
        
    feats_8d = np.concatenate([feats, extra], axis=1)
    
    # ---- SLICE DOWN TO 4D ----
    feats_4d = feats_8d[:, KEEP_INDICES]

    # Apply Instance Normalization if requested
    if instance_norm:
        mu    = feats_4d.mean(axis=0, keepdims=True)
        sigma = feats_4d.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        feats_4d = (feats_4d - mu) / sigma

    if not np.isfinite(feats_4d).all():
        return None, None, None

    rows, cols = np.where(nbr_mask)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = dist[rows, cols].reshape(-1, 1).astype(np.float32)
    return feats_4d, edge_index, edge_attr


def build_datasets(snapshots, instance_norm=False):
    """Build classification and regression PyG datasets simultaneously."""
    clf_data, reg_data = [], []
    skipped = 0
    
    for snap in snapshots:
        cyc = snap['cycle']
        gid = int(snap['glass_id'])
        
        nf, ei, ea = extract_4d_features(snap['positions'], instance_norm=instance_norm)
        if nf is None:
            skipped += 1
            continue
            
        x_t  = torch.from_numpy(nf).float()
        ei_t = torch.from_numpy(ei).long()
        ea_t = torch.from_numpy(ea).float()

        # Classification
        if cyc in PRISTINE_CYC:
            clf_data.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, y=0, glass_id=gid))
        elif cyc in FATIGUED_CYC:
            clf_data.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, y=1, glass_id=gid))
            
        # Regression
        reg_label = torch.tensor([cyc / N_CYCLES], dtype=torch.float32)
        reg_data.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, y=reg_label, glass_id=gid))

    print(f"  Built clf: {len(clf_data)} | reg: {len(reg_data)} graphs ({skipped} skipped)")
    return clf_data, reg_data


# ══════════════════════════════════════════════════════════════════════════
# MODELS (GATv2 - Identical Architecture)
# ══════════════════════════════════════════════════════════════════════════

class GATv2Base(nn.Module):
    def __init__(self, in_dim=4, hidden=HIDDEN_DIM, heads=N_HEADS):
        super().__init__()
        self.enc  = nn.Linear(in_dim, hidden)
        self.gat1 = GATv2Conv(hidden, hidden, heads=heads, edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(hidden*heads, hidden, heads=heads, edge_dim=1, concat=True)
        self.post = nn.Sequential(
            nn.Linear(hidden*heads, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward_repr(self, data):
        x = F.relu(self.enc(data.x))
        x = F.relu(self.gat1(x, data.edge_index, data.edge_attr))
        x = F.relu(self.gat2(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        return self.post(x)

class GATv2Classifier(GATv2Base):
    def forward(self, data):
        return self.head(self.forward_repr(data)).squeeze(-1)

class GATv2Regressor(GATv2Base):
    def forward(self, data):
        return torch.sigmoid(self.head(self.forward_repr(data))).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
# RUNNERS & EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def glass_level_folds(data_list, n_folds=N_FOLDS, seed=42):
    glass_ids   = np.array([d.glass_id for d in data_list])
    unique_gids = np.unique(glass_ids)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_gids, va_gids in kf.split(unique_gids):
        tr_set = set(unique_gids[tr_gids])
        va_set = set(unique_gids[va_gids])
        tr_idx = [i for i, d in enumerate(data_list) if d.glass_id in tr_set]
        va_idx = [i for i, d in enumerate(data_list) if d.glass_id in va_set]
        yield tr_idx, va_idx


def run_classification(clf_data, tag=""):
    print(f"\n  [ CLASSIFICATION ] {tag}")
    fold_results = []
    
    for fold_i, (tr_idx, va_idx) in enumerate(glass_level_folds(clf_data)):
        tr_loader = DataLoader([clf_data[i] for i in tr_idx], batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEM)
        va_loader = DataLoader([clf_data[i] for i in va_idx], batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEM)

        model     = GATv2Classifier(in_dim=N_FEAT).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
        scaler    = GradScaler(enabled=(DEVICE.type=='cuda'))

        best_loss = float('inf')
        best_acc = best_auc = 0.0
        patience_ctr = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for batch in tr_loader:
                batch = batch.to(DEVICE, non_blocking=PIN_MEM)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                    loss = nn.BCEWithLogitsLoss()(model(batch), batch.y.float())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            
            scheduler.step()
            
            # Eval
            model.eval()
            logits_all, labels_all = [], []
            with torch.no_grad():
                for batch in va_loader:
                    batch = batch.to(DEVICE, non_blocking=PIN_MEM)
                    logits_all.append(model(batch).cpu())
                    labels_all.append(batch.y.cpu())
                    
            logits = torch.cat(logits_all)
            labels = torch.cat(labels_all).float()
            vl     = nn.BCEWithLogitsLoss()(logits, labels).item()
            probs  = torch.sigmoid(logits).numpy()
            preds  = (probs > 0.5).astype(int)
            labs   = labels.numpy().astype(int)
            
            va   = (preds == labs).mean() * 100
            vauc = roc_auc_score(labs, probs)

            if vl < best_loss:
                best_loss, best_acc, best_auc = vl, va, vauc
                patience_ctr = 0
            else:
                patience_ctr += 1
            
            if epoch % 10 == 0: print(f"      Epoch {epoch:3d} | Acc: {va:.1f}%")

            if patience_ctr >= PATIENCE:
                break
                
        fold_results.append((best_acc, best_auc))
        print(f"    Fold {fold_i+1}: Acc {best_acc:.2f}% | AUC {best_auc:.4f}")

    accs = [r[0] for r in fold_results]
    aucs = [r[1] for r in fold_results]
    print(f"    MEAN: Acc {np.mean(accs):.2f}±{np.std(accs):.2f}% | AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}")
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)


def run_regression(reg_data, tag=""):
    print(f"\n  [ REGRESSION ] {tag}")
    fold_results = []
    
    for fold_i, (tr_idx, va_idx) in enumerate(glass_level_folds(reg_data)):
        tr_loader = DataLoader([reg_data[i] for i in tr_idx], batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEM)
        va_loader = DataLoader([reg_data[i] for i in va_idx], batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEM)

        model     = GATv2Regressor(in_dim=N_FEAT).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
        scaler    = GradScaler(enabled=(DEVICE.type=='cuda'))

        best_r2 = -np.inf
        patience_ctr = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for batch in tr_loader:
                batch = batch.to(DEVICE, non_blocking=PIN_MEM)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                    loss = nn.MSELoss()(model(batch), batch.y.squeeze(-1).float())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                
            scheduler.step()
            
            # Eval
            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for batch in va_loader:
                    batch = batch.to(DEVICE, non_blocking=PIN_MEM)
                    preds_all.append(model(batch).cpu())
                    labels_all.append(batch.y.squeeze(-1).cpu())
                    
            preds  = torch.cat(preds_all).numpy()
            labels = torch.cat(labels_all).numpy()
            vr2    = r2_score(labels, preds)

            if vr2 > best_r2:
                best_r2 = vr2
                patience_ctr = 0
            else:
                patience_ctr += 1
            if epoch % 10 == 0: print(f"      Epoch {epoch:3d} | R²: {vr2:.4f}")
            if patience_ctr >= PATIENCE:
                break
                
        fold_results.append(best_r2)
        print(f"    Fold {fold_i+1}: R² {best_r2:.4f}")

    print(f"    MEAN: R² {np.mean(fold_results):.4f}±{np.std(fold_results):.4f}")
    return np.mean(fold_results), np.std(fold_results)


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_wall = time.time()
    
    print("\nLoading snapshots...")
    snapshots = list(np.load(SNAP_PATH, allow_pickle=True))
    
    print("\n" + "═"*55)
    print(" 4D BASELINE (No Normalisation)")
    print("═"*55)
    clf_base, reg_base = build_datasets(snapshots, instance_norm=False)
    base_acc_m, base_acc_s, base_auc_m, base_auc_s = run_classification(clf_base, "4D Baseline")
    base_r2_m, base_r2_s = run_regression(reg_base, "4D Baseline")
    
    print("\n" + "═"*55)
    print(" 4D TEST A (With Instance Normalisation)")
    print("═"*55)
    clf_norm, reg_norm = build_datasets(snapshots, instance_norm=True)
    norm_acc_m, norm_acc_s, norm_auc_m, norm_auc_s = run_classification(clf_norm, "4D + InstNorm")
    norm_r2_m, norm_r2_s = run_regression(reg_norm, "4D + InstNorm")
    
    print(f"\n{'═'*65}")
    print(" 4D MINIMAL FEATURE ABLATION SUMMARY (N=256)")
    print(f"{'═'*65}")
    print(" Reference 8D Paper Results:")
    print("   8D Baseline : Acc = 91.67±4.22% | R² = 0.062")
    print("   8D Test A   : Acc = 98.33±1.49% | R² = 0.311")
    print("-" * 65)
    print(" New 4D Results:")
    print(f"   4D Baseline : Acc = {base_acc_m:.2f}±{base_acc_s:.2f}% | R² = {base_r2_m:.4f}±{base_r2_s:.4f}")
    print(f"   4D Test A   : Acc = {norm_acc_m:.2f}±{norm_acc_s:.2f}% | R² = {norm_r2_m:.4f}±{norm_r2_s:.4f}")
    print("=" * 65)
    print(f"Runtime: {(time.time() - t_wall)/60:.1f} min")

if __name__ == "__main__":
    main()
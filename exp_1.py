#!/usr/bin/env python3
"""
==========================================================================
BATTERY GLASS FATIGUE EXPERIMENT
Geometric Signatures of Cyclic Mechanical Fatigue in Model Glasses
==========================================================================
Companion study to: "Geometric Encoding of Thermal History in Glasses"

SCIENTIFIC QUESTION:
  Does cyclic volumetric strain (simulating battery charge/discharge)
  encode a learnable geometric signature in model LJ glasses?

EXPERIMENTS:
  1. Classification  : pristine (cycle 0-5) vs fatigued (cycle 40-50)
  2. Regression      : predict cycle number from geometry alone
  3. Visualization   : force chains + attention weight maps

TARGET RUNTIME: ~60-90 min on Kaggle T4 GPU
==========================================================================
"""

# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — Environment Setup
# Run this cell first. Sets JAX memory limits BEFORE importing JAX.
# ══════════════════════════════════════════════════════════════════════════
import os, sys, time, gc, warnings
warnings.filterwarnings('ignore')

# CRITICAL: Set JAX memory limits before importing jax
# Leaves ~55% GPU RAM for PyTorch GNN training later
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.40')

# Install PyTorch Geometric (Kaggle has PyTorch pre-installed)
print("Installing torch_geometric...")
os.system("pip install -q torch_geometric 2>/dev/null")
print("Done.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — Core Imports & Device Check
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# JAX — physics simulation engine
import jax
import jax.numpy as jnp
from jax import jit, random
import jax.lax as lax

# PyTorch — GNN training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Sklearn utilities
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_absolute_error

# ── Device report ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  JAX  version : {jax.__version__}")
print(f"  JAX  devices : {jax.devices()}")
print(f"  Torch version: {torch.__version__}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Torch device : {DEVICE}")
if DEVICE.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU name     : {props.name}")
    print(f"  GPU memory   : {props.total_memory / 1e9:.1f} GB")
print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — Experiment Parameters
# ══════════════════════════════════════════════════════════════════════════

# ── Physical (LJ reduced units: σ = ε = m = 1) ────────────────────────────
N_ATOMS   = 256
RHO       = 1.2                            # number density (standard LJ glass)
BOX_L     = float((N_ATOMS / RHO)**(1/3)) # ≈ 5.57 σ
RC_LJ     = 2.5                            # LJ potential cutoff
DT        = 5e-4                           # timestep τ

# ── Glass generation temperatures ────────────────────────────────────────
T_HIGH    = 2.0     # starting liquid state
T_LOW     = 0.10    # deep glass (T_g ≈ 0.45 for LJ at ρ=1.2)

# ── Cycling parameters (simulate battery charge/discharge) ────────────────
# PHYSICS FIX v2:
#   T=0.25 was too cold (T_g≈0.45): purely elastic regime, atoms spring back.
#   T=0.42 sits just below T_g — cooperative rearrangements activate and
#   permanent structural damage accumulates across cycles. This is the
#   physically correct regime for mechanical fatigue in a model glass.
#   1000 steps = 0.5τ per half-cycle gives proper relaxation time.
T_BATTERY    = 0.42  # just below T_g≈0.45 → plastic rearrangements active
STRAIN_AMP   = 0.08  # 8% volumetric strain (upper end of LiPON: 3-8%)
STEPS_PHASE  = 500   # 500 x dt=5e-4 = 0.25 tau per half-cycle
N_CYCLES     = 400   # 400 cycles: ~3% total drift expected vs 0.5% before
SAVE_AT      = [0, 50, 100, 200, 300, 400]  # snapshot cycle points

# ── Dataset settings ──────────────────────────────────────────────────────
N_GLASSES     = 100                    # independent glass instances
PRISTINE_CYC  = {0}                   # label 0: cycle 0 only (pristine)
FATIGUED_CYC  = {300, 400}            # label 1: heavily fatigued

# ── Graph construction ────────────────────────────────────────────────────
RC_GRAPH  = 1.5   # first coordination shell cutoff at ρ=1.2 (≈12 neighbours)

# ── JAX internal: steps per JIT-compiled scan chunk ──────────────────────
SCAN_CHUNK = 100   # must divide STEPS_PHASE evenly

# ── Feature extraction ───────────────────────────────────────────────────
# Set False first to confirm signal exists; set True to test robustness.
INSTANCE_NORM = False   # per-graph z-score normalization

# ── GNN training ──────────────────────────────────────────────────────────
HIDDEN_DIM  = 64
N_HEADS     = 4
BATCH_SIZE  = 32
LR          = 3e-4   # lower LR: subtle signal needs careful optimization
MAX_EPOCHS  = 150    # more epochs to find the subtle signal
PATIENCE    = 25     # more patience: loss may plateau before improving
N_FOLDS     = 5

# ── Output directory ──────────────────────────────────────────────────────
OUT_DIR = "/content/test"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Summary ───────────────────────────────────────────────────────────────
total_graphs = N_GLASSES * len(SAVE_AT)
clf_graphs   = N_GLASSES * (len(PRISTINE_CYC) + len(FATIGUED_CYC))
print(f"Box length    : {BOX_L:.3f} σ  (ρ = {RHO})")
print(f"Total glasses : {N_GLASSES}")
print(f"Snapshots/glass: {len(SAVE_AT)}  → {total_graphs} total graphs")
print(f"Classification: {clf_graphs} graphs  "
      f"({N_GLASSES*len(PRISTINE_CYC)} pristine + "
      f"{N_GLASSES*len(FATIGUED_CYC)} fatigued)")
print(f"Regression    : {total_graphs} graphs  (predict cycle 0–{N_CYCLES})")


# ══════════════════════════════════════════════════════════════════════════
# CELL 4 — JAX MD Engine
# Core physics: vectorized LJ forces + Brownian dynamics via lax.scan
# ══════════════════════════════════════════════════════════════════════════

@jit
def lj_forces_pbc(pos, box):
    """
    Vectorized LJ forces with minimum-image PBC.

    Args:
        pos : (N, 3) float32  — particle positions
        box : scalar float32  — cubic box side length

    Returns:
        forces : (N, 3) float32
    """
    # All pairwise displacement vectors: (N, N, 3)
    dr = pos[:, None, :] - pos[None, :, :]
    dr = dr - box * jnp.round(dr / box)   # minimum image convention

    r2 = jnp.sum(dr ** 2, axis=-1)        # (N, N) squared distances

    # Mask: exclude self-pairs (r2=0) and pairs beyond cutoff
    mask = (r2 > 1e-6) & (r2 < RC_LJ * RC_LJ)

    # Safe denominator (avoids NaN in masked positions)
    r2s   = jnp.where(mask, r2, 1.0)
    inv2  = 1.0 / r2s
    inv6  = inv2 ** 3
    inv12 = inv6 ** 2

    # LJ force coefficient: 24ε · [2(σ/r)¹² − (σ/r)⁶] / r²
    coeff = jnp.where(mask, 24.0 * inv2 * (2.0 * inv12 - inv6), 0.0)

    # Sum j contributions → net force on each i: (N, 3)
    forces = jnp.sum(coeff[:, :, None] * dr, axis=1)
    return forces


@jit
def md_chunk(pos, key, box, temperature):
    """
    Run SCAN_CHUNK Brownian dynamics steps.
    All arguments are JAX dynamic values → single compilation for any T/box.

    Args:
        pos         : (N, 3) float32
        key         : JAX PRNGKey
        box         : scalar float32  (box side length)
        temperature : scalar float32

    Returns:
        pos_out : (N, 3) float32
        key_out : PRNGKey
    """
    dt          = jnp.float32(DT)
    sqrt_2Tdt   = jnp.sqrt(2.0 * temperature * dt)  # noise amplitude

    def step(carry, _):
        p, k = carry
        # Compute and clamp forces
        f = lj_forces_pbc(p, box)
        f = jnp.clip(f, -50.0, 50.0)
        # Stochastic displacement
        k, sk   = random.split(k)
        noise   = random.normal(sk, p.shape, dtype=jnp.float32)
        p_new   = p + f * dt + sqrt_2Tdt * noise
        p_new   = p_new % box   # periodic wrap
        return (p_new, k), None

    (pos_out, key_out), _ = lax.scan(step, (pos, key), None, length=SCAN_CHUNK)
    return pos_out, key_out


def run_md(pos, key, box, temperature, total_steps):
    """
    Run arbitrary number of Brownian dynamics steps via repeated md_chunk calls.
    total_steps should be a multiple of SCAN_CHUNK.
    """
    box_f = jnp.float32(box)
    T_f   = jnp.float32(temperature)
    n_calls = max(1, total_steps // SCAN_CHUNK)
    for _ in range(n_calls):
        pos, key = md_chunk(pos, key, box_f, T_f)
    return pos, key


# ── JIT warm-up (trigger compilation once) ────────────────────────────────
print("Compiling JAX kernels (first run ~30-60 s)...")
t0        = time.time()
_dummy_p  = jnp.zeros((N_ATOMS, 3), dtype=jnp.float32)
_dummy_k  = random.PRNGKey(0)
_dp, _dk  = md_chunk(_dummy_p, _dummy_k,
                     jnp.float32(BOX_L), jnp.float32(T_LOW))
jax.block_until_ready(_dp)
print(f"  Compilation done in {time.time() - t0:.1f} s")


# ══════════════════════════════════════════════════════════════════════════
# CELL 5 — Glass Generation & Cycling Protocol
# ══════════════════════════════════════════════════════════════════════════

def init_on_lattice(key):
    """
    Place N_ATOMS on a slightly perturbed simple-cubic lattice.
    Much faster equilibration than random placement (avoids hard-core overlaps).
    """
    n_side  = int(np.ceil(N_ATOMS ** (1 / 3)))
    spacing = BOX_L / n_side
    pts     = np.array(
        [(i, j, k) for i in range(n_side)
                   for j in range(n_side)
                   for k in range(n_side)],
        dtype=np.float32
    )[:N_ATOMS] * spacing
    # Small random perturbation to break symmetry
    key, sk = random.split(key)
    pts     = pts + random.normal(sk, pts.shape, dtype=jnp.float32) * 0.05
    pts     = pts % jnp.float32(BOX_L)
    return pts, key


def fast_cool_glass(key, n_cool_chunks=40):
    """
    Generate a fast-cooled LJ glass via linear cooling T_HIGH → T_LOW.

    Protocol:
      1. Lattice initialization
      2. Hot equilibration at T_HIGH  (10 × SCAN_CHUNK steps)
      3. Linear cooling               (n_cool_chunks × SCAN_CHUNK steps)
      4. Final equilibration at T_LOW (20 × SCAN_CHUNK steps)

    Returns:
        positions : numpy (N, 3)
        key       : updated PRNGKey
    """
    box = jnp.float32(BOX_L)

    # 1. Lattice + hot equilibration
    pos, key = init_on_lattice(key)
    for _ in range(10):
        pos, key = md_chunk(pos, key, box, jnp.float32(T_HIGH))

    # 2. Linear cooling
    temps = np.linspace(T_HIGH, T_LOW, n_cool_chunks, dtype=np.float32)
    for T_val in temps:
        pos, key = md_chunk(pos, key, box, jnp.float32(T_val))

    # 3. Final equilibration at T_LOW
    for _ in range(20):
        pos, key = md_chunk(pos, key, box, jnp.float32(T_LOW))

    return np.array(pos), key


def cycle_once(pos, key):
    """
    Simulate one charge/discharge cycle:
      Charge  : affinely expand box+coords by STRAIN_AMP,
                run STEPS_PHASE NVT steps at T_BATTERY.
      Discharge: affinely compress back,
                 run STEPS_PHASE NVT steps at T_BATTERY.

    Affine scaling is the standard strain-application technique in MD.
    Instance normalization in feature extraction removes global volume shift,
    forcing the GNN to detect structural (not volumetric) changes.

    Returns:
        positions : numpy (N, 3) in original box
        key       : updated PRNGKey
    """
    L_exp  = jnp.float32(BOX_L * (1.0 + STRAIN_AMP))
    L_orig = jnp.float32(BOX_L)
    T      = jnp.float32(T_BATTERY)
    scale  = jnp.float32(1.0 + STRAIN_AMP)

    # ── Charge: expand ───────────────────────────────────────────────────
    pos_j   = jnp.array(pos) * scale          # affine coordinate scaling
    pos_j, key = run_md(pos_j, key, L_exp, T, STEPS_PHASE)

    # ── Discharge: compress ──────────────────────────────────────────────
    pos_j   = pos_j / scale                   # affine compression
    pos_j, key = run_md(pos_j, key, L_orig, T, STEPS_PHASE)

    return np.array(pos_j), key


def generate_dataset(save_path=None):
    """
    Main data generation loop.

    Returns:
        snapshots : list of dicts with keys
                    {'glass_id', 'cycle', 'positions'}
    """
    snapshots   = []
    master_key  = random.PRNGKey(2024)
    save_set    = set(SAVE_AT)
    t_start     = time.time()

    print(f"\nGenerating {N_GLASSES} glasses × {N_CYCLES} cycles "
          f"× {len(SAVE_AT)} snapshots ...")

    for gid in range(N_GLASSES):
        master_key, subkey = random.split(master_key)

        # ── Step 1: generate pristine glass ──────────────────────────────
        pos, key = fast_cool_glass(subkey)

        # Save cycle-0 snapshot
        if 0 in save_set:
            snapshots.append({'glass_id': gid, 'cycle': 0,
                              'positions': pos.copy()})

        # ── Step 2: run cycling protocol ─────────────────────────────────
        for cyc in range(1, N_CYCLES + 1):
            pos, key = cycle_once(pos, key)
            if cyc in save_set:
                snapshots.append({'glass_id': gid, 'cycle': cyc,
                                  'positions': pos.copy()})

        # ── Progress report every 10 glasses ─────────────────────────────
        if (gid + 1) % 10 == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / (gid + 1) * (N_GLASSES - gid - 1)
            print(f"  [{gid+1:3d}/{N_GLASSES}]  "
                  f"elapsed {elapsed/60:.1f} min  |  "
                  f"ETA {eta/60:.1f} min  |  "
                  f"snapshots so far: {len(snapshots)}")

    # ── Optional: save to disk for notebook restart safety ────────────────
    if save_path:
        np.save(save_path, snapshots)
        print(f"\n  Saved {len(snapshots)} snapshots → {save_path}")

    total_time = time.time() - t_start
    print(f"\nData generation complete: {len(snapshots)} snapshots "
          f"in {total_time/60:.1f} min")
    return snapshots


# ══════════════════════════════════════════════════════════════════════════
# CELL 6 — Feature Extraction (5D Bond-Length Statistics)
# Identical pipeline to KA cross-system validation in the thermal paper.
# ══════════════════════════════════════════════════════════════════════════

def extract_5d_features(positions, box_length=BOX_L, rc=RC_GRAPH):
    """
    Compute 5D pure-geometry node features with instance normalization.

    Feature vector per particle i:
        [r̄_i, σ_r,i, r_min,i, r_max,i, d̃_i]
    where r̄_i = mean bond length, σ_r = std, d̃_i = normalised coordination.

    Instance normalisation (per-graph mean/std subtraction) removes global
    volume differences, forcing the GNN to detect relational geometry.

    Returns:
        node_feat  : (N, 5)  float32
        edge_index : (2, E)  int64
        edge_attr  : (E, 1)  float32   (raw bond lengths, pre-normalisation)
    """
    N   = len(positions)
    pos = np.array(positions, dtype=np.float32)

    # ── Pairwise distances with PBC minimum image ─────────────────────────
    dr   = pos[:, None, :] - pos[None, :, :]                  # (N, N, 3)
    dr   = dr - box_length * np.round(dr / box_length)
    dist = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr))          # (N, N)

    # ── Neighbour mask ────────────────────────────────────────────────────
    nbr_mask = (dist > 1e-6) & (dist < rc)                    # (N, N) bool

    # ── Per-particle statistics ───────────────────────────────────────────
    feats = np.zeros((N, 5), dtype=np.float32)
    coords = nbr_mask.sum(axis=1).astype(np.float32)           # (N,)
    d_max  = coords.max() if coords.max() > 0 else 1.0

    for i in range(N):
        nbr_dists = dist[i][nbr_mask[i]]
        if len(nbr_dists) == 0:
            feats[i] = [rc, 0.0, rc, rc, 0.0]
            continue
        feats[i, 0] = nbr_dists.mean()                        # mean bond length
        feats[i, 1] = nbr_dists.std() if len(nbr_dists) > 1 else 0.0
        feats[i, 2] = nbr_dists.min()                         # min bond length
        feats[i, 3] = nbr_dists.max()                         # max bond length
        feats[i, 4] = coords[i] / d_max                       # normalised coord

    # ── Add 3 extra features: skewness proxy, Q25, Q75 ─────────────────────
    # These capture asymmetric bond-length damage that the 5D set misses.
    # Skewness: mean(r³) - 3·mean(r)·var(r) - mean(r)³  (simplified proxy)
    extra = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        nbr_dists = dist[i][nbr_mask[i]]
        if len(nbr_dists) < 3:
            extra[i] = [0.0, feats[i,2], feats[i,3]]
            continue
        q25 = np.percentile(nbr_dists, 25)
        q75 = np.percentile(nbr_dists, 75)
        skew_proxy = float(np.mean((nbr_dists - nbr_dists.mean())**3) /
                           (nbr_dists.std()**3 + 1e-8))
        extra[i] = [np.clip(skew_proxy, -5, 5), q25, q75]
    feats = np.concatenate([feats, extra], axis=1)  # → (N, 8)

    # ── Instance normalisation (per-graph z-score) ────────────────────────
    if INSTANCE_NORM:
        mu    = feats.mean(axis=0, keepdims=True)
        sigma = feats.std(axis=0, keepdims=True) + 1e-8
        feats = (feats - mu) / sigma

    # ── Edge construction ─────────────────────────────────────────────────
    rows, cols = np.where(nbr_mask)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = dist[rows, cols].reshape(-1, 1).astype(np.float32)

    return feats, edge_index, edge_attr


def build_pyg_datasets(snapshots):
    """
    Convert raw snapshots → two PyG datasets.

    clf_data  : list[Data]  — classification (label 0/1)
    reg_data  : list[Data]  — regression (label = cycle / N_CYCLES ∈ [0,1])
    """
    clf_data, reg_data = [], []
    nan_count = 0

    print("\nExtracting features & building graphs...")
    t0 = time.time()

    for i, snap in enumerate(snapshots):
        cyc = snap['cycle']
        pos = snap['positions']

        # ── Feature extraction ────────────────────────────────────────────
        nf, ei, ea = extract_5d_features(pos)

        # Skip NaN/Inf samples
        if not (np.isfinite(nf).all() and np.isfinite(ea).all()):
            nan_count += 1
            continue

        ei_t = torch.from_numpy(ei).long()
        ea_t = torch.from_numpy(ea).float()
        x_t  = torch.from_numpy(nf).float()

        # ── Classification dataset ────────────────────────────────────────
        if cyc in PRISTINE_CYC:
            clf_data.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, y=0))
        elif cyc in FATIGUED_CYC:
            clf_data.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, y=1))

        # ── Regression dataset (all cycles, normalised target) ────────────
        reg_label = torch.tensor([cyc / N_CYCLES], dtype=torch.float32)
        reg_data.append(
            Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, y=reg_label,
                 cycle=cyc, glass_id=snap['glass_id'])
        )

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(snapshots)} snapshots "
                  f"({nan_count} NaN skipped)")

    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"clf: {len(clf_data)} graphs, reg: {len(reg_data)} graphs, "
          f"NaN skipped: {nan_count}")
    return clf_data, reg_data


# ══════════════════════════════════════════════════════════════════════════
# CELL 7 — GATv2 Models
# Identical architecture to the thermal-history paper for fair comparison.
# ══════════════════════════════════════════════════════════════════════════

class GATv2Classifier(nn.Module):
    """
    GATv2 graph classifier (binary: pristine vs. fatigued).
    Architecture mirrors the thermal-history paper exactly.
    """
    def __init__(self, in_dim=8, hidden=HIDDEN_DIM, heads=N_HEADS):
        super().__init__()
        self.enc   = nn.Linear(in_dim, hidden)

        self.gat1  = GATv2Conv(hidden,   hidden, heads=heads,
                               edge_dim=1, concat=True)   # → hidden*heads
        self.gat2  = GATv2Conv(hidden * heads, hidden, heads=heads,
                               edge_dim=1, concat=True)   # → hidden*heads

        pool_dim   = hidden * heads
        self.post  = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head  = nn.Linear(64, 1)   # BCEWithLogitsLoss expects logit

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.enc(x))
        x = F.relu(self.gat1(x, ei, ea))
        x = F.relu(self.gat2(x, ei, ea))
        x = global_mean_pool(x, batch)
        x = self.post(x)
        return self.head(x).squeeze(-1)   # (B,)


class GATv2Regressor(nn.Module):
    """
    GATv2 regression model: predicts normalised cycle number ∈ [0, 1].
    Identical backbone to classifier; only the output head differs.
    """
    def __init__(self, in_dim=8, hidden=HIDDEN_DIM, heads=N_HEADS):
        super().__init__()
        self.enc   = nn.Linear(in_dim, hidden)

        self.gat1  = GATv2Conv(hidden,       hidden, heads=heads,
                               edge_dim=1, concat=True)
        self.gat2  = GATv2Conv(hidden*heads, hidden, heads=heads,
                               edge_dim=1, concat=True)

        pool_dim   = hidden * heads
        self.post  = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head  = nn.Linear(64, 1)    # scalar regression

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.enc(x))
        x = F.relu(self.gat1(x, ei, ea))
        x = F.relu(self.gat2(x, ei, ea))
        x = global_mean_pool(x, batch)
        x = self.post(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)  # bounded [0,1]


# ══════════════════════════════════════════════════════════════════════════
# CELL 8 — Training Utilities
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch_clf(model, loader, optimizer, scaler):
    """Binary classification training step with mixed-precision AMP."""
    model.train()
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            logits = model(batch)
            labels = batch.y.float()
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_clf(model, loader):
    """Evaluate binary classifier; return loss, accuracy, AUC, F1."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).float()
    loss   = criterion(logits, labels).item()
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)
    labs   = labels.numpy().astype(int)

    acc  = (preds == labs).mean() * 100
    auc  = roc_auc_score(labs, probs)
    f1   = f1_score(labs, preds)
    return loss, acc, auc, f1


def train_one_epoch_reg(model, loader, optimizer, scaler):
    """Regression training step."""
    model.train()
    total_loss = 0.0
    criterion  = nn.MSELoss()

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            preds = model(batch)
            loss  = criterion(preds, batch.y.squeeze(-1).float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_reg(model, loader):
    """Evaluate regression model; return MSE, MAE, R²."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        preds = model(batch)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.squeeze(-1).cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    mse = ((preds - labels) ** 2).mean()
    mae = np.abs(preds - labels).mean()
    r2  = r2_score(labels, preds)
    return mse, mae, r2


# ══════════════════════════════════════════════════════════════════════════
# CELL 9 — Experiment 1: Classification (Pristine vs. Fatigued)
# ══════════════════════════════════════════════════════════════════════════

def run_classification(clf_data):
    """
    5-fold stratified cross-validation for binary classification.
    Reports accuracy, AUC, F1 per fold.
    """
    print(f"\n{'='*55}")
    print(f"  EXPERIMENT 1: Classification")
    print(f"  Pristine (cy {sorted(PRISTINE_CYC)}) vs "
          f"Fatigued (cy {sorted(FATIGUED_CYC)})")
    print(f"  {len(clf_data)} graphs | {N_FOLDS}-fold stratified CV")
    print(f"{'='*55}")

    labels = np.array([d.y for d in clf_data])
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    all_train_curves, all_val_curves = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labels, labels)):
        print(f"\n  ── Fold {fold_idx + 1}/{N_FOLDS} ──────────────────────")

        train_data = [clf_data[i] for i in train_idx]
        val_data   = [clf_data[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=2)

        model     = GATv2Classifier().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
        scaler    = GradScaler(enabled=(DEVICE.type == 'cuda'))

        best_val_loss = float('inf')   # FIXED: track loss not accuracy
        best_val_acc  = 0.0
        best_state    = None
        patience_ctr  = 0
        train_losses, val_accs = [], []

        for epoch in range(1, MAX_EPOCHS + 1):
            t_loss              = train_one_epoch_clf(model, train_loader,
                                                      optimizer, scaler)
            v_loss, v_acc, v_auc, v_f1 = eval_clf(model, val_loader)
            scheduler.step()

            train_losses.append(t_loss)
            val_accs.append(v_acc)

            # FIXED: use val_loss for early stopping, not val_acc.
            # val_acc at epoch 1 is always 50% (random init → predict all-same),
            # so tracking acc causes immediate false-best and premature stopping.
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_val_acc  = v_acc
                best_state    = {k: v.clone() for k, v in
                                 model.state_dict().items()}
                best_epoch    = epoch
                best_auc      = v_auc
                best_f1       = v_f1
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if epoch % 20 == 0:
                print(f"    ep {epoch:3d} | train_loss {t_loss:.4f} | "
                      f"val_loss {v_loss:.4f} | val_acc {v_acc:.1f}% | "
                      f"val_AUC {v_auc:.4f}")

            if patience_ctr >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

        fold_results.append({
            'fold':       fold_idx + 1,
            'best_acc':   best_val_acc,
            'best_auc':   best_auc,
            'best_f1':    best_f1,
            'best_epoch': best_epoch,
        })
        all_train_curves.append(train_losses)
        all_val_curves.append(val_accs)

        print(f"  ✓ Fold {fold_idx+1}: acc={best_val_acc:.2f}%  "
              f"AUC={best_auc:.4f}  F1={best_f1:.4f}  "
              f"(best epoch {best_epoch})")

    # ── Aggregate results ──────────────────────────────────────────────────
    accs = [r['best_acc'] for r in fold_results]
    aucs = [r['best_auc'] for r in fold_results]
    f1s  = [r['best_f1']  for r in fold_results]

    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION RESULTS  (5-fold CV)")
    print(f"{'─'*55}")
    print(f"  {'Fold':<6} {'Acc (%)':>8} {'AUC':>8} {'F1':>8}")
    print(f"{'─'*55}")
    for r in fold_results:
        print(f"  {r['fold']:<6} {r['best_acc']:>8.2f} "
              f"{r['best_auc']:>8.4f} {r['best_f1']:>8.4f}")
    print(f"{'─'*55}")
    print(f"  {'Mean':<6} {np.mean(accs):>8.2f}±{np.std(accs):.2f}  "
          f"{np.mean(aucs):>8.4f}±{np.std(aucs):.4f}  "
          f"{np.mean(f1s):>8.4f}±{np.std(f1s):.4f}")
    print(f"{'='*55}")

    return fold_results, all_train_curves, all_val_curves


# ══════════════════════════════════════════════════════════════════════════
# CELL 10 — Experiment 2: Regression (Predict Cycle Number)
# ══════════════════════════════════════════════════════════════════════════

def run_regression(reg_data):
    """
    5-fold CV regression: predict normalised cycle number ∈ [0,1].
    Strong R² signals monotonic structural change with cycling.
    """
    print(f"\n{'='*55}")
    print(f"  EXPERIMENT 2: Regression — Predict Cycle Number")
    print(f"  {len(reg_data)} graphs | {N_FOLDS}-fold CV")
    print(f"{'='*55}")

    indices = np.arange(len(reg_data))
    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n  ── Fold {fold_idx + 1}/{N_FOLDS} ──────────────────────")

        train_data = [reg_data[i] for i in train_idx]
        val_data   = [reg_data[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=2)

        model     = GATv2Regressor().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
        scaler    = GradScaler(enabled=(DEVICE.type == 'cuda'))

        best_r2    = -np.inf
        best_state = None
        patience_ctr = 0
        best_mse, best_mae = None, None

        for epoch in range(1, MAX_EPOCHS + 1):
            train_one_epoch_reg(model, train_loader, optimizer, scaler)
            v_mse, v_mae, v_r2 = eval_reg(model, val_loader)
            scheduler.step()

            if v_r2 > best_r2:
                best_r2    = v_r2
                best_mse   = v_mse
                best_mae   = v_mae
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 20 == 0:
                print(f"    ep {epoch:3d} | val_MSE {v_mse:.4f} "
                      f"val_MAE {v_mae:.4f} val_R² {v_r2:.4f}")

            if patience_ctr >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

        fold_results.append({
            'fold': fold_idx+1,
            'r2':   best_r2,
            'mse':  best_mse,
            'mae':  best_mae,
            'epoch': best_epoch,
        })
        print(f"  ✓ Fold {fold_idx+1}: R²={best_r2:.4f}  "
              f"MSE={best_mse:.4f}  MAE={best_mae:.4f}")

    r2s  = [r['r2']  for r in fold_results]
    mses = [r['mse'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]

    print(f"\n{'='*55}")
    print(f"  REGRESSION RESULTS  (5-fold CV)")
    print(f"{'─'*55}")
    print(f"  {'Fold':<6} {'R²':>8} {'MSE':>8} {'MAE':>8}")
    print(f"{'─'*55}")
    for r in fold_results:
        print(f"  {r['fold']:<6} {r['r2']:>8.4f} "
              f"{r['mse']:>8.4f} {r['mae']:>8.4f}")
    print(f"{'─'*55}")
    print(f"  {'Mean':<6} {np.mean(r2s):>8.4f}±{np.std(r2s):.4f}  "
          f"{np.mean(mses):>8.4f}±{np.std(mses):.4f}  "
          f"{np.mean(maes):>8.4f}±{np.std(maes):.4f}")
    print(f"{'='*55}")

    return fold_results


# ══════════════════════════════════════════════════════════════════════════
# CELL 11 — Visualizations
# ══════════════════════════════════════════════════════════════════════════

def plot_force_chains(pristine_pos, fatigued_pos, out_path, percentile=85):
    """
    Reproduce force-chain visualization from thermal-history paper.
    High-strain bonds (top 15%) shown in 3D; slow-cycled glass should show
    more extended, organized stress patterns.
    """
    fig = plt.figure(figsize=(14, 6))
    titles = ['Pristine (Cycle 0)', f'Fatigued (Cycle {N_CYCLES})']
    color_map = cm.plasma

    for ax_idx, (pos, title) in enumerate(
            zip([pristine_pos, fatigued_pos], titles)):

        ax = fig.add_subplot(1, 2, ax_idx + 1, projection='3d')
        ax.set_title(title, fontsize=13, pad=12)

        p = np.array(pos, dtype=np.float32)

        # Volume-normalise: scale so mean bond length = 1.0
        dr   = p[:, None, :] - p[None, :, :]
        dr   = dr - BOX_L * np.round(dr / BOX_L)
        dist = np.sqrt(np.sum(dr**2, axis=-1))
        mask = (dist > 1e-6) & (dist < RC_GRAPH)
        mean_bond = dist[mask].mean()
        p   /= mean_bond

        # Recompute distances in normalised frame
        box_n = BOX_L / mean_bond
        dr    = p[:, None, :] - p[None, :, :]
        dr    = dr - box_n * np.round(dr / box_n)
        dist  = np.sqrt(np.sum(dr**2, axis=-1))
        mask  = (dist > 1e-6) & (dist < RC_GRAPH / mean_bond)

        # Strain from equilibrium (unit bond)
        strain = np.abs(dist - 1.0)

        # Top-percentile bonds
        thresh = np.percentile(strain[mask], percentile)
        rows, cols = np.where(mask & (strain > thresh))

        # Colour by strain magnitude
        s_vals = strain[rows, cols]
        s_norm = (s_vals - s_vals.min()) / ((s_vals.max() - s_vals.min()) + 1e-8)

        for ri, ci, sv in zip(rows, cols, s_norm):
            xs = [p[ri, 0], p[ci, 0]]
            ys = [p[ri, 1], p[ci, 1]]
            zs = [p[ri, 2], p[ci, 2]]
            ax.plot(xs, ys, zs, '-', color=color_map(sv), alpha=0.6, lw=0.8)

        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)

    plt.suptitle('Strain Topology: Cyclic Battery Fatigue\n'
                 '(Top 15% strained bonds, volume-normalised)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Force chain plot saved → {out_path}")


def plot_learning_curves(train_curves, val_curves, title, out_path):
    """Plot loss / accuracy curves for all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.5, N_FOLDS))

    for fold_i, (tc, vc) in enumerate(zip(train_curves, val_curves)):
        axes[0].plot(tc, color=colors[fold_i], alpha=0.8,
                     label=f'Fold {fold_i+1}')
        axes[1].plot(vc, color=colors[fold_i], alpha=0.8,
                     label=f'Fold {fold_i+1}')

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCEWithLogitsLoss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Learning curves saved → {out_path}")


def plot_cycle_distribution(snapshots, out_path):
    """
    Per-cycle bond-length statistics to confirm structural drift with cycling.
    If fatigue encodes a geometric signature, mean/std bond length should
    show a systematic trend across cycles.
    """
    cycle_vals = sorted(set(s['cycle'] for s in snapshots))
    means, stds, mins, maxs = [], [], [], []

    for cyc in cycle_vals:
        cyc_snaps = [s for s in snapshots if s['cycle'] == cyc]
        all_bonds = []
        for snap in cyc_snaps[:20]:   # first 20 glasses per cycle
            pos = snap['positions']
            dr  = pos[:, None, :] - pos[None, :, :]
            dr  = dr - BOX_L * np.round(dr / BOX_L)
            dist = np.sqrt(np.sum(dr**2, axis=-1))
            mask = (dist > 1e-6) & (dist < RC_GRAPH)
            all_bonds.extend(dist[mask].tolist())
        arr = np.array(all_bonds)
        means.append(arr.mean())
        stds.append(arr.std())
        mins.append(arr.min())
        maxs.append(arr.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].errorbar(cycle_vals, means, yerr=stds, fmt='o-', capsize=4,
                     color='steelblue', label='Mean ± std')
    axes[0].set_xlabel('Cycle number')
    axes[0].set_ylabel('Bond length (σ)')
    axes[0].set_title('Mean Bond Length vs Cycle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cycle_vals, stds, 's--', color='tomato', label='Std bond length')
    axes[1].set_xlabel('Cycle number')
    axes[1].set_ylabel('Bond length std (σ)')
    axes[1].set_title('Bond Length Disorder vs Cycle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Structural Drift Under Cyclic Strain\n'
                 '(averaged over first 20 glasses per cycle point)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bond statistics plot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 12 — MAIN: Run All Experiments
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # ────────────────────────────────────────────────────────────────────
    # Phase 1: Data Generation (JAX)
    # ────────────────────────────────────────────────────────────────────
    snap_path = os.path.join(OUT_DIR, "battery_snapshots.npy")

    # NOTE: always regenerate — parameters changed (T_BATTERY, STEPS_PHASE)
    # Delete battery_snapshots.npy manually if you want to use old data.
    snapshots = generate_dataset(save_path=snap_path)

    # ────────────────────────────────────────────────────────────────────
    # Phase 2: Feature Extraction & Graph Construction (NumPy)
    # ────────────────────────────────────────────────────────────────────
    clf_data, reg_data = build_pyg_datasets(snapshots)

    # ── Structural drift diagnostic ─────────────────────────────────────
    # CRITICAL CHECK: if mean bond length does NOT shift across cycles,
    # the cycling protocol is not creating detectable structural change
    # and classification will be near-random regardless of model.
    print("\nStructural drift check (mean bond length per cycle):")
    print(f"  {'Cycle':>6}  {'Mean bond':>10}  {'Std bond':>10}  {'N bonds':>8}")
    print(f"  {'─'*42}")
    cycle_vals_check = sorted(set(s['cycle'] for s in snapshots))
    for cyc in cycle_vals_check:
        cyc_pos = [s['positions'] for s in snapshots
                   if s['cycle'] == cyc][:20]
        all_bonds = []
        for pos in cyc_pos:
            dr   = pos[:, None, :] - pos[None, :, :]
            dr   = dr - BOX_L * np.round(dr / BOX_L)
            dist = np.sqrt(np.sum(dr**2, axis=-1))
            mask = (dist > 1e-6) & (dist < RC_GRAPH)
            all_bonds.extend(dist[mask].tolist())
        arr = np.array(all_bonds)
        print(f"  {cyc:>6}  {arr.mean():>10.5f}  {arr.std():>10.5f}  {len(arr):>8}")
    print(f"  {'─'*42}")
    print("  If mean bond length is FLAT across cycles → T_BATTERY too low.")
    print("  If mean bond length DRIFTS → structural fatigue is accumulating.\n")

    # ────────────────────────────────────────────────────────────────────
    # Phase 3: Visualise structural drift BEFORE GNN training
    # ────────────────────────────────────────────────────────────────────
    print("\nGenerating structural diagnostics...")
    plot_cycle_distribution(
        snapshots,
        os.path.join(OUT_DIR, "bond_stats_vs_cycle.png")
    )

    # Force chain comparison: cycle 0 vs cycle 50
    prist_pos = next(s['positions'] for s in snapshots
                     if s['cycle'] == 0 and s['glass_id'] == 0)
    fat_pos   = next((s['positions'] for s in snapshots
                      if s['cycle'] == N_CYCLES and s['glass_id'] == 0),
                     None)
    if fat_pos is not None:
        plot_force_chains(
            prist_pos, fat_pos,
            os.path.join(OUT_DIR, "force_chains_battery.png")
        )

    # ────────────────────────────────────────────────────────────────────
    # Phase 4: Experiment 1 — Classification (PyTorch)
    # ────────────────────────────────────────────────────────────────────
    # Free JAX GPU memory before PyTorch training
    gc.collect()

    clf_results, train_curves, val_curves = run_classification(clf_data)

    plot_learning_curves(
        train_curves, val_curves,
        title=f"Classification: Pristine vs Fatigued Glass\n"
              f"(GATv2, 5D bond features, 5-fold CV)",
        out_path=os.path.join(OUT_DIR, "clf_learning_curves.png")
    )

    # ────────────────────────────────────────────────────────────────────
    # Phase 5: Experiment 2 — Regression (PyTorch)
    # ────────────────────────────────────────────────────────────────────
    reg_results = run_regression(reg_data)

    # ────────────────────────────────────────────────────────────────────
    # Phase 6: Final Summary
    # ────────────────────────────────────────────────────────────────────
    clf_acc  = np.mean([r['best_acc'] for r in clf_results])
    clf_std  = np.std( [r['best_acc'] for r in clf_results])
    clf_auc  = np.mean([r['best_auc'] for r in clf_results])
    reg_r2   = np.mean([r['r2']       for r in reg_results])
    reg_r2s  = np.std( [r['r2']       for r in reg_results])

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY")
    print(f"{'─'*55}")
    print(f"  System        : LJ glass, N={N_ATOMS}, ρ={RHO}, "
          f"T_cycle={T_BATTERY}")
    print(f"  Strain        : {STRAIN_AMP*100:.0f}%  "
          f"({N_CYCLES} cycles)")
    print(f"  Features      : 8D bond stats (INSTANCE_NORM={INSTANCE_NORM})")
    print(f"  Model         : GATv2 (2 layers, {N_HEADS} heads)")
    print(f"{'─'*55}")
    print(f"  Exp 1  Acc    : {clf_acc:.2f} ± {clf_std:.2f} %")
    print(f"  Exp 1  AUC    : {clf_auc:.4f}")
    print(f"  Exp 2  R²     : {reg_r2:.4f} ± {reg_r2s:.4f}")
    print(f"{'─'*55}")
    print(f"  Total runtime : {total_min:.1f} minutes")
    print(f"  Output dir    : {OUT_DIR}")
    print(f"{'='*55}")

    print("\nSaved files:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size  = os.path.getsize(fpath) / 1e3
        print(f"  {fname:<45} {size:>8.1f} KB")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
=============================================================================
KOVACS MEMORY EFFECT — ANALYSIS & GNN STRUCTURAL PROBE  (v3 — fast)
=============================================================================
Key fix: replaced O(N²) full distance matrix with a cell-list / KD-tree
neighbor search. For N=10000 atoms this reduces work from 100M pairs to
~120K pairs per frame — about 800x faster.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
import os, time, warnings
warnings.filterwarnings("ignore")

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
    print("PyTorch Geometric available — GNN inference enabled.")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found — structural analysis only.")

RC_GRAPH = 1.5   # first-shell cutoff (σ units)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PE TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def load_thermo(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n[ERROR] {path} not found.\n")
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    data.append(float(line))
                except ValueError:
                    pass
    arr = np.array(data, dtype=np.float64)
    print(f"  Loaded {path}: {len(arr)} points, "
          f"PE range [{arr.min():.4f}, {arr.max():.4f}]")
    return arr


def compute_delta_pe(kovacs_pe, ref_pe):
    n         = min(len(kovacs_pe), len(ref_pe))
    tail      = int(0.80 * len(ref_pe))
    pe_ref_eq = ref_pe[tail:].mean()
    return kovacs_pe[:n] - pe_ref_eq, ref_pe[:n], pe_ref_eq, n


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KOVACS PEAK
# ─────────────────────────────────────────────────────────────────────────────

def find_kovacs_peak(delta_pe, step_interval=100):
    window = max(11, min(101, (len(delta_pe) // 20) * 2 + 1))
    smooth = savgol_filter(delta_pe, window_length=window, polyorder=3)
    d_smooth = np.gradient(smooth)
    skip     = 5
    inflect  = skip + int(np.argmax(np.abs(d_smooth[skip:])))
    steps    = np.arange(len(delta_pe)) * step_interval

    tau_est = None
    try:
        def kww(t, A, tau, beta, C):
            return A * np.exp(-(t / (tau + 1e-10)) ** beta) + C
        A0   = delta_pe[0]
        tau0 = max(float(steps[max(len(delta_pe) // 10, 1)]), 1000.0)
        p0   = [A0, tau0, 0.7, delta_pe[-1]]
        bnds = ([-np.inf, 1, 0.1, -np.inf], [0.0, steps[-1]*3, 1.0, np.inf])
        popt, _ = curve_fit(kww, steps, delta_pe, p0=p0,
                            bounds=bnds, maxfev=10000)
        tau_est = popt[1]
        print(f"  KWW: A={popt[0]:.4f}, τ={popt[1]:.1f} steps "
              f"({popt[1]*0.005:.2f} τ_LJ), β={popt[2]:.3f}, C={popt[3]:.4f}")
    except Exception as e:
        print(f"  KWW fit failed ({e})")

    result = dict(
        peak_idx=inflect, peak_step=int(inflect * step_interval),
        delta_at_peak=float(smooth[inflect]),
        final_delta=float(delta_pe[-1]),
        initial_delta=float(delta_pe[0]),
        relaxation_tau=tau_est, smooth=smooth, steps=steps,
    )
    print(f"  Initial ΔPE: {result['initial_delta']:+.5f}  |  "
          f"Inflection @ step {result['peak_step']:,}  |  "
          f"Final ΔPE: {result['final_delta']:+.5f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — OVITO METRICS
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_metrics():
    metrics = {}
    for fname, key in [("metric_d2min.txt","d2min"),("metric_shear.txt","shear")]:
        if os.path.exists(fname):
            metrics[key] = np.loadtxt(fname)
            print(f"  Loaded {fname}: {len(metrics[key])} frames")
        else:
            print(f"  [WARN] {fname} not found")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LAMMPS DUMP PARSER  (state machine)
# ─────────────────────────────────────────────────────────────────────────────

def parse_lammps_dump(dump_path, max_frames=None):
    if not os.path.exists(dump_path):
        print(f"  [WARN] {dump_path} not found.")
        return []

    SEEK=0; READ_STEP=1; READ_NATOMS=2; READ_BOX=3; READ_ATOMS=4
    frames=[]; state=SEEK; step=0; n_atoms=0; box_lines=[]; positions=[]

    print(f"  Parsing {dump_path} ...", end=" ", flush=True)
    t0 = time.time()
    with open(dump_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("ITEM: TIMESTEP"):
                if positions and len(positions) == n_atoms > 0:
                    try: lo,hi = map(float, box_lines[0].split()[:2]); box=hi-lo
                    except: box=20.274
                    frames.append({'step':step,'box':box,
                                   'positions':np.array(positions,dtype=np.float32)})
                    if max_frames and len(frames)>=max_frames: break
                state=READ_STEP; positions=[]; box_lines=[]
            elif line.startswith("ITEM: NUMBER OF ATOMS"): state=READ_NATOMS
            elif line.startswith("ITEM: BOX BOUNDS"):       state=READ_BOX
            elif line.startswith("ITEM: ATOMS"):            state=READ_ATOMS
            elif state==READ_STEP:
                try: step=int(line)
                except: pass
                state=SEEK
            elif state==READ_NATOMS:
                try: n_atoms=int(line)
                except: pass
                state=SEEK
            elif state==READ_BOX:
                box_lines.append(line)
                if len(box_lines)>=3: state=SEEK
            elif state==READ_ATOMS:
                p = line.split()
                if len(p)>=5:
                    try: positions.append([float(p[2]),float(p[3]),float(p[4])])
                    except: pass

    if positions and n_atoms>0 and len(positions)==n_atoms:
        try: lo,hi=map(float,box_lines[0].split()[:2]); box=hi-lo
        except: box=20.274
        if not max_frames or len(frames)<max_frames:
            frames.append({'step':step,'box':box,
                           'positions':np.array(positions,dtype=np.float32)})

    elapsed = time.time()-t0
    print(f"{len(frames)} frames in {elapsed:.1f}s")
    if frames:
        print(f"    Steps: {frames[0]['step']} … {frames[-1]['step']}, "
              f"box={frames[0]['box']:.3f}")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FAST BOND STATS  (cKDTree — O(N) not O(N²))
# ─────────────────────────────────────────────────────────────────────────────

def extract_bond_stats_fast(positions, box, rc=RC_GRAPH):
    """
    Use scipy cKDTree with periodic images to find neighbours in O(N log N).
    For N=10000 this takes ~30 ms vs ~8 s for the full distance matrix.
    """
    pos = np.array(positions, dtype=np.float64)

    # Wrap into [0, box)
    pos = pos % box

    # Build tree on 3×3×3 periodic images to handle PBC correctly
    # Only needed within one shell — since rc << box/2 we can use
    # the boxed coords + query_ball_tree approach instead.
    tree = cKDTree(pos, boxsize=box)   # boxsize enables periodic query

    # For each atom, get neighbours within rc (excluding self)
    all_bonds = []
    idx_pairs = tree.query_pairs(rc, output_type='ndarray')  # (M, 2) array

    if len(idx_pairs) == 0:
        return {'mean_r':0,'std_r':0,'skewness':0,'q25':0,'q75':0,
                'r_max':0,'n_bonds':0}

    i_idx = idx_pairs[:, 0]
    j_idx = idx_pairs[:, 1]

    dr   = pos[i_idx] - pos[j_idx]
    # Minimum image convention
    dr   -= box * np.round(dr / box)
    dist = np.sqrt((dr**2).sum(axis=1))

    # Filter: already within rc by construction, but re-check PBC dist
    valid = dist < rc
    dist  = dist[valid]

    if len(dist) == 0:
        return {'mean_r':0,'std_r':0,'skewness':0,'q25':0,'q75':0,
                'r_max':0,'n_bonds':0}

    mu   = dist.mean()
    sig  = dist.std()
    skew = (float(np.mean((dist - mu)**3) / (sig**3 + 1e-8))
            if len(dist) > 2 else 0.0)

    return {
        'mean_r':   float(mu),
        'std_r':    float(sig),
        'skewness': float(np.clip(skew, -5, 5)),
        'q25':      float(np.percentile(dist, 25)),
        'q75':      float(np.percentile(dist, 75)),
        'r_max':    float(dist.max()),
        'n_bonds':  int(len(dist)),
    }


def extract_structural_trajectory(frames, rc=RC_GRAPH):
    """Process all frames with progress reporting."""
    if not frames:
        return {}

    keys    = ['step','mean_r','std_r','skewness','q25','q75','r_max']
    metrics = {k: [] for k in keys}

    print(f"  Extracting bond stats from {len(frames)} frames "
          f"(cKDTree, rc={rc})...")
    t0 = time.time()

    for fi, fr in enumerate(frames):
        stats = extract_bond_stats_fast(fr['positions'], fr['box'], rc)
        metrics['step'].append(fr['step'])
        for k in keys[1:]:
            metrics[k].append(stats[k])

        if (fi + 1) % 50 == 0 or fi == len(frames)-1:
            elapsed = time.time()-t0
            eta     = elapsed/(fi+1)*(len(frames)-fi-1)
            print(f"    Frame {fi+1:>3}/{len(frames)}  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s  "
                  f"std_r={stats['std_r']:.5f}")

    print(f"  Done in {time.time()-t0:.1f}s")
    return {k: np.array(v) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GNN GRAPH BUILDER  (also uses cKDTree)
# ─────────────────────────────────────────────────────────────────────────────

def build_pyg_graph(positions, box, rc=RC_GRAPH):
    if not TORCH_AVAILABLE:
        return None
    pos  = np.array(positions, dtype=np.float64) % box
    N    = len(pos)
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(rc, output_type='ndarray')

    if len(pairs) == 0:
        return None

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    dr   = pos[i_idx] - pos[j_idx]
    dr  -= box * np.round(dr / box)
    dist = np.sqrt((dr**2).sum(axis=1))
    valid = dist < rc
    i_idx, j_idx, dist = i_idx[valid], j_idx[valid], dist[valid]

    # Build per-atom neighbour lists for node features
    nbrs  = [[] for _ in range(N)]
    for ii, jj, d in zip(i_idx, j_idx, dist):
        nbrs[ii].append(d)
        nbrs[jj].append(d)

    feats  = np.zeros((N, 8), dtype=np.float32)
    coords = np.array([len(nbrs[i]) for i in range(N)], dtype=np.float32)
    d_max  = coords.max() if coords.max() > 0 else 1.0

    for i in range(N):
        nd = np.array(nbrs[i])
        if len(nd) == 0:
            feats[i] = [rc, 0, rc, rc, 0, 0, rc, rc]
            continue
        skew = (float(np.mean((nd-nd.mean())**3)/(nd.std()**3+1e-8))
                if len(nd)>2 else 0.0)
        feats[i] = [nd.mean(),
                    nd.std() if len(nd) > 1 else 0.0,
                    nd.min(), nd.max(),
                    coords[i] / d_max,
                    np.clip(skew, -5, 5),
                    np.percentile(nd, 25),
                    np.percentile(nd, 75)]

    # Make edges bidirectional
    src = np.concatenate([i_idx, j_idx])
    dst = np.concatenate([j_idx, i_idx])
    eds = np.concatenate([dist,  dist])

    return Data(
        x          = torch.from_numpy(feats),
        edge_index = torch.from_numpy(np.stack([src,dst]).astype(np.int64)),
        edge_attr  = torch.from_numpy(eds.reshape(-1,1).astype(np.float32))
    )


def identify_key_frames(peak_result, n_frames, steps_per_frame=1000):
    peak_frame = min(max(peak_result['peak_step'] // steps_per_frame, 1),
                     n_frames - 1)
    mid_frame  = (peak_frame + (n_frames - 1)) // 2
    return [
        ("t=0   (just jumped to T=0.4)", 0),
        ("t=peak (Kovacs inflection)",   peak_frame),
        ("t=mid  (relaxing)",            mid_frame),
        ("t=end  (near equilibrium)",    n_frames - 1),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_pe(kovacs_pe, ref_pe_aligned, delta_pe, peak_result,
                  pe_ref_eq, out_path):
    steps = np.arange(len(kovacs_pe)) * 100
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Kovacs Memory Effect — Potential Energy Analysis\n'
                 'NVT Ensemble, LJ Glass (T_quench=0.3 → T_hold=0.4)',
                 fontsize=13, y=0.98)

    ax = axes[0]
    ax.plot(steps, kovacs_pe, color='#2c7bb6', lw=1.0, alpha=0.7,
            label='Kovacs path (quenched → aged → T=0.4)')
    ax.plot(steps, ref_pe_aligned, color='#d7191c', lw=1.0, alpha=0.6,
            ls='--', label='Reference (liquid → direct T=0.4)')
    ax.axhline(y=pe_ref_eq, color='#d7191c', lw=1.5, ls=':',
               label=f'PE_ref_eq = {pe_ref_eq:.4f}')
    ax.set_ylabel('Potential Energy (LJ units)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    smooth = peak_result['smooth']
    ax2.axhline(y=0, color='gray', lw=1.0, ls=':')
    ax2.plot(steps, delta_pe, color='#4dac26', lw=0.6, alpha=0.35,
             label='ΔPE (raw)')
    ax2.plot(steps[:len(smooth)], smooth, color='#4dac26', lw=2.2,
             label='ΔPE (smoothed)')

    peak_step = peak_result['peak_step']
    peak_val  = peak_result['delta_at_peak']
    ax2.axvline(x=peak_step, color='orange', lw=2.0, ls='--',
                label=f'Inflection @ step {peak_step:,}')
    ax2.scatter([peak_step], [peak_val], color='orange', s=100, zorder=5)
    ax2.annotate(f'  Memory\n  inflection\n  ΔPE={peak_val:+.4f}',
                 xy=(peak_step, peak_val),
                 xytext=(max(peak_step + steps[-1]*0.03, 5000),
                         peak_val + abs(delta_pe.min())*0.05),
                 fontsize=9, color='darkorange',
                 arrowprops=dict(arrowstyle='->', color='darkorange'))

    if peak_result['relaxation_tau']:
        tau = peak_result['relaxation_tau']
        ax2.annotate(f"τ_KWW ≈ {tau:,.0f} steps  ({tau*0.005:.2f} τ_LJ)",
                     xy=(0.55, 0.12), xycoords='axes fraction', fontsize=10,
                     color='#4dac26',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                               alpha=0.8))

    ax2.set_xlabel('Simulation Steps (hold at T=0.4)', fontsize=11)
    ax2.set_ylabel('ΔPE = PE_kovacs − PE_ref_eq', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_structural_trajectory(traj_k, traj_r, peak_result, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('GNN Feature Trajectories: Kovacs vs Reference\n'
                 '(Bond-length statistics — exact features used by GATv2)',
                 fontsize=13)

    specs    = [('r_max',   'r_max (max bond length)',  '#e66101'),
                ('std_r',   'σ_r (bond-length std)',     '#5e3c99'),
                ('skewness','Skewness (asymmetry)',       '#1a9641')]
    peak_s   = peak_result['peak_step']

    for ax, (key, ylabel, color) in zip(axes, specs):
        plotted = False
        if traj_k and len(traj_k.get(key, [])) > 1:
            ax.plot(traj_k['step'], traj_k[key],
                    color=color, lw=1.8, label='Kovacs path')
            plotted = True
        if traj_r and len(traj_r.get(key, [])) > 1:
            ax.plot(traj_r['step'], traj_r[key],
                    color=color, lw=1.5, ls='--', alpha=0.55,
                    label='Reference (equilibrium)')
            plotted = True
        ax.axvline(x=peak_s, color='orange', lw=1.5, ls='--', alpha=0.8,
                   label=f'Kovacs inflection (step {peak_s:,})')
        if not plotted:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=13, color='red')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Simulation Steps (hold at T=0.4)', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_gnn_bridge(delta_pe, d2min, peak_result, out_path,
                    steps_pe=100, steps_d2=1000):
    steps_pe_arr = np.arange(len(delta_pe)) * steps_pe
    steps_d2_arr = np.arange(len(d2min))    * steps_d2

    fig, ax1 = plt.subplots(figsize=(13, 5))
    fig.suptitle('Bridge: Macroscopic Memory ↔ Microscopic Topology\n'
                 'ΔPE (Kovacs signal) vs Mean Non-Affine Displacement D²_min',
                 fontsize=13)

    smooth = peak_result['smooth']
    steps_s = np.arange(len(smooth)) * steps_pe
    color_pe = '#2c7bb6'
    ax1.plot(steps_pe_arr, delta_pe, color=color_pe, lw=0.6, alpha=0.3)
    ax1.plot(steps_s, smooth, color=color_pe, lw=2.2,
             label='ΔPE = PE_kovacs − PE_ref (smoothed)')
    ax1.axhline(0, color='gray', lw=0.8, ls=':')

    peak_s = peak_result['peak_step']
    ax1.axvline(x=peak_s, color='orange', lw=2.0, ls='--', alpha=0.9,
                label=f'Kovacs inflection (step {peak_s:,})')
    ax1.axvspan(0,      peak_s,             alpha=0.05, color='orange')
    ax1.axvspan(peak_s, steps_pe_arr[-1],   alpha=0.05, color='green')

    ax1.set_xlabel('Simulation Steps (hold at T=0.4)', fontsize=11)
    ax1.set_ylabel('ΔPE (LJ units)', fontsize=11, color=color_pe)
    ax1.tick_params(axis='y', labelcolor=color_pe)

    ax2 = ax1.twinx()
    color_d2 = '#d7191c'
    ax2.plot(steps_d2_arr, d2min, 'o-', color=color_d2, lw=1.5, ms=3.5,
             label='Mean D²_min (non-affine strain)')
    ax2.set_ylabel('Mean Non-affine Strain (D²_min)', fontsize=11,
                   color=color_d2)
    ax2.tick_params(axis='y', labelcolor=color_d2)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_key_snapshot_features(frames, key_frames, out_path):
    if not frames:
        print("  [SKIP] snapshot features — no frames")
        return
    feature_names = ['mean_r', 'std_r', 'r_max', 'skewness']
    n_snaps = len(key_frames)
    vals    = np.zeros((n_snaps, len(feature_names)))
    labels  = []
    for si, (label, fidx) in enumerate(key_frames):
        fidx  = min(fidx, len(frames)-1)
        stats = extract_bond_stats_fast(frames[fidx]['positions'],
                                        frames[fidx]['box'])
        for fi, fn in enumerate(feature_names):
            vals[si, fi] = stats[fn]
        labels.append(f"{label}\n(frame {fidx})")

    # NumPy 2.0 safe: use max-min instead of ptp
    v_range   = vals.max(axis=0) - vals.min(axis=0)
    vals_norm = (vals - vals.min(axis=0)) / (v_range + 1e-8)

    fig, axes = plt.subplots(1, len(feature_names), figsize=(14, 4))
    fig.suptitle('GNN Feature Values at Key Kovacs Snapshots\n'
                 '(Normalised [0,1] per feature)', fontsize=12)
    colors = ['#2c7bb6', '#fdae61', '#a6d96a', '#1a9641']

    for fi, (ax, fname) in enumerate(zip(axes, feature_names)):
        bars = ax.bar(range(n_snaps), vals_norm[:, fi],
                      color=colors[:n_snaps], edgecolor='black', lw=0.8)
        ax.set_title(fname, fontsize=10)
        ax.set_xticks(range(n_snaps))
        ax.set_xticklabels([f'F{i}' for i in range(n_snaps)], fontsize=8)
        ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals[:, fi]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=l.split('\n')[0])
               for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def write_summary(peak_result, frames, key_frames, out_path):
    lines = [
        "="*68, "KOVACS MEMORY EFFECT — SUMMARY", "="*68, "",
        "SIMULATION",
        "  N=10000, binary LJ, ρ=1.2, NVT",
        "  Quench T=2.0→0.3 (50k steps)  |  Age T=0.3 (100k steps)",
        "  Kovacs hold T=0.4 (200k steps)",
        "", "PE RELAXATION",
        f"  Initial ΔPE : {peak_result['initial_delta']:+.6f} LJ",
        f"  Inflection  : step {peak_result['peak_step']:,}  "
        f"ΔPE={peak_result['delta_at_peak']:+.6f}",
        f"  Final ΔPE   : {peak_result['final_delta']:+.6f} LJ",
    ]
    if peak_result['relaxation_tau']:
        tau = peak_result['relaxation_tau']
        lines.append(f"  KWW τ       : {tau:,.1f} steps ({tau*0.005:.2f} τ_LJ)")

    lines += ["", "KEY SNAPSHOT STRUCTURAL METRICS",
              f"  {'Frm':<5}{'Label':<36}{'std_r':>8}{'r_max':>8}{'skew':>8}",
              "  "+"-"*65]
    for label, fidx in key_frames:
        fidx = min(fidx, len(frames)-1) if frames else 0
        if frames and fidx < len(frames):
            s = extract_bond_stats_fast(frames[fidx]['positions'],
                                        frames[fidx]['box'])
            lines.append(f"  {fidx:<5}{label[:36]:<36}"
                         f"{s['std_r']:>8.5f}{s['r_max']:>8.5f}"
                         f"{s['skewness']:>8.4f}")
    lines += ["", "="*68]
    with open(out_path, "w") as f:
        f.write("\n".join(lines)+"\n")
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*68)
    print("  KOVACS MEMORY EFFECT ANALYSIS  (v3 — fast cKDTree)")
    print("="*68)

    print("\n[1] Loading PE time series...")
    kovacs_pe = load_thermo("kovacs_thermo.txt")
    ref_pe    = load_thermo("reference_thermo.txt")
    delta_pe, ref_pe_al, pe_ref_eq, n = compute_delta_pe(kovacs_pe, ref_pe)
    print(f"  PE_ref_eq = {pe_ref_eq:.6f}  |  n={n} points")

    print("\n[2] Detecting Kovacs inflection...")
    peak_result = find_kovacs_peak(delta_pe, step_interval=100)

    print("\n[3] Loading OVITO metrics...")
    ovito = load_existing_metrics()

    print("\n[4] Parsing LAMMPS trajectories...")
    kovacs_frames = parse_lammps_dump("dump.kovacs")
    ref_frames    = parse_lammps_dump("dump.reference")

    print("\n[5] Extracting structural trajectories (cKDTree)...")
    traj_k = extract_structural_trajectory(kovacs_frames)
    traj_r = extract_structural_trajectory(ref_frames)

    print("\n[6] Identifying key frames...")
    key_frames = identify_key_frames(peak_result,
                                     n_frames=max(len(kovacs_frames),1))
    for label, fidx in key_frames:
        print(f"    Frame {fidx:>4}: {label}")

    if TORCH_AVAILABLE and len(kovacs_frames) > 1:
        print("\n[7] Building PyG graphs at key frames...")
        for label, fidx in key_frames:
            fidx = min(fidx, len(kovacs_frames)-1)
            g    = build_pyg_graph(kovacs_frames[fidx]['positions'],
                                   kovacs_frames[fidx]['box'])
            if g:
                print(f"    Frame {fidx}: {g.num_nodes} nodes, "
                      f"{g.num_edges} edges")

    print("\n[8] Generating figures...")
    plot_delta_pe(kovacs_pe[:n], ref_pe_al, delta_pe, peak_result,
                  pe_ref_eq, "kovacs_delta_PE.png")
    plot_structural_trajectory(traj_k, traj_r, peak_result,
                               "kovacs_structural.png")
    if "d2min" in ovito:
        plot_gnn_bridge(delta_pe, ovito["d2min"], peak_result,
                        "kovacs_gnn_bridge.png")
    plot_key_snapshot_features(kovacs_frames, key_frames,
                               "kovacs_snapshot_features.png")
    write_summary(peak_result, kovacs_frames, key_frames, "kovacs_summary.txt")

    print("\n"+"="*68+"  DONE  "+"="*68)
    for f in ["kovacs_delta_PE.png","kovacs_structural.png",
              "kovacs_gnn_bridge.png","kovacs_snapshot_features.png",
              "kovacs_summary.txt"]:
        print(f"  {'✓' if os.path.exists(f) else '✗'}  {f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
==========================================================================
BATTERY GLASS FATIGUE — DUAL T4 GPU DATA GENERATION
N=1024 | 300 glasses | Split 150 per GPU via JAX device placement
==========================================================================

STRATEGY:
  JAX detected [CudaDevice(id=0), CudaDevice(id=1)] on Kaggle.
  We spawn two Python threads. Each thread calls jax.device_put()
  to pin all its arrays to its assigned device, so both GPUs run
  independent glass trajectories in parallel.

  JAX dispatch is thread-safe across *different* devices — no locks needed.
  Each GPU sees its own XLA compilation cache (compiled once per device).

EXPECTED SPEEDUP:
  Single GPU: 300 glasses × 0.5 min/glass ≈ 2.5 hrs
  Dual GPU  : 150 glasses per GPU           ≈ 1.25 hrs  (≈2× wall time)

TIMING ESTIMATE (from your benchmark: 115 ms/cycle on T4):
  150 glasses × 400 cycles × 0.115 s = ~115 min per GPU (parallel)
  Total wall time ≈ 1.9–2.2 hrs  ✓ very comfortable

OUTPUT:
  /kaggle/working/battery_LJN1024_gpu0_snapshots.npy   (150 glasses)
  /kaggle/working/battery_LJN1024_gpu1_snapshots.npy   (150 glasses)
  /kaggle/working/battery_LJN1024_ALL300_snapshots.npy (merged, 1800 snaps)
  /kaggle/working/ckpt_gpu{0,1}/                       (per-glass checkpoints)
==========================================================================
"""

# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — Environment (MUST run before JAX import)
# ══════════════════════════════════════════════════════════════════════════
import os, sys, time, gc, warnings, threading
warnings.filterwarnings('ignore')

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Split GPU memory evenly: each device gets 42% (leaves headroom)
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.42')

os.system("pip install -q torch_geometric 2>/dev/null")


# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — Imports & Device Check
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import jax
import jax.numpy as jnp
from jax import jit, random
import jax.lax as lax

import torch

print(f"{'='*60}")
print(f"  JAX  version : {jax.__version__}")
jax_devices = jax.devices()
print(f"  JAX  devices : {jax_devices}")
print(f"  Torch version: {torch.__version__}")
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Torch device : {TORCH_DEVICE}")

if len(jax_devices) < 2:
    print("\n  ⚠  WARNING: Only 1 JAX device found!")
    print("     This script will still work but won't parallelise across GPUs.")
    print("     In Kaggle: Settings → Accelerator → GPU T4 x2")
    N_GPU = 1
else:
    N_GPU = 2
    print(f"\n  ✓  {N_GPU} GPUs detected — parallel generation enabled.")

for i, dev in enumerate(jax_devices[:N_GPU]):
    if torch.cuda.is_available() and i < torch.cuda.device_count():
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  ({props.total_memory/1e9:.1f} GB)")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — CONFIG
# ══════════════════════════════════════════════════════════════════════════

N_ATOMS       = 1024
N_GLASSES     = 300          # total; split evenly across GPUs
RHO           = 1.2
BOX_L         = float((N_ATOMS / RHO) ** (1/3))   # ≈ 9.485 σ
RC_LJ         = 2.5
DT            = 5e-4

T_HIGH        = 2.0
T_LOW         = 0.10
T_BATTERY     = 0.42
STRAIN_AMP    = 0.08
STEPS_PHASE   = 500
N_CYCLES      = 400
SAVE_AT       = [0, 50, 100, 200, 300, 400]

RC_GRAPH      = 1.5
PRISTINE_CYC  = {0}
FATIGUED_CYC  = {300, 400}

SCAN_CHUNK    = STEPS_PHASE        # 1 Python call per half-cycle
MAX_RETRIES   = 5
FORCE_CLIP    = 50.0

OUT_DIR       = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

# Per-GPU checkpoint dirs and output paths
CKPT_DIRS   = [os.path.join(OUT_DIR, f"ckpt_gpu{i}") for i in range(N_GPU)]
GPU_PATHS   = [os.path.join(OUT_DIR, f"battery_LJN{N_ATOMS}_gpu{i}_snapshots.npy")
               for i in range(N_GPU)]
MERGED_PATH = os.path.join(OUT_DIR, f"battery_LJN{N_ATOMS}_ALL{N_GLASSES}_snapshots.npy")

for d in CKPT_DIRS:
    os.makedirs(d, exist_ok=True)

# Split glass IDs across GPUs
glasses_per_gpu = N_GLASSES // N_GPU
GPU_GLASS_RANGES = []
for i in range(N_GPU):
    start = i * glasses_per_gpu
    end   = (i + 1) * glasses_per_gpu if i < N_GPU - 1 else N_GLASSES
    GPU_GLASS_RANGES.append(list(range(start, end)))

print(f"  N_ATOMS      : {N_ATOMS}")
print(f"  BOX_L        : {BOX_L:.4f} σ")
print(f"  N_GLASSES    : {N_GLASSES}  →  {[len(r) for r in GPU_GLASS_RANGES]} per GPU")
print(f"  SAVE_AT      : {SAVE_AT}  →  {N_GLASSES * len(SAVE_AT)} total snapshots")
print(f"  Clf graphs   : {N_GLASSES * (len(PRISTINE_CYC) + len(FATIGUED_CYC))}")
print(f"  Reg graphs   : {N_GLASSES * len(SAVE_AT)}")

assert RC_LJ < BOX_L / 2, f"MIC violated: RC_LJ={RC_LJ} >= BOX_L/2={BOX_L/2:.3f}"


# ══════════════════════════════════════════════════════════════════════════
# CELL 4 — JAX Physics (device-agnostic; arrays pinned at call time)
# ══════════════════════════════════════════════════════════════════════════

@jit
def lj_forces_pbc(pos, box):
    """LJ forces with minimum-image PBC. Works on any JAX device."""
    dr    = pos[:, None, :] - pos[None, :, :]
    dr    = dr - box * jnp.round(dr / box)
    r2    = jnp.sum(dr ** 2, axis=-1)
    mask  = (r2 > 1e-6) & (r2 < RC_LJ * RC_LJ)
    r2s   = jnp.where(mask, r2, 1.0)
    inv2  = 1.0 / r2s
    inv6  = inv2 ** 3
    inv12 = inv6 ** 2
    coeff = jnp.where(mask, 24.0 * inv2 * (2.0 * inv12 - inv6), 0.0)
    return jnp.sum(coeff[:, :, None] * dr, axis=1)


@jit
def md_chunk_safe(pos, key, box, temperature):
    """
    SCAN_CHUNK Brownian dynamics steps.
    Carries ok flag: once NaN appears it propagates to end of chunk.
    Device is determined by where pos lives (set by jax.device_put upstream).
    """
    dt        = jnp.float32(DT)
    sqrt_2Tdt = jnp.sqrt(2.0 * temperature * dt)

    def step(carry, _):
        p, k, ok = carry
        f     = lj_forces_pbc(p, box)
        f     = jnp.clip(f, -FORCE_CLIP, FORCE_CLIP)
        k, sk = random.split(k)
        noise = random.normal(sk, p.shape, dtype=jnp.float32)
        p_new = p + f * dt + sqrt_2Tdt * noise
        p_new = p_new % box
        ok_new = ok & jnp.all(jnp.isfinite(p_new))
        return (p_new, k, ok_new), None

    (pos_out, key_out, ok_out), _ = lax.scan(
        step, (pos, key, jnp.bool_(True)), None, length=SCAN_CHUNK
    )
    return pos_out, key_out, ok_out


@jit
def md_chunk_tight(pos, key, box, temperature):
    """Recovery kernel: tighter force cap = 20."""
    dt        = jnp.float32(DT)
    sqrt_2Tdt = jnp.sqrt(2.0 * temperature * dt)

    def step(carry, _):
        p, k, ok = carry
        f     = lj_forces_pbc(p, box)
        f     = jnp.clip(f, -20.0, 20.0)
        k, sk = random.split(k)
        noise = random.normal(sk, p.shape, dtype=jnp.float32)
        p_new = p + f * dt + sqrt_2Tdt * noise
        p_new = p_new % box
        ok_new = ok & jnp.all(jnp.isfinite(p_new))
        return (p_new, k, ok_new), None

    (pos_out, key_out, ok_out), _ = lax.scan(
        step, (pos, key, jnp.bool_(True)), None, length=SCAN_CHUNK
    )
    return pos_out, key_out, ok_out


def run_md(pos, key, box, temperature, total_steps, tight=False):
    """Run total_steps BD steps. Returns (pos, key, is_valid)."""
    box_f   = jnp.float32(box)
    T_f     = jnp.float32(temperature)
    kernel  = md_chunk_tight if tight else md_chunk_safe
    n_calls = max(1, total_steps // SCAN_CHUNK)

    for _ in range(n_calls):
        pos, key, ok = kernel(pos, key, box_f, T_f)
        jax.block_until_ready(ok)
        if not bool(ok):
            return pos, key, False
    return pos, key, True


# ══════════════════════════════════════════════════════════════════════════
# CELL 5 — Compile on BOTH devices (do this before threading)
# ══════════════════════════════════════════════════════════════════════════

def compile_on_device(dev_id):
    """Trigger XLA compilation on a specific device."""
    dev   = jax.devices()[dev_id]
    dummy = jax.device_put(jnp.zeros((N_ATOMS, 3), dtype=jnp.float32), dev)
    key   = jax.device_put(random.PRNGKey(0), dev)
    out, _, _ = md_chunk_safe(dummy, key,
                               jax.device_put(jnp.float32(BOX_L), dev),
                               jax.device_put(jnp.float32(T_LOW),  dev))
    jax.block_until_ready(out)
    return dev


print("Compiling JAX kernels on all devices...")
t0 = time.time()
compile_threads = []
for dev_id in range(N_GPU):
    t = threading.Thread(target=compile_on_device, args=(dev_id,))
    t.start()
    compile_threads.append(t)
for t in compile_threads:
    t.join()
print(f"  All devices compiled in {time.time()-t0:.1f} s\n")


# ══════════════════════════════════════════════════════════════════════════
# CELL 6 — Glass Generation (device-aware)
# ══════════════════════════════════════════════════════════════════════════

def init_on_lattice(key, dev):
    n_side  = int(np.ceil(N_ATOMS ** (1/3)))
    spacing = BOX_L / n_side
    pts = np.array(
        [(i, j, k) for i in range(n_side)
                   for j in range(n_side)
                   for k in range(n_side)],
        dtype=np.float32
    )[:N_ATOMS] * spacing
    key_dev = jax.device_put(key, dev)
    key_dev, sk = random.split(key_dev)
    pts = jax.device_put(jnp.array(pts), dev)
    pts = pts + random.normal(sk, pts.shape, dtype=jnp.float32) * 0.05
    pts = pts % jnp.float32(BOX_L)
    return pts, key_dev


def fast_cool_glass(key, dev, n_cool_chunks=40):
    """Generate pristine glass on the specified JAX device."""
    box = jax.device_put(jnp.float32(BOX_L), dev)
    pos, key = init_on_lattice(key, dev)

    for _ in range(10):
        pos, key, _ = md_chunk_safe(pos, key, box, jax.device_put(jnp.float32(T_HIGH), dev))

    temps = np.linspace(T_HIGH, T_LOW, n_cool_chunks, dtype=np.float32)
    for T_val in temps:
        pos, key, _ = md_chunk_safe(pos, key, box, jax.device_put(jnp.float32(T_val), dev))

    for _ in range(20):
        pos, key, _ = md_chunk_safe(pos, key, box, jax.device_put(jnp.float32(T_LOW), dev))

    return np.array(pos), key


def cycle_once_safe(pos, key, dev, glass_id=-1, cyc=-1):
    """
    One charge/discharge cycle on the specified device.
    Returns (positions_numpy, key, is_valid).
    """
    scale    = float(1.0 + STRAIN_AMP)
    box_exp  = BOX_L * scale
    box_orig = BOX_L
    T        = T_BATTERY

    # Pin to device
    pos_j   = jax.device_put(jnp.array(pos), dev)

    # ── Charge ────────────────────────────────────────────────────────
    pos_before = np.array(pos_j).copy()
    pos_j   = pos_j * jnp.float32(scale)
    pos_j, key, ok = run_md(pos_j, key, box_exp, T, STEPS_PHASE)

    if not ok:
        pos_j = jax.device_put(jnp.array(pos_before), dev) * jnp.float32(scale)
        pos_j, key, ok = run_md(pos_j, key, box_exp, T, STEPS_PHASE, tight=True)
        if not ok:
            return np.array(pos_j), key, False

    # ── Discharge ─────────────────────────────────────────────────────
    pos_before_d = np.array(pos_j).copy()
    pos_j = pos_j / jnp.float32(scale)
    pos_j, key, ok = run_md(pos_j, key, box_orig, T, STEPS_PHASE)

    if not ok:
        pos_j = jax.device_put(jnp.array(pos_before_d), dev) / jnp.float32(scale)
        pos_j, key, ok = run_md(pos_j, key, box_orig, T, STEPS_PHASE, tight=True)
        if not ok:
            return np.array(pos_j), key, False

    return np.array(pos_j), key, True


# ══════════════════════════════════════════════════════════════════════════
# CELL 7 — Checkpoint Utilities
# ══════════════════════════════════════════════════════════════════════════

def ckpt_path(glass_id, gpu_id):
    return os.path.join(CKPT_DIRS[gpu_id], f"glass_{glass_id:04d}.npy")


def save_ckpt(glass_id, gpu_id, glass_snaps):
    np.save(ckpt_path(glass_id, gpu_id), glass_snaps)


def load_existing_snapshots(gpu_id):
    """Load completed glasses for this GPU from its checkpoint dir."""
    snapshots, completed = [], set()
    ckpt_files = sorted(
        f for f in os.listdir(CKPT_DIRS[gpu_id])
        if f.startswith("glass_") and f.endswith(".npy")
    )
    for fname in ckpt_files:
        data = list(np.load(os.path.join(CKPT_DIRS[gpu_id], fname),
                            allow_pickle=True))
        snapshots.extend(data)
        completed.add(data[0]['glass_id'])

    if completed:
        print(f"  [GPU {gpu_id}] Resumed: {len(completed)} glasses "
              f"({len(snapshots)} snapshots) already done.")
    return snapshots, completed


# ══════════════════════════════════════════════════════════════════════════
# CELL 8 — Per-GPU Worker Function (runs in its own thread)
# ══════════════════════════════════════════════════════════════════════════

def worker(gpu_id, result_dict, nan_log_dict, timing_dict):
    """
    Generates all glasses assigned to gpu_id.
    Stores snapshots in result_dict[gpu_id] when done.
    Thread-safe: only writes to its own gpu_id key and checkpoint dir.
    """
    dev          = jax.devices()[gpu_id]
    glass_ids    = GPU_GLASS_RANGES[gpu_id]
    save_set     = set(SAVE_AT)
    nan_events   = []

    print(f"\n  [GPU {gpu_id}] Starting: {len(glass_ids)} glasses "
          f"(ids {glass_ids[0]}–{glass_ids[-1]})  on {dev}")

    # Resume
    snapshots, completed = load_existing_snapshots(gpu_id)
    remaining = [g for g in glass_ids if g not in completed]

    if not remaining:
        print(f"  [GPU {gpu_id}] All glasses already done.")
        result_dict[gpu_id] = snapshots
        timing_dict[gpu_id] = 0.0
        return

    # Unique master key per GPU (offset by gpu_id * 10000 for independence)
    master_key = random.PRNGKey(2024 + N_ATOMS + gpu_id * 10000)
    master_key = jax.device_put(master_key, dev)

    new_snapshots = []
    t_start       = time.time()

    for gi, gid in enumerate(remaining):
        master_key, subkey = random.split(master_key)
        success = False

        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                msg = f"GPU{gpu_id} glass {gid} retry {attempt+1}"
                nan_events.append(msg)
                subkey, _ = random.split(subkey)

            # Generate pristine
            pos, key = fast_cool_glass(subkey, dev)

            if not np.isfinite(pos).all():
                nan_events.append(f"GPU{gpu_id} NaN in pristine glass {gid}")
                continue

            glass_snaps = []
            if 0 in save_set:
                glass_snaps.append({'glass_id': gid, 'cycle': 0,
                                    'positions': pos.copy()})

            nan_hit = False
            for cyc in range(1, N_CYCLES + 1):
                pos, key, ok = cycle_once_safe(pos, key, dev,
                                               glass_id=gid, cyc=cyc)
                if not ok or not np.isfinite(pos).all():
                    nan_events.append(
                        f"GPU{gpu_id} glass {gid} cy {cyc} NaN (attempt {attempt+1})")
                    nan_hit = True
                    break

                if cyc in save_set:
                    glass_snaps.append({'glass_id': gid, 'cycle': cyc,
                                        'positions': pos.copy()})

            if nan_hit:
                continue

            save_ckpt(gid, gpu_id, glass_snaps)
            new_snapshots.extend(glass_snaps)
            success = True
            break

        if not success:
            nan_events.append(f"GPU{gpu_id} glass {gid} FAILED all {MAX_RETRIES} retries")

        # Progress (thread-safe print — threads interleave but that's OK)
        elapsed     = time.time() - t_start
        done_so_far = gi + 1
        eta         = elapsed / done_so_far * (len(remaining) - done_so_far)
        print(f"  [GPU {gpu_id}] [{done_so_far:3d}/{len(remaining)}] "
              f"glass {gid:3d} | {elapsed/60:.1f} min elapsed | "
              f"ETA {eta/60:.1f} min | NaN events: {len(nan_events)}")

        gc.collect()

    # Save partial results
    all_snaps = snapshots + new_snapshots
    np.save(GPU_PATHS[gpu_id], all_snaps)
    elapsed_total = time.time() - t_start

    print(f"\n  [GPU {gpu_id}] DONE — {len(all_snaps)} snapshots | "
          f"{elapsed_total/60:.1f} min | {len(nan_events)} NaN events")

    result_dict[gpu_id]  = all_snaps
    nan_log_dict[gpu_id] = nan_events
    timing_dict[gpu_id]  = elapsed_total


# ══════════════════════════════════════════════════════════════════════════
# CELL 9 — Speed Benchmark (both GPUs simultaneously)
# ══════════════════════════════════════════════════════════════════════════

def benchmark_both_gpus(n_cycles=5):
    """Run n_cycles on a dummy glass on each GPU simultaneously."""
    print("Benchmarking both GPUs in parallel...")
    times = {}

    def bench_worker(dev_id):
        dev = jax.devices()[dev_id]
        key = jax.device_put(random.PRNGKey(99 + dev_id), dev)
        pos = jax.device_put(jnp.zeros((N_ATOMS, 3), dtype=jnp.float32), dev)
        pos, key, _ = run_md(pos, key, BOX_L, T_LOW, SCAN_CHUNK * 2)

        t0 = time.time()
        for _ in range(n_cycles):
            scale = float(1.0 + STRAIN_AMP)
            pos_e = pos * jnp.float32(scale)
            pos_e, key, _ = run_md(pos_e, key, BOX_L * scale,  T_BATTERY, STEPS_PHASE)
            pos_e = pos_e / jnp.float32(scale)
            pos_e, key, _ = run_md(pos_e, key, BOX_L,          T_BATTERY, STEPS_PHASE)
        jax.block_until_ready(pos_e)
        times[dev_id] = (time.time() - t0) / n_cycles

    threads = [threading.Thread(target=bench_worker, args=(i,)) for i in range(N_GPU)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"\n  {'GPU':<6} {'ms/cycle':>10}  {'proj. hrs (150 glasses)':>25}")
    print(f"  {'─'*45}")
    for dev_id, spc in times.items():
        proj_hrs = (GPU_GLASS_RANGES[dev_id].__len__() * N_CYCLES * spc) / 3600
        print(f"  {dev_id:<6} {spc*1000:>10.1f}  {proj_hrs:>25.2f}")

    wall_hrs = max(
        len(GPU_GLASS_RANGES[i]) * N_CYCLES * times[i] / 3600
        for i in range(N_GPU)
    )
    print(f"\n  Projected wall time (parallel): {wall_hrs:.2f} hrs")
    if wall_hrs < 10:
        print(f"  ✓  GO — {(12-wall_hrs)*60:.0f} min buffer remaining")
    else:
        print(f"  ⚠  TIGHT — consider reducing N_GLASSES")
    return times

benchmark_both_gpus()


# ══════════════════════════════════════════════════════════════════════════
# CELL 10 — Merge + Validate
# ══════════════════════════════════════════════════════════════════════════

def merge_and_validate(result_dict):
    """Merge GPU results, re-index glass IDs if needed, run NaN audit."""
    all_snaps = []
    for gpu_id in range(N_GPU):
        snaps = result_dict.get(gpu_id, [])
        print(f"  GPU {gpu_id}: {len(snaps)} snapshots")
        all_snaps.extend(snaps)

    # Verify no duplicate glass IDs
    gids = [s['glass_id'] for s in all_snaps]
    unique_gids = set(gids)
    print(f"\n  Unique glass IDs : {len(unique_gids)}  (expected {N_GLASSES})")
    if len(unique_gids) != N_GLASSES:
        print(f"  ⚠  Missing glass IDs: "
              f"{set(range(N_GLASSES)) - unique_gids}")

    # Cycle coverage
    cycles_found = sorted(set(s['cycle'] for s in all_snaps))
    print(f"  Cycles found     : {cycles_found}  (expected {SAVE_AT})")

    # NaN check
    n_nan = sum(1 for s in all_snaps if not np.isfinite(s['positions']).all())
    print(f"  NaN snapshots    : {n_nan}  "
          f"({'✓ clean' if n_nan == 0 else '✗ REMOVE BEFORE TRAINING'})")

    if n_nan > 0:
        all_snaps = [s for s in all_snaps if np.isfinite(s['positions']).all()]
        print(f"  Filtered to      : {len(all_snaps)} clean snapshots")

    np.save(MERGED_PATH, all_snaps)
    print(f"\n  ✓  Merged dataset saved → {MERGED_PATH}")
    print(f"     ({len(all_snaps)} snapshots, "
          f"{os.path.getsize(MERGED_PATH)/1e6:.1f} MB)")
    return all_snaps


def print_bond_stats(snapshots, n_per_cycle=20):
    """Quick structural drift table."""
    print(f"\n  {'Cycle':>6}  {'Mean r':>10}  {'Std r':>10}  {'N bonds':>10}")
    print(f"  {'─'*46}")
    for cyc in sorted(set(s['cycle'] for s in snapshots)):
        bonds = []
        for s in [x for x in snapshots if x['cycle'] == cyc][:n_per_cycle]:
            pos = s['positions'].astype(np.float32)
            dr  = pos[:, None, :] - pos[None, :, :]
            dr  = dr - BOX_L * np.round(dr / BOX_L)
            d   = np.sqrt(np.sum(dr**2, axis=-1))
            m   = (d > 1e-6) & (d < RC_GRAPH)
            bonds.extend(d[m].tolist())
        a = np.array(bonds)
        print(f"  {cyc:>6}  {a.mean():>10.5f}  {a.std():>10.5f}  {len(a):>10,}")
    print(f"  {'─'*46}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 11 — MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_wall = time.time()

    print(f"\n{'═'*60}")
    print(f"  DUAL-GPU BATTERY GLASS FATIGUE GENERATION")
    print(f"  N_ATOMS={N_ATOMS} | N_GLASSES={N_GLASSES} | "
          f"{N_GPU} GPU(s) | {N_GLASSES//N_GPU} glasses/GPU")
    print(f"{'═'*60}\n")

    # ── Check if merged file already exists ──────────────────────────
    if os.path.exists(MERGED_PATH):
        print(f"Merged file found: {MERGED_PATH}")
        print("Loading... (delete to regenerate)")
        snapshots = list(np.load(MERGED_PATH, allow_pickle=True))
        print(f"  Loaded {len(snapshots)} snapshots.")
        print_bond_stats(snapshots)
        return snapshots

    # ── Launch one thread per GPU ────────────────────────────────────
    result_dict  = {}
    nan_log_dict = {}
    timing_dict  = {}

    threads = []
    for gpu_id in range(N_GPU):
        t = threading.Thread(
            target=worker,
            args=(gpu_id, result_dict, nan_log_dict, timing_dict),
            name=f"GPU-{gpu_id}-worker"
        )
        threads.append(t)

    print(f"Launching {N_GPU} worker thread(s)...\n")
    t_launch = time.time()
    for t in threads:
        t.start()

    # Progress ticker (main thread: print a heartbeat every 5 min)
    while any(t.is_alive() for t in threads):
        time.sleep(300)   # 5 min
        alive = sum(t.is_alive() for t in threads)
        print(f"  [main] {(time.time()-t_launch)/60:.0f} min elapsed | "
              f"{alive}/{N_GPU} workers still running ...")

    for t in threads:
        t.join()
    wall_min = (time.time() - t_launch) / 60
    print(f"\nAll workers done in {wall_min:.1f} min.")

    # ── NaN logs ──────────────────────────────────────────────────────
    all_nan = []
    for gpu_id, evts in nan_log_dict.items():
        all_nan.extend(evts)
    if all_nan:
        log_path = os.path.join(OUT_DIR, "nan_log.txt")
        with open(log_path, 'w') as f:
            f.write('\n'.join(all_nan))
        print(f"  ⚠  {len(all_nan)} NaN events logged → {log_path}")
    else:
        print(f"  ✓  Zero NaN events across all GPUs.")

    # ── Merge and validate ────────────────────────────────────────────
    print("\nMerging GPU datasets...")
    snapshots = merge_and_validate(result_dict)

    # ── Structural drift ─────────────────────────────────────────────
    print("\nStructural drift summary:")
    print_bond_stats(snapshots)

    # ── Final summary ─────────────────────────────────────────────────
    total_min = (time.time() - t_wall) / 60
    print(f"\n{'═'*60}")
    print(f"  Total wall time : {total_min:.1f} min")
    print(f"  Snapshots       : {len(snapshots)}")
    print(f"  Clf graphs      : "
          f"{sum(1 for s in snapshots if s['cycle'] in PRISTINE_CYC|FATIGUED_CYC)}")
    print(f"  Merged file     : {MERGED_PATH}")
    print(f"{'═'*60}")
    print(f"\nNext step: load {MERGED_PATH} in the GNN training script.")
    print(f"Update FINAL_PATH, N_ATOMS, BOX_L to match this configuration.")

    return snapshots


if __name__ == "__main__":
    snapshots = main()
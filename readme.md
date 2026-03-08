# Geometric Encoding of Cyclic Mechanical Fatigue in Model Glasses
### Graph Neural Networks as Structural Probes for Battery-Relevant Degradation

**Author:** Saurab Singh — Independent Researcher  
**Preprint:** [arXiv link — add when posted]  
**Paper:** `paper_final.tex` — submission-ready RevTeX4-2

---

## Overview

This repository contains all simulation code, GNN training experiments, and generated data for the study of cyclic mechanical fatigue in a Lennard-Jones model glass, motivated by solid-state battery electrolyte degradation.

The central finding is that repeated charge-discharge cycling encodes a **learnable geometric signature** in the local bond-length distribution of the glass network — detectable by a GATv2 graph neural network with >98% accuracy — and that this signal resides in **relational geometry**, not in bulk volume changes.

---

## Key Results

| Experiment | Features | Clf Acc (%) | Reg R² |
|---|---|---|---|
| Baseline | 8D, no norm | 91.67 ± 4.22 | 0.062 ± 0.125 |
| **Test A** | **8D + instance norm** | **98.33 ± 1.49** | **0.311 ± 0.039** |
| Test B | 5D only (no shape) | 86.33 ± 4.64 | 0.006 ± 0.012 |
| Test C | 8D, extended training | — | 0.235 ± 0.118 |
| **N=1024 Test A** | **8D + instance norm** | **100.00 ± 0.00** | **0.428 ± 0.041** |

*5-fold stratified cross-validation throughout. Classification: pristine (cycle 0) vs. fatigued (cycles 300–400).*

---

## Repository Structure

```
solid_state_battery/
│
├── data_gen_1k.py          # Glass generation + cyclic strain protocol for N=1024
├── exp_1.py                # Baseline + Test A + Test B classification & regression (N=256)
├── exp_2.py                # Test C: extended regression training (N=256)
├── exp_3.py                # Permutation importance analysis (N=256)
├── exp_4.py                # Full ablation at N=1024 (finite-size scaling)
│
├── test/                   # Raw outputs from initial N=256 experiments
│   ├── battery_snapshots.npy
│   ├── bond_stats_vs_cycle.png
│   ├── clf_learning_curves.png
│   └── force_chains_battery.png
│
├── test_v2/                # Final N=256 experiment figures (used in paper)
│   ├── battery_snapshots.npy
│   ├── bond_stats_vs_cycle.png
│   ├── force_chains_battery.png
│   ├── baseline_clf_curves.png
│   ├── testA_clf_curves.png
│   └── testB_clf_curves.png
│
├── test_3/                 # Permutation importance figures (used in paper)
│   ├── permutation_importance_bar_fixed.png
│   ├── permutation_importance_heatmap_fixed.png
│   └── permutation_importance_table.txt
│
├── results_N1024/          # N=1024 experiment outputs (used in paper)
│   ├── battery_LJN1024_ALL300_merged.npy
│   ├── bond_stats_N1024.png
│   ├── force_chains_N1024.png
│   ├── baseline_clf_curves_N1024.png
│   ├── testA_clf_curves_N1024.png
│   └── testB_clf_curves_N1024.png
│
└── LICENSE
```

---

## Methods Summary

### Model System
Single-component Lennard-Jones glass, `N = 256` and `N = 1024` particles, density `ρ = 1.2 σ⁻³`, periodic cubic box. Glass transition temperature `Tg ≈ 0.45` (reduced units).

### Cyclic Strain Protocol
Each cycle applies 8% affine volumetric expansion (charge) followed by compression back to original volume (discharge), with 500 Brownian dynamics steps at `T = 0.42 ≈ 0.93 Tg` after each half-cycle. Snapshots saved at cycles `{0, 50, 100, 200, 300, 400}`.

### Node Features
Per-atom bond-length statistics computed within cutoff `rc = 1.5 σ`:

| Feature set | Dimensions | Features |
|---|---|---|
| 5D base | 5 | mean, std, min, max, coordination |
| 8D extended | 8 | 5D + skewness, Q25, Q75 |

### GNN Architecture
GATv2 with 2 message-passing layers (4 attention heads each), global mean pooling, shared encoder/head for both classification and regression. Identical architecture used at both N=256 and N=1024.

### Training
Adam optimizer, `lr = 3×10⁻⁴`, cosine annealing, early stopping on validation loss. 5-fold stratified cross-validation, glass-level splits (no data leakage between train/val folds).

---

## Reproducing the Experiments

### Requirements

```bash
pip install jax jaxlib torch torch-geometric numpy matplotlib scipy
```

A CUDA-capable GPU is recommended. All experiments were run on a Tesla T4 (Google Colab free tier).

### Step 1 — Generate glass data (N=256)

Glass generation is embedded in `exp_1.py`. Pre-generated snapshots are provided in `test_v2/battery_snapshots.npy`.

### Step 2 — Run N=256 experiments

```bash
# Baseline + Test A + Test B (classification and regression)
python exp_1.py

# Test C: extended regression
python exp_2.py
```

### Step 3 — Permutation importance

```bash
python exp_3.py
```

### Step 4 — Generate N=1024 data

```bash
python data_gen_1k.py
```

Pre-generated data is available in `results_N1024/battery_LJN1024_ALL300_merged.npy`.

### Step 5 — Run N=1024 experiments

```bash
python exp_4.py
```

---

## Figures in the Paper

| Figure | Source file |
|---|---|
| Fig. 1 | Algorithm listing (LaTeX) |
| Fig. 2 — Structural drift (N=256) | `test_v2/bond_stats_vs_cycle.png` |
| Fig. 3 — Strain topology (N=256) | `test_v2/force_chains_battery.png` |
| Fig. 4 — Test A learning curves | `test_v2/testA_clf_curves.png` |
| Fig. 5 — Baseline learning curves | `test_v2/baseline_clf_curves.png` |
| Fig. 6 — Test B learning curves | `test_v2/testB_clf_curves.png` |
| Fig. 7 — Permutation importance bar | `test_3/permutation_importance_bar_fixed.png` |
| Fig. 8 — Permutation importance heatmap | `test_3/permutation_importance_heatmap_fixed.png` |
| Fig. 9 — Structural drift (N=1024) | `results_N1024/bond_stats_N1024.png` |
| Fig. 10 — Strain topology (N=1024) | `results_N1024/force_chains_N1024.png` |
| Baseline clf curves (N=1024) | `results_N1024/baseline_clf_curves_N1024.png` |
| Test A clf curves (N=1024) | `results_N1024/testA_clf_curves_N1024.png` |
| Test B clf curves (N=1024) | `results_N1024/testB_clf_curves_N1024.png` |

---

## Companion Study

This work extends the geometric-encoding framework from:

> S. Singh, "Geometric Encoding of Thermal History in Glasses: Strain Topology as a Learnable Structural Signature," arXiv preprint (2025).  
> Code: https://github.com/Saurabsingh778/Thermal_history_in_glass

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{singh2026fatigue,
  title   = {Geometric Encoding of Cyclic Mechanical Fatigue in Model Glasses:
             Graph Neural Networks as Structural Probes for Battery-Relevant Degradation},
  author  = {Singh, Saurab},
  year    = {2026},
  note    = {arXiv preprint — link to be added}
}
```

---

## License

This project is licensed under the terms of the LICENSE file in this repository.
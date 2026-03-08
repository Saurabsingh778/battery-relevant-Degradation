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

## Results

### Main Ablation (N=256, pristine cycle 0 vs. fatigued cycles 300–400)

| Experiment | Features | Clf Acc (%) | Reg R² |
|---|---|---|---|
| Baseline | 8D, no norm | 91.67 ± 4.22 | 0.062 ± 0.125 |
| **Test A** | **8D + instance norm** | **98.33 ± 1.49** | **0.311 ± 0.039** |
| Test B | 5D only (no shape features) | 86.33 ± 4.64 | 0.006 ± 0.012 |
| Test C | 8D, extended training (300 epochs) | — | 0.235 ± 0.118 |

*5-fold stratified cross-validation. Classification: pristine (cycle 0) vs. fatigued (cycles 300–400). Regression: predict normalised cycle number from all 600 graphs.*

---

### 4D Minimal Sufficient Descriptor Set (exp_3.py)

Permutation importance identified {r_max, r̄, σ_r, γ̃} as the four dominant features. Training the identical GATv2 with only these four features:

| Config | Features | Clf Acc (%) | Reg R² |
|---|---|---|---|
| 8D Baseline | 8D, no norm | 91.67 ± 4.22 | 0.062 ± 0.125 |
| 8D Test A | 8D + norm | 98.33 ± 1.49 | 0.311 ± 0.039 |
| **4D Baseline** | **4D, no norm** | **91.67 ± 2.36** | **0.264 ± 0.036** |
| 4D Test A | 4D + norm | 94.67 ± 1.94 | 0.287 ± 0.045 |

Three key findings from the 4D ablation:

1. **Classification accuracy is exactly preserved** at 91.67% despite a 50% reduction in feature dimensionality — the four dropped features (r_min, d̃, Q25, Q75) contribute no measurable classification information when the four dominant features are already present.
2. **Fold-to-fold variance nearly halves** (±4.22% → ±2.36%), confirming the dropped features were actively introducing gradient noise and causing the unstable folds 4–5 seen in the 8D baseline.
3. **Regression R² improves dramatically without normalisation** (0.062 → 0.264): the 4D set in a single standard run matches what required either extended training (Test C: R²=0.235) or instance normalisation (Test A: R²=0.311) in the full 8D case. The dropped features were actively degrading regression convergence.

Context-dependence note: under normalisation, 8D+norm (R²=0.311) still outperforms 4D+norm (R²=0.287) because instance normalisation removes the amplitude noise that was masking the signal in Q25 and Q75, allowing the full 8D set to use all features productively. The 8D+norm configuration is therefore the best overall configuration.

---

### Permutation Importance Feature Ranking (exp_3.py, baseline model, N=256)

| Rank | Feature | ΔAUC (mean) | ±std |
|---|---|---|---|
| 1 | r_max | +0.601 | 0.207 |
| 2 | r̄ (mean bond length) | +0.547 | 0.273 |
| 3 | skewness γ̃ | +0.317 | 0.270 |
| 4 | σ_r (bond length std) | +0.316 | 0.270 |
| 5 | Q25 (lower quartile) | +0.200 | 0.169 |
| 6 | d̃ (normalised coordination) | +0.168 | 0.233 |
| 7 | r_min | +0.164 | 0.131 |
| 8 | Q75 (upper quartile) | +0.097 | 0.073 |

*Baseline AUC = 0.9620 ± 0.0549. Importance = AUC drop when feature is zeroed out across all validation graphs (zero-out, not shuffle, to avoid leaking dataset-level signals).*

r_max is the single most important feature across all five folds — it directly detects the most highly strained bond per atom, the primary signature of plastic damage sites. Q75 ranks last because its upper-tail information is largely redundant with r_max already present in the set.

---

### Finite-Size Scaling: N=1024 (exp_4.py)

| Config | Clf Acc N=256 (%) | Clf Acc N=1024 (%) | Reg R² N=256 | Reg R² N=1024 |
|---|---|---|---|---|
| Baseline | 91.67 ± 4.22 | 99.11 ± 1.03 | 0.062 ± 0.125 | 0.296 ± 0.149 † |
| **Test A** | **98.33 ± 1.49** | **100.00 ± 0.00** | **0.311 ± 0.039** | **0.428 ± 0.041** |
| Test B | 86.33 ± 4.64 | 99.44 ± 0.61 | 0.006 ± 0.012 | 0.000 ± 0.000 ‡ |
| Test C | — | — | 0.235 ± 0.118 | 0.379 ± 0.019 |

† Fold 3 collapsed to R²=0.0006 (early stop epoch 30); remaining four folds: R² = 0.370 ± 0.026.  
‡ All five folds predict the dataset mean (MSE=0.1236, MAE=0.3124; early stop within 27–36 epochs).

Notable N=1024 findings:
- Test B (5D only) reaches 99.44% classification but **complete regression failure** (R²=0.000) — shape features are strictly necessary for continuous cycle tracking at any system size
- Test A (8D + norm) hits **100.00% classification** and **R²=0.428 regression**, the best result in the entire study
- Extended training (Test C, R²=0.379) cannot substitute for instance normalisation (Test A, R²=0.428): the normalisation benefit on regression is structural, not a training-budget artefact
- Normalisation at N=1024 primarily delivers **6× faster convergence** (~15 vs ~100+ epochs) and **elimination of fold collapses**, rather than accuracy gains

---

## Repository Structure

```
solid_state_battery/
│
├── data_gen_1k.py          # Glass generation + cyclic strain protocol for N=1024
├── exp_1.py                # Baseline + Test A + Test B classification & regression (N=256)
├── exp_2.py                # Test C: extended regression training (N=256)
├── exp_3.py                # Permutation importance + 4D minimal descriptor ablation (N=256)
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
├── test_3/                 # Permutation importance + 4D ablation figures (used in paper)
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

| Feature set | Dimensions | Features |
|---|---|---|
| 5D base | 5 | mean r̄, std σ_r, r_min, r_max, coordination d̃ |
| 8D extended | 8 | 5D + skewness γ̃, Q25, Q75 |
| **4D minimal** | **4** | **r_max, r̄, σ_r, γ̃** (minimal sufficient set) |

### GNN Architecture
GATv2 with 2 message-passing layers (4 attention heads each), global mean pooling, shared encoder/head for classification and regression. Identical architecture used at N=256 and N=1024, with encoder input dimension inferred automatically from data.

### Training
Adam optimizer, `lr = 3×10⁻⁴`, cosine annealing, early stopping on **validation loss** (not accuracy — accuracy creates a false optimum at epoch 1 for random-weight models). 5-fold stratified cross-validation, glass-level splits to prevent data leakage.

---

## Reproducing the Experiments

### Requirements

```bash
pip install jax jaxlib torch torch-geometric numpy matplotlib scipy
```

A CUDA-capable GPU is recommended. All experiments were run on a Tesla T4 (Google Colab free tier). N=256 data generation takes ~10 minutes; N=1024 takes ~665 minutes.

### Step 1 — N=256 experiments (Baseline, Test A, Test B, Test C)

Pre-generated snapshots are available in `test_v2/battery_snapshots.npy`.

```bash
python exp_1.py   # Baseline + Test A + Test B (classification and regression)
python exp_2.py   # Test C: extended regression (300 epochs, patience 50)
```

### Step 2 — Permutation importance + 4D minimal descriptor ablation

```bash
python exp_3.py
```

Produces: permutation importance bar chart, per-fold heatmap, permutation table, and 4D vs 8D comparison.

### Step 3 — Generate N=1024 glass data

```bash
python data_gen_1k.py
```

Pre-generated data: `results_N1024/battery_LJN1024_ALL300_merged.npy` (300 glasses × 6 snapshots).

### Step 4 — N=1024 finite-size scaling experiments

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
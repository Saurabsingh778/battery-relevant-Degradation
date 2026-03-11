import matplotlib.pyplot as plt
import numpy as np

# 1. Load the generated datasets
# Macroscopic energy (2001 points sampled every 100 steps)
energy_data = np.loadtxt("kovacs_thermo.txt")

# Microscopic topology (201 points sampled every 1000 steps)
d2min_data = np.loadtxt("metric_d2min.txt")

# 2. Create independent time axes for both (Total of 200,000 steps)
# Time axis for energy data (0, 100, 200 ... 200,000)
time_energy = np.linspace(0, 200000, len(energy_data))

# Time axis for metric data (0, 1000, 2000 ... 200,000)
time_metrics = np.linspace(0, 200000, len(d2min_data))

# 3. Create the Publication-Ready Figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Axis: Macroscopic Potential Energy (The "Truth")
color = 'tab:blue'
ax1.set_xlabel('Simulation Steps (Hold at T=0.4)', fontsize=12)
ax1.set_ylabel('Potential Energy (LJ Units)', color=color, fontsize=12, fontweight='bold')
ax1.plot(time_energy, energy_data, color=color, linewidth=2.0, label='Macroscopic Energy')
ax1.tick_params(axis='y', labelcolor=color)

# Right Axis: Microscopic Strain Topology (Your Theory)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Mean Non-affine Strain ($D^2_{min}$)', color=color, fontsize=12, fontweight='bold')
ax2.plot(time_metrics, d2min_data, color=color, linestyle='--', linewidth=2.5, marker='o', markersize=4, alpha=0.7, label='Strain Topology Metric')
ax2.tick_params(axis='y', labelcolor=color)

# Formatting for "Physical Review Materials" style
plt.title('Bridge Between Macroscopic Memory and Microscopic Topology', fontsize=14, fontweight='bold', pad=20)
fig.tight_layout()
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# 4. Save and Show
plt.savefig("Final_Validation_Figure.png", dpi=300)
print("Validation figure successfully generated: Final_Validation_Figure.png")
plt.show()
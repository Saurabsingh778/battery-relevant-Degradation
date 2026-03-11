import numpy as np
from ovito.io import import_file
from ovito.modifiers import AtomicStrainModifier

# 1. Load the trajectory
pipeline = import_file("dump.kovacs")

# 2. Configure the Modifier
modifier = AtomicStrainModifier(
    cutoff = 3.0, 
    reference_frame = 0
)
modifier.output_nonaffine_squared_displacements = True
pipeline.modifiers.append(modifier)

total_frames = pipeline.source.num_frames
d2min_evolution = []
shear_evolution = []

print(f"Parsing {total_frames} frames...")

for frame in range(total_frames):
    data = pipeline.compute(frame)
    
    # THE FIX: Exact string match from your terminal error message
    d2min = np.array(data.particles['Nonaffine Squared Displacement'])
    d2min_evolution.append(np.mean(d2min))
    
    # Extract Shear Strain
    shear = np.array(data.particles['Shear Strain'])
    shear_evolution.append(np.mean(shear))
    
    if frame % 20 == 0:
        print(f"Frame {frame}/{total_frames} | Mean D2min: {d2min_evolution[-1]:.6f}")

# 3. Save metrics
np.savetxt("metric_d2min.txt", d2min_evolution)
np.savetxt("metric_shear.txt", shear_evolution)

print("\nSuccess! Files metric_d2min.txt and metric_shear.txt created.")
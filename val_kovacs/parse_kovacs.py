# diagnose_kovacs.py
from ovito.io import import_file
from ovito.modifiers import AtomicStrainModifier

print("Loading trajectory...")
pipeline = import_file("dump.kovacs")

# Initialize modifier
modifier = AtomicStrainModifier(
    cutoff = 3.0,
    reference_frame = 0
)

# Let's explicitly try to force it to output the full tensors
try:
    modifier.output_strain_tensors = True
except AttributeError:
    pass # Ignore if this specific version of OVITO uses a different flag

pipeline.modifiers.append(modifier)

print("Computing final frame to check maximum strain generation...")
# We compute the very last frame to ensure the physics have evolved
last_frame = pipeline.source.num_frames - 1
data = pipeline.compute(last_frame)

print("\n=== OVITO DIAGNOSTIC REPORT ===")
print(f"Total Atoms: {data.particles.count}")
print(f"Total Frames: {pipeline.source.num_frames}")

print("\n--- Available Particle Properties ---")
for prop in data.particles.keys():
    print(f" -> '{prop}'")

print("\n--- Modifier Internal Attributes ---")
# This prints all the settings available inside the AtomicStrainModifier
for attr in dir(modifier):
    if not attr.startswith("_"):
        print(f" -> {attr} = {getattr(modifier, attr)}")
from PIL import Image

# fix heatmap
img = Image.open("permutation_importance_heatmap.png")
img.convert("RGB").save("permutation_importance_heatmap_fixed.png")

# fix bar plot
img = Image.open("permutation_importance_bar.png")
img.convert("RGB").save("permutation_importance_bar_fixed.png")
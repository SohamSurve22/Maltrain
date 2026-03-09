# visualize_embeddings.py
"""
High-level research-grade embedding visualization using UMAP.
Loads combined embeddings and labels from results/phase2/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap  # pip install umap-learn
# from sklearn.manifold import TSNE  # alternative to UMAP if preferred
import os

# ---------------------------
# Load embeddings and labels
# ---------------------------
EMBED_DIR = "results/phase2"
embeddings = np.load(os.path.join(EMBED_DIR, "all_embeddings.npy"))
labels = np.load(os.path.join(EMBED_DIR, "all_labels.npy"))

print(f"Loaded embeddings: {embeddings.shape}, Labels: {labels.shape}")

# ---------------------------
# Dimensionality Reduction
# ---------------------------
reducer = umap.UMAP(
    n_neighbors=30,       # controls local vs global structure
    min_dist=0.3,         # controls cluster tightness
    metric='cosine',
    random_state=42
)
emb_2d = reducer.fit_transform(embeddings)
print(f"2D embeddings shape: {emb_2d.shape}")

# ---------------------------
# Prepare DataFrame for plotting
# ---------------------------
df = pd.DataFrame({
    'x': emb_2d[:,0],
    'y': emb_2d[:,1],
    'label': labels
})

# ---------------------------
# Assign colors to families
# ---------------------------
unique_labels = np.unique(labels)
palette = sns.color_palette("tab20", n_colors=len(unique_labels))
label_to_color = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
df['color'] = df['label'].map(label_to_color)

# ---------------------------
# Create figure
# ---------------------------
plt.figure(figsize=(14,12))
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='label',
    palette=label_to_color,
    s=35,         # marker size
    alpha=0.9,    # slightly transparent for dense clusters
    linewidth=0
)

plt.title("UMAP Visualization of Malware Embeddings", fontsize=16)
plt.xlabel("UMAP-1", fontsize=12)
plt.ylabel("UMAP-2", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=9)
plt.tight_layout()

# Ensure output folder exists
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/umap_malware_embeddings.png", dpi=300)
plt.show()

# ---------------------------
# Optional: annotate centroids for each family
# ---------------------------
centroids = df.groupby('label')[['x','y']].mean()
for idx, row in centroids.iterrows():
    plt.text(row['x'], row['y'], idx, fontsize=9, weight='bold')
# final_family_similarity_heatmap.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Configuration
# ==========================
EMBEDDINGS_FILE = "results/phase2/all_embeddings.npy"
LABELS_NAMED_FILE = "results/phase2/labels_named.npy"  # mapped names from previous step
OUTPUT_HEATMAP_FILE = "results/phase2/family_similarity_heatmap_realnames.png"

# Optional: limit top N families for clarity
TOP_N_FAMILIES = None  # e.g., 15 to show top 15 families

# ==========================
# Step 1: Load embeddings and named labels
# ==========================
embeddings = np.load(EMBEDDINGS_FILE)
labels_named = np.load(LABELS_NAMED_FILE, allow_pickle=True)

# Optional: filter top N families by sample count
if TOP_N_FAMILIES is not None:
    from collections import Counter
    counts = Counter(labels_named)
    top_families = [fam for fam, _ in counts.most_common(TOP_N_FAMILIES)]
    mask = [label in top_families for label in labels_named]
    embeddings = embeddings[mask]
    labels_named = labels_named[mask]

# ==========================
# Step 2: Compute family centroids
# ==========================
unique_families = sorted(list(set(labels_named)))
family_embeddings = {}
for family in unique_families:
    mask = labels_named == family
    family_embeddings[family] = embeddings[mask].mean(axis=0)

# ==========================
# Step 3: Compute similarity matrix
# ==========================
centroid_matrix = np.stack([family_embeddings[f] for f in unique_families])
similarity_matrix = cosine_similarity(centroid_matrix)

# ==========================
# Step 4: Plot heatmap
# ==========================
plt.figure(figsize=(14, 12))
sns.set(font_scale=1.0)
sns.heatmap(
    similarity_matrix,
    xticklabels=unique_families,
    yticklabels=unique_families,
    cmap="viridis",
    annot=False,
    square=True,
    cbar_kws={'label': 'Cosine Similarity'}
)
plt.title("Malware Family Similarity Heatmap (Real Names)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP_FILE, dpi=300)
plt.show()

print(f"[INFO] Heatmap saved to {OUTPUT_HEATMAP_FILE} with {len(unique_families)} families.")
# prepare_embeddings.py
"""
Combine train, validation, and test embeddings with labels
and save as a single dataset for analysis and visualization.
Works for embeddings stored in 'results/phase2/'.
"""

import numpy as np
import pandas as pd
import os

# Folder where your embeddings are stored
EMBED_DIR = "results/phase2"

# Ensure output folders exist
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(os.path.join(EMBED_DIR, "csv"), exist_ok=True)

# File paths
embedding_files = {
    "train": os.path.join(EMBED_DIR, "train_embeddings.npy"),
    "val": os.path.join(EMBED_DIR, "val_embeddings.npy"),
    "test": os.path.join(EMBED_DIR, "test_embeddings.npy")
}

label_files = {
    "train": os.path.join(EMBED_DIR, "train_labels.npy"),
    "val": os.path.join(EMBED_DIR, "val_labels.npy"),
    "test": os.path.join(EMBED_DIR, "test_labels.npy")
}

# Check all files exist
for key, path in {**embedding_files, **label_files}.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load and combine embeddings and labels
embeddings_list = []
labels_list = []

for split in ["train", "val", "test"]:
    emb = np.load(embedding_files[split])
    lbl = np.load(label_files[split])
    embeddings_list.append(emb)
    labels_list.append(lbl)

# Stack all embeddings and labels
all_embeddings = np.vstack(embeddings_list)
all_labels = np.concatenate(labels_list)

print(f"Total samples: {all_embeddings.shape[0]}, Embedding dim: {all_embeddings.shape[1]}")

# Save combined numpy arrays
np.save(os.path.join(EMBED_DIR, "all_embeddings.npy"), all_embeddings)
np.save(os.path.join(EMBED_DIR, "all_labels.npy"), all_labels)
print(f"Saved combined embeddings and labels in {EMBED_DIR}")

# Optional: save as CSV for inspection or plotting
df = pd.DataFrame(all_embeddings)
df['label'] = all_labels
csv_path = os.path.join(EMBED_DIR, "csv", "all_embeddings_labels.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV at {csv_path}")
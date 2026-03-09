# interactive_umap_malware.py

import os
import numpy as np
import pandas as pd
import plotly.express as px
import umap
import webbrowser

# ==========================
# Paths
# ==========================
EMBEDDINGS_PATH = 'results/phase2/all_embeddings.npy'
LABELS_NAMED_PATH = 'results/phase2/labels_named.npy'          # mapped family names
FIGURE_PATH = 'figures/interactive_malware_embeddings.html'
FIGURE_DIR = os.path.dirname(FIGURE_PATH)

# Ensure output directory exists
os.makedirs(FIGURE_DIR, exist_ok=True)

# ==========================
# Load embeddings and labels
# ==========================
embeddings = np.load(EMBEDDINGS_PATH)
labels_named = np.load(LABELS_NAMED_PATH, allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels_named.shape}")
print(f"Sample labels: {labels_named[:10]}")

# ==========================
# UMAP dimensionality reduction
# ==========================
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)
print(f"UMAP embeddings shape: {umap_embeddings.shape}")

# ==========================
# Create DataFrame for Plotly
# ==========================
viz_df = pd.DataFrame({
    'UMAP1': umap_embeddings[:, 0],
    'UMAP2': umap_embeddings[:, 1],
    'Family': labels_named
})

# ==========================
# Plotly Express interactive scatter plot
# ==========================
fig = px.scatter(
    viz_df,
    x='UMAP1',
    y='UMAP2',
    color='Family',
    hover_name='Family',
    opacity=0.8,
    width=1000,
    height=800
)
fig.update_traces(marker=dict(size=6))
fig.update_traces(hovertemplate='<b>%{hovertext}</b>')
fig.update_layout(
    legend_title_text='Malware Family',
    title='Interactive Malware Embeddings Visualization',
    hovermode='closest'
)

# ==========================
# Save and open plot
# ==========================
fig.write_html(FIGURE_PATH)
print(f"Saved interactive plot to {FIGURE_PATH}")
webbrowser.open(FIGURE_PATH)
print("Plot opened in browser.")
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

# ────────────────────────────────────────────────
#   Your 25 malware families — IMPORTANT: must match training order exactly
# ────────────────────────────────────────────────
class_names = [
    "Adialer.C",
    "Agent.FYI",
    "Allaple.A",
    "Allaple.L",
    "Alueron.gen!J",
    "Autorun.K",
    "C2LOP.gen!g",
    "C2LOP.P",
    "Dialplatform.B",
    "Dontovo.A",
    "Fakerean",
    "Instantaccess",
    "Lolyda.AA1",
    "Lolyda.AA2",
    "Lolyda.AA3",
    "Lolyda.AT",
    "Malex.gen!J",
    "Obfuscator.AD",
    "Rbot!gen",
    "Skintrim.N",
    "Swizzor.gen!E",
    "Swizzor.gen!I",
    "VB.AT",
    "Wintrim.BX",
    "Yuner.A"
]

def build_similarity_graph(
    features,
    labels,
    class_names,
    threshold=0.62,           # ← start here, tune between 0.58–0.68
    min_samples_per_class=5   # optional safety
):
    """
    Build graph where nodes = malware families
    Edge exists if cosine similarity between class centroids > threshold
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Safety check
    valid_classes = unique_classes[counts >= min_samples_per_class]
    if len(valid_classes) < len(unique_classes):
        print(f"Warning: some classes have < {min_samples_per_class} samples → excluded")

    centroids = []
    family_names = []

    for cls in valid_classes:
        mask = (labels == cls)
        class_features = features[mask]
        if len(class_features) == 0:
            continue
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)
        family_names.append(class_names[cls])   # assumes class_names[cls] is correct index

    if not centroids:
        raise ValueError("No valid class centroids found")

    centroids = np.array(centroids)

    # Cosine similarity (most common for CNN embeddings)
    # We normalize here in case you didn't do it earlier
    centroids = normalize(centroids)

    distance_matrix = cdist(centroids, centroids, metric='cosine')
    similarity_matrix = 1 - distance_matrix

    # Build graph
    G = nx.Graph()

    for name in family_names:
        G.add_node(name)

    n = len(family_names)
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim > threshold:
                G.add_edge(
                    family_names[i],
                    family_names[j],
                    weight=sim
                )

    print(f"Graph built - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(threshold = {threshold})")
    return G


def plot_similarity_graph(G, layout_k=0.65, node_size=2200, font_size=10):
    if G.number_of_nodes() == 0:
        print("Empty graph — nothing to plot")
        return

    plt.figure(figsize=(16, 12))

    # spring_layout usually works best for this kind of graph
    pos = nx.spring_layout(G, k=layout_k, iterations=80, seed=42)

    # Edge width proportional to similarity
    edges = G.edges(data=True)
    weights = [d['weight'] * 4.5 for _, _, d in edges]   # tune multiplier if needed

    nx.draw_networkx_nodes(G, pos,
                           node_size=node_size,
                           node_color="lightsteelblue",
                           edgecolors="darkblue",
                           linewidths=1.8)

    nx.draw_networkx_edges(G, pos,
                           width=weights,
                           edge_color="darkorange",
                           alpha=0.75)

    nx.draw_networkx_labels(G, pos,
                            font_size=font_size,
                            font_weight="bold",
                            font_color="black")

    plt.title("Malware Family Similarity Graph\n(CNN Embedding Space – Centroid Cosine Similarity)", 
              fontsize=14, fontweight="bold", pad=20)
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────
#          Main execution block
# ────────────────────────────────────────────────
if __name__ == "__main__":

    # Adjust paths if you're running from different directory
    FEATURES_PATH = "results/best_model_cnn_embeddings.npy"
    LABELS_PATH   = "results/best_model_embedding_labels.npy"

    try:
        features = np.load(FEATURES_PATH)
        labels   = np.load(LABELS_PATH)
        print(f"Loaded embeddings: {features.shape[0]} samples × {features.shape[1]} dims")
        print(f"Labels shape: {labels.shape}, unique classes: {len(np.unique(labels))}")
    except Exception as e:
        print(f"Error loading files:\n{e}")
        exit(1)

    # Usually good to normalize again (idempotent if already normalized)
    features = normalize(features)

    # Try different thresholds — 0.60–0.67 usually gives nicest / most interpretable graphs
    G = build_similarity_graph(
        features,
        labels,
        class_names,
        threshold=0.62
    )

    # Tune layout_k if nodes are too close / too far apart
    plot_similarity_graph(G, layout_k=0.70, node_size=2400, font_size=10.5)
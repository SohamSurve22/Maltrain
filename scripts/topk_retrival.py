import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ==========================
# Configuration
# ==========================
EMBEDDINGS_FILE = "results/phase2/all_embeddings.npy"
LABELS_NAMED_FILE = "results/phase2/labels_named.npy"
TOP_K = 5  # retrieve top 5 similar samples per query
RESULTS_CSV = "results/phase2/malware_retrieval_topk.csv"

# ==========================
# Load embeddings and labels
# ==========================
embeddings = np.load(EMBEDDINGS_FILE)
labels_named = np.load(LABELS_NAMED_FILE, allow_pickle=True)
num_samples = embeddings.shape[0]
print(f"[INFO] Loaded {num_samples} embeddings and labels.")

# ==========================
# Compute full cosine similarity matrix
# ==========================
similarity_matrix = cosine_similarity(embeddings)
print("[INFO] Computed similarity matrix.")

# ==========================
# Retrieve top-k similar samples for each query
# ==========================
retrieval_results = []

for i in range(num_samples):
    # Ignore self-similarity
    sim_scores = similarity_matrix[i]
    sim_scores[i] = -1  # exclude query itself

    # Get top-k indices
    topk_indices = sim_scores.argsort()[::-1][:TOP_K]
    topk_labels = labels_named[topk_indices]
    topk_scores = sim_scores[topk_indices]

    # Check retrieval accuracy (how many top-k belong to the same family)
    query_family = labels_named[i]
    correct_count = sum(topk_labels == query_family)
    accuracy = correct_count / TOP_K

    # Store result
    retrieval_results.append({
        "Query_Index": i,
        "Query_Family": query_family,
        "TopK_Indices": topk_indices.tolist(),
        "TopK_Families": topk_labels.tolist(),
        "TopK_Scores": topk_scores.tolist(),
        "TopK_Accuracy": accuracy
    })

# ==========================
# Save results to CSV
# ==========================
df_results = pd.DataFrame(retrieval_results)
df_results.to_csv(RESULTS_CSV, index=False)
print(f"[INFO] Malware retrieval results saved to {RESULTS_CSV}")

# ==========================
# Summary statistics
# ==========================
mean_accuracy = df_results["TopK_Accuracy"].mean()
print(f"[INFO] Mean Top-{TOP_K} Retrieval Accuracy: {mean_accuracy:.4f}")
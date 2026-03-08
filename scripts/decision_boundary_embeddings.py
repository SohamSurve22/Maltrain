import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

phase2 = os.path.join(BASE_DIR, "results", "phase2")
models_dir = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------
# Load embeddings
# ---------------------------------------------------
X = np.load(os.path.join(phase2, "test_embeddings.npy"))
y = np.load(os.path.join(phase2, "test_labels.npy"))

print("Embeddings:", X.shape)

# ---------------------------------------------------
# Reduce to 2D using PCA
# ---------------------------------------------------
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print("Reduced embeddings:", X_2d.shape)

# ---------------------------------------------------
# Train models in 2D space (for visualization)
# ---------------------------------------------------
svm = SVC(kernel="rbf", gamma="auto")
knn = KNeighborsClassifier(n_neighbors=5)

svm.fit(X_2d, y)
knn.fit(X_2d, y)

# ---------------------------------------------------
# Mesh grid for decision boundary
# ---------------------------------------------------
x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
svm_pred = svm.predict(grid).reshape(xx.shape)
knn_pred = knn.predict(grid).reshape(xx.shape)

# ---------------------------------------------------
# Plot
# ---------------------------------------------------
fig, axes = plt.subplots(1,2, figsize=(16,7))

# ---- SVM ----
axes[0].contourf(xx, yy, svm_pred, alpha=0.3, cmap="tab20")
axes[0].scatter(X_2d[:,0], X_2d[:,1], c=y, cmap="tab20", s=8)
axes[0].set_title("SVM Decision Boundary on Malware Embeddings")

# ---- KNN ----
axes[1].contourf(xx, yy, knn_pred, alpha=0.3, cmap="tab20")
axes[1].scatter(X_2d[:,0], X_2d[:,1], c=y, cmap="tab20", s=8)
axes[1].set_title("KNN Decision Boundary on Malware Embeddings")

plt.tight_layout()

save_path = os.path.join(BASE_DIR, "results", "decision_boundary_embeddings.png")

plt.savefig(save_path, dpi=400)

plt.show()

print("Saved figure:", save_path)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "splits"
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_best_model.h5"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("Loading dataset...")

X_test = np.load(DATA_PATH / "X_test.npy")
y_test = np.load(DATA_PATH / "y_test.npy")

print("Dataset loaded")
print("X_test shape:", X_test.shape)


# --------------------------------------------------
# Malware Family Names
# --------------------------------------------------

class_names = [
"Adialer.C","Agent.FYI","Allaple.A","Allaple.L","Alueron.gen!J",
"Autorun.K","C2LOP.gen!g","C2LOP.P","Dialplatform.B","Dontovo.A",
"Fakerean","Instantaccess","Lolyda.AA1","Lolyda.AA2","Lolyda.AA3",
"Lolyda.AT","Malex.gen!J","Obfuscator.AD","Rbot!gen","Skintrim.N",
"Swizzor.gen!E","Swizzor.gen!I","VB.AT","Wintrim.BX","Yuner.A"
]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

model.build((None,64,64,1))

print("Model loaded")


# --------------------------------------------------
# Extract CNN Features
# --------------------------------------------------

feature_model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

print("Extracting CNN features...")

features = feature_model.predict(X_test, batch_size=32, verbose=1)

print("Feature shape:", features.shape)


# --------------------------------------------------
# PCA
# --------------------------------------------------

print("Running PCA...")

pca = PCA(n_components=50, random_state=42)
features_pca = pca.fit_transform(features)


# --------------------------------------------------
# t-SNE
# --------------------------------------------------

print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    random_state=42
)

tsne_features = tsne.fit_transform(features_pca)

print("t-SNE completed")


# --------------------------------------------------
# Publication Quality Plot
# --------------------------------------------------

plt.figure(figsize=(14,12))

colors = plt.cm.tab20(np.linspace(0,1,20))
extra_colors = plt.cm.Set3(np.linspace(0,1,5))
colors = np.vstack((colors,extra_colors))

unique_labels = np.unique(y_test)

for label in unique_labels:

    idx = y_test == label

    x = tsne_features[idx,0]
    y = tsne_features[idx,1]

    plt.scatter(
        x,
        y,
        s=10,
        alpha=0.7,
        color=colors[label]
    )

    # cluster centroid
    cx = np.mean(x)
    cy = np.mean(y)

    plt.text(
        cx,
        cy,
        class_names[label],
        fontsize=9,
        weight="bold",
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.7,
            edgecolor="none",
            boxstyle="round,pad=0.3"
        )
    )

plt.title(
    "t-SNE Visualization of CNN Feature Embeddings for Malware Families",
    fontsize=16,
    weight="bold"
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.grid(alpha=0.15)

plt.tight_layout()

output_file = RESULTS_PATH / "tsne_malware_publication.png"

plt.savefig(output_file, dpi=400, bbox_inches="tight")

plt.show()

print("Publication-quality plot saved to:", output_file)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

emb_path = os.path.join(BASE_DIR, "results", "phase2", "test_embeddings.npy")
lab_path = os.path.join(BASE_DIR, "results", "phase2", "test_labels.npy")

X = np.load(emb_path)
y = np.load(lab_path)

family_names = {
0:"Adialer.C",
1:"Agent.FYI",
2:"Allaple.A",
3:"Allaple.L",
4:"Alueron.gen!J",
5:"Autorun.K",
6:"C2LOP.gen!g",
7:"C2LOP.P",
8:"Dialplatform.B",
9:"Dontovo.A",
10:"Fakerean",
11:"Instantaccess",
12:"Lolyda.AA1",
13:"Lolyda.AA2",
14:"Lolyda.AA3",
15:"Lolyda.AT",
16:"Malex.gen!J",
17:"Obfuscator.AD",
18:"Rbot!gen",
19:"Skintrim.N",
20:"Swizzor.gen!E",
21:"Swizzor.gen!I",
22:"VB.AT",
23:"Wintrim.BX",
24:"Yuner.A"
}

print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# t-SNE
# -----------------------------
print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000,
    verbose=1
)

X_tsne = tsne.fit_transform(X)

print("t-SNE complete")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(15,13))

unique_labels = np.unique(y)
colors = plt.cm.tab20(np.linspace(0,1,len(unique_labels)))

for i, label in enumerate(unique_labels):

    idx = y == label

    plt.scatter(
        X_tsne[idx,0],
        X_tsne[idx,1],
        s=12,
        color=colors[i],
        label=family_names[label],
        alpha=0.7
    )

    # Cluster center
    center_x = np.mean(X_tsne[idx,0])
    center_y = np.mean(X_tsne[idx,1])

    plt.text(
        center_x,
        center_y,
        family_names[label],
        fontsize=8,
        weight="bold"
    )

plt.title(
"Malware Family Clustering using CNN Embeddings (t-SNE)",
fontsize=18
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.legend(
    bbox_to_anchor=(1.05,1),
    loc="upper left",
    fontsize=8
)

plt.grid(alpha=0.2)

plt.tight_layout()

save_path = os.path.join(BASE_DIR,"results","tsne_malware_clusters_named.png")
plt.savefig(save_path,dpi=400)

plt.show()
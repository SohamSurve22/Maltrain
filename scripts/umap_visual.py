import numpy as np
import matplotlib.pyplot as plt
import umap
import os

# -----------------------------
# Malware family names
# -----------------------------
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

# -----------------------------
# Load embeddings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

emb_path = os.path.join(BASE_DIR,"results","phase2","test_embeddings.npy")
lab_path = os.path.join(BASE_DIR,"results","phase2","test_labels.npy")

X = np.load(emb_path)
y = np.load(lab_path)

print("Embeddings:",X.shape)

# -----------------------------
# UMAP
# -----------------------------
print("Running UMAP...")

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.2,
    n_components=2,
    random_state=42
)

X_umap = reducer.fit_transform(X)

print("UMAP complete")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(15,13))

unique_labels = np.unique(y)
colors = plt.cm.tab20(np.linspace(0,1,len(unique_labels)))

for i,label in enumerate(unique_labels):

    idx = y==label

    plt.scatter(
        X_umap[idx,0],
        X_umap[idx,1],
        s=12,
        color=colors[i],
        label=family_names[label],
        alpha=0.7
    )

    cx = np.mean(X_umap[idx,0])
    cy = np.mean(X_umap[idx,1])

    plt.text(
        cx,
        cy,
        family_names[label],
        fontsize=8,
        weight="bold"
    )

plt.title("Malware Family Manifold using CNN Embeddings (UMAP)",fontsize=18)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.legend(
    bbox_to_anchor=(1.05,1),
    loc="upper left",
    fontsize=8
)

plt.grid(alpha=0.2)

plt.tight_layout()

save_path = os.path.join(BASE_DIR,"results","umap_malware_clusters.png")
plt.savefig(save_path,dpi=400)

plt.show()
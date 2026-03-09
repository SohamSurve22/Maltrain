# Maltrain: Malware Clustering and Visualization Framework

## 1. Project Overview
**Maltrain** is a deep learning framework designed for **malware family clustering, analysis, and visualization**. It leverages **Convolutional Neural Networks (CNNs)** to extract high-dimensional embeddings from malware images (Malimg dataset) and employs dimensionality reduction, similarity graphs, and interactive visualization to analyze and explore malware family relationships.

**Key goals:**
- Accurately classify malware families using CNN embeddings.  
- Provide interpretable 2D visualizations (t-SNE, UMAP) for research analysis.  
- Evaluate classical ML methods on CNN embeddings for benchmarking.  
- Enable interactive malware retrieval based on embedding similarity.  

---

## 2. Repository Structure
Maltrain/
├─ data/ # Raw and processed dataset splits
├─ scripts/ # Primary execution scripts (training, extraction, visualization)
├─ src/ # Core logic for EDA, model definitions, utilities
├─ models/ # Trained CNN models and classical ML classifiers
├─ results/ # Plots, embeddings, similarity matrices, CSV reports
├─ figures/ # Interactive HTML visualizations
├─ logs/ # (Optional) training and execution logs
└─ README.md # Project documentation


---

## 3. Component Breakdown

| Component | Purpose | Input Files | Output Files | Script | Status |
|-----------|--------|------------|-------------|--------|--------|
| Data Preparation | Load, resize, normalize malware images, create train/test splits | data/raw/Malimg | data/splits/X_train.npy, y_train.npy, ... | scripts/prepare_data.py | ✅ Complete |
| CNN Training | Train 3-layer CNN with 256-D embedding layer | data/splits/*.npy | models/cnn_best_model.h5, cnn_final_model.keras | scripts/cnn_training.py | ✅ Complete |
| Embedding Extraction | Extract high-level CNN feature vectors | models/cnn_best_model.h5, data/splits/*.npy | results/phase2/*_embeddings.npy, *_labels.npy | scripts/extract_embedding.py | ✅ Complete |
| Classical ML Comparison | Benchmark embeddings with KNN, SVM, Random Forest | results/phase2/*.npy | results/baseline_comparison.csv, models/*_classifier.pkl | scripts/train_classical_ml.py, scripts/baseline_benchmark.py | ✅ Complete |
| Similarity Graph | Construct family-level network using centroid cosine similarity | results/phase2/*.npy | results/malware_similarity_network.png | scripts/similarity_analysis_network.py | ✅ Complete |
| Embedding Visualization | t-SNE/UMAP 2D projections | results/phase2/*.npy | results/tsne_malware_publication.png, results/umap_malware_clusters.png | scripts/tsne_visualization.py, scripts/umap_visual.py | ✅ Complete |
| Interactive Visualization | Plotly UMAP interactive scatter plot | results/phase2/all_embeddings.npy | figures/interactive_malware_embeddings.html | scripts/interactive_visualization.py | ✅ Complete |
| Heatmaps | Pairwise cosine similarity of malware family centroids | results/phase2/all_embeddings.npy | results/phase2/family_similarity_heatmap_realnames.png | scripts/family_sim_heatmap.py | ✅ Complete |
| Malware Retrieval | Top-K similarity search for nearest neighbor malware lookup | results/phase2/all_embeddings.npy | results/phase2/malware_retrieval_topk.csv | scripts/topk_retrieval.py | ✅ Complete |

---

## 4. Dependencies & Dataset

**Dataset:**
- **Source:** Malimg Malware Dataset (Kaggle)  
- **Format:** Grayscale malware images  
- **Classes:** 25 distinct malware families (e.g., Adialer.C, Allaple.A, Yuner.A)

**Python Libraries:**
- **Deep Learning:** `tensorflow`, `keras`  
- **Data Science:** `numpy`, `pandas`, `scikit-learn`  
- **Visualization:** `matplotlib`, `seaborn`, `plotly`, `networkx`  
- **Dimensionality Reduction:** `umap-learn`, PCA, t-SNE  
- **Image Processing:** `opencv-python`  

---

## 5. Research Experiments & Results

### Classification Performance

| Model | Accuracy |
|-------|---------|
| CNN | ~98.0% |
| SVM (RBF) | ~98.4% |
| KNN | ~98.3% |
| Random Forest | ~98.2% |

*Insight:* CNN embeddings + SVM often slightly outperform the end-to-end CNN classifier due to better margin optimization.

### Structural Insights

- **Clustering:** t-SNE and UMAP plots show clear separation for most malware families.  
- **Similarity Graph:** "Lolyda" variants cluster together, revealing shared code/obfuscation.  
- **Retrieval Accuracy:** Mean Top-5 Retrieval Accuracy is consistently high, proving nearest-neighbor effectiveness.  

### Dimensionality Reduction Comparison

- **PCA:** Captures variance but fails on overlapping families in 2D.  
- **t-SNE:** Excellent for local cluster preservation; used for static, publication-quality plots.  
- **UMAP:** Preserves global structure, faster, used for interactive exploration.

---

## 6. Usage Instructions

Train CNN:

python scripts/cnn_training.py

Extract Embeddings:

python scripts/extract_embedding.py

Classical ML Benchmarking:

python scripts/train_classical_ml.py

Similarity Graph & Heatmaps:

python scripts/similarity_analysis_network.py
python scripts/family_sim_heatmap.py

Embedding Visualization:

python scripts/tsne_visualization.py
python scripts/umap_visual.py
python scripts/interactive_visualization.py

Malware Retrieval:

python scripts/topk_retrieval.py

1. **Data Preparation:**  
   ```bash
   python scripts/prepare_data.py

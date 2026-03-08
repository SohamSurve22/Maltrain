# Maltrain

## Overview

Maltrain is a deep learning–based malware family clustering and visualization framework designed for research exploration in malware classification. The project focuses on extracting learned representations from malware images and analyzing family-level similarity structures using CNN embeddings, PCA dimensionality reduction, and t-SNE visualization.

The primary dataset used in this project is the **Malimg malware dataset** from Kaggle.

Dataset source:
https://www.kaggle.com/datasets/manmandes/malimg

---

## Project Structure

```
Maltrain/
│
├── data/                # Dataset storage (not included in repo)
├── scripts/             # Training, visualization, and analysis scripts
├── src/                 # Core model and feature extraction pipeline
├── results/             # Generated plots, embeddings, and reports
├── models/             # Saved trained models
└── README.md
```

---

## Dataset Setup

### Download Dataset

1. Visit:
   https://www.kaggle.com/datasets/manmandes/malimg

2. Download the dataset archive.

3. Extract the dataset inside the `data/` directory:

```
data/
└── malimg_dataset/
    ├── Family1/
    ├── Family2/
    └── ...
```

The final path should resemble:

```
Maltrain/data/malimg_dataset/
```

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Pipeline

Run preprocessing and model training:

```bash
python scripts/train_cnn.py
```

This script will:

* Load malware image samples
* Perform normalization and augmentation.
* Train CNN feature extractor.
* Save model checkpoints.

---

## Visualization Pipeline

### PCA Projection

```bash
python scripts/pca_visualization.py
```

Generates:

* Family-wise separability plots.

---

### t-SNE Embedding Visualization

```bash
python scripts/tsne_visualization.py
```

Produces publication-style embedding plots for malware family clustering.

---

## Output Results

Results are stored inside:

```
results/
├── embeddings/
├── plots/
└── metrics/
```

Typical outputs include:

* Confusion matrices
* Family similarity graphs
* Dimensionality reduction projections

---

## Research Purpose

This project is intended for:

* Malware family clustering research
* Representation learning analysis
* Visualization of high-dimensional malware embeddings
* Experimental cybersecurity ML exploration

It is not currently optimized for production deployment.

---

## Future Improvements

* Add stratified cross-validation evaluation.
* Incorporate transformer-based malware representation models.
* Extend analysis to adversarial robustness testing.
* Add SHAP-based explainability visualization.
* Benchmark against classical ML baselines.

---

## Citation

If you use this project for academic or research purposes, please cite the repository.

Dataset citation:
Malimg Malware Dataset — https://www.kaggle.com/datasets/manmandes/malimg

---

## Author

Maintained by SohamSurve22

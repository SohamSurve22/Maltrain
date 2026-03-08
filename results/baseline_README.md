# Baseline Performance Benchmarking

This benchmark evaluates three classical models on extracted GLCM texture and histogram features from the Malimg dataset.

## Dataset Split
- Stratified split: 70% train, 10% validation, 20% test
- Fixed random seed for reproducibility

## Feature Extraction
- GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
- Histogram statistics (mean, std, skewness)

## Models
- SVM (RBF kernel)
- Random Forest
- k-NN (cosine distance)

## Metrics
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score

## Results
Results are saved in `results/baseline_comparison.csv`.

## How to Run
1. Prepare data splits and features using `src/prepare_data.py`.
2. Run baseline benchmark:

```bash
python scripts/baseline_benchmark.py
```

## Contact
For questions, contact the project maintainer.

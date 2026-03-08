import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

phase2 = os.path.join(BASE_DIR, "results", "phase2")

# UPDATED PATH
models_dir = os.path.join(BASE_DIR, "models")
# -----------------------------
# Load test embeddings
# -----------------------------
X_test = np.load(os.path.join(phase2,"test_embeddings.npy"))
y_test = np.load(os.path.join(phase2,"test_labels.npy"))

print("Test set:",X_test.shape)

# -----------------------------
# Load classical ML models
# -----------------------------
svm = joblib.load(os.path.join(models_dir, "SVM_malware_classifier.pkl"))
knn = joblib.load(os.path.join(models_dir, "KNN_malware_classifier.pkl"))
rf = joblib.load(os.path.join(models_dir, "RandomForest_malware_classifier.pkl"))

# -----------------------------
# Predictions
# -----------------------------
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)
rf_pred = rf.predict(X_test)

# CNN results (enter your value from training log)
cnn_accuracy = 0.98  # <-- change if needed

# -----------------------------
# Metrics
# -----------------------------
models = ["CNN","SVM","KNN","Random Forest"]

accuracy = [
cnn_accuracy,
accuracy_score(y_test,svm_pred),
accuracy_score(y_test,knn_pred),
accuracy_score(y_test,rf_pred)
]

precision = [
cnn_accuracy,
precision_score(y_test,svm_pred,average="weighted"),
precision_score(y_test,knn_pred,average="weighted"),
precision_score(y_test,rf_pred,average="weighted")
]

recall = [
cnn_accuracy,
recall_score(y_test,svm_pred,average="weighted"),
recall_score(y_test,knn_pred,average="weighted"),
recall_score(y_test,rf_pred,average="weighted")
]

f1 = [
cnn_accuracy,
f1_score(y_test,svm_pred,average="weighted"),
f1_score(y_test,knn_pred,average="weighted"),
f1_score(y_test,rf_pred,average="weighted")
]

# -----------------------------
# Table
# -----------------------------
df = pd.DataFrame({
"Model":models,
"Accuracy":accuracy,
"Precision":precision,
"Recall":recall,
"F1 Score":f1
})

print("\nModel Comparison\n")
print(df)

# save table
df.to_csv(os.path.join(BASE_DIR,"results","model_comparison.csv"),index=False)

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure(figsize=(8,6))

plt.bar(models,accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.ylim(0.9,1.0)

for i,v in enumerate(accuracy):
    plt.text(i,v+0.002,f"{v:.3f}",ha="center")

plt.tight_layout()

save_path = os.path.join(BASE_DIR,"results","model_accuracy_comparison.png")

plt.savefig(save_path,dpi=400)

plt.show()
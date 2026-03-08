import os
import numpy as np
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

EMBED_PATH = BASE_DIR / "results" / "phase2"
MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(exist_ok=True)

print("Loading embeddings from:", EMBED_PATH)

# -------------------------------------------------
# Load embeddings
# -------------------------------------------------

X_train = np.load(EMBED_PATH / "train_embeddings.npy")
X_val   = np.load(EMBED_PATH / "val_embeddings.npy")
X_test  = np.load(EMBED_PATH / "test_embeddings.npy")

y_train = np.load(EMBED_PATH / "train_labels.npy")
y_val   = np.load(EMBED_PATH / "val_labels.npy")
y_test  = np.load(EMBED_PATH / "test_labels.npy")

print("\nDataset Shapes")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# -------------------------------------------------
# Models
# -------------------------------------------------

models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42
    )
}

# -------------------------------------------------
# Training Loop
# -------------------------------------------------

for name, model in models.items():

    print("\n==============================")
    print("Training:", name)
    print("==============================")

    model.fit(X_train, y_train)

    # Validation evaluation
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print("Validation Accuracy:", val_acc)

    # Test evaluation
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print("Test Accuracy:", test_acc)

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Save model
    model_file = MODEL_PATH / f"{name}_malware_classifier.pkl"
    joblib.dump(model, model_file)

    print("\nModel saved to:", model_file)

print("\nAll models trained successfully.")
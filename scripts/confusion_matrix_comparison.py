import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# Malware family names (Malimg dataset)
# ---------------------------------------------------
family_names = [
"Adialer.C",
"Agent.FYI",
"Allaple.A",
"Allaple.L",
"Alueron.gen!J",
"Autorun.K",
"C2LOP.gen!g",
"C2LOP.P",
"Dialplatform.B",
"Dontovo.A",
"Fakerean",
"Instantaccess",
"Lolyda.AA1",
"Lolyda.AA2",
"Lolyda.AA3",
"Lolyda.AT",
"Malex.gen!J",
"Obfuscator.AD",
"Rbot!gen",
"Skintrim.N",
"Swizzor.gen!E",
"Swizzor.gen!I",
"VB.AT",
"Wintrim.BX",
"Yuner.A"
]

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

phase2 = os.path.join(BASE_DIR, "results", "phase2")
models_dir = os.path.join(BASE_DIR, "models")
results_dir = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------
# Load embeddings and labels
# ---------------------------------------------------
X_test = np.load(os.path.join(phase2, "test_embeddings.npy"))
y_test = np.load(os.path.join(phase2, "test_labels.npy"))

print("Test embeddings:", X_test.shape)

# ---------------------------------------------------
# Load classical ML models
# ---------------------------------------------------
svm = joblib.load(os.path.join(models_dir, "SVM_malware_classifier.pkl"))
knn = joblib.load(os.path.join(models_dir, "KNN_malware_classifier.pkl"))
rf = joblib.load(os.path.join(models_dir, "RandomForest_malware_classifier.pkl"))

# ---------------------------------------------------
# CNN model
# NOTE:
# CNN needs images, not embeddings.
# If you have test images saved separately, load them.
# ---------------------------------------------------
cnn_model_path = os.path.join(models_dir, "cnn_best_model.h5")

cnn_available = os.path.exists(cnn_model_path)

if cnn_available:
    cnn = load_model(cnn_model_path)
    print("CNN model loaded.")
else:
    print("CNN model not found — skipping CNN confusion matrix.")

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)
rf_pred = rf.predict(X_test)

# ---------------------------------------------------
# Function to plot confusion matrix
# ---------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14,12))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=family_names
    )

    disp.plot(
        cmap="Blues",
        xticks_rotation=90,
        colorbar=True
    )

    plt.title(f"{model_name} Confusion Matrix", fontsize=18)

    plt.tight_layout()

    save_path = os.path.join(
        results_dir,
        f"{model_name.lower().replace(' ','_')}_confusion_matrix.png"
    )

    plt.savefig(save_path, dpi=400)

    print(f"{model_name} confusion matrix saved to:", save_path)

    plt.show()


# ---------------------------------------------------
# Generate confusion matrices
# ---------------------------------------------------
plot_confusion_matrix(y_test, svm_pred, "SVM")

plot_confusion_matrix(y_test, knn_pred, "KNN")

plot_confusion_matrix(y_test, rf_pred, "Random Forest")

print("All confusion matrices generated.")
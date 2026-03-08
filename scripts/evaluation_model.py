import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd

from plot_confusion_matrix import plot_confusion_matrix
from analyze_confusions import analyze_confusions

# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Load Test Data
# =========================

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Test samples:", X_test.shape)

# =========================
# Load Models
# =========================

best_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "cnn_best_model.h5")
)

final_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "cnn_final_model.h5")
)

# =========================
# Malware Family Names
# =========================

class_names = [
"Adialer.C","Agent.FYI","Allaple.A","Allaple.L","Alueron.gen!J",
"Autorun.K","C2LOP.gen!g","C2LOP.P","Dialplatform.B","Dontovo.A",
"Fakerean","Instantaccess","Lolyda.AA1","Lolyda.AA2","Lolyda.AA3",
"Lolyda.AT","Malex.gen!J","Obfuscator.AD","Rbot!gen","Skintrim.N",
"Swizzor.gen!E","Swizzor.gen!I","VB.AT","Wintrim.BX","Yuner.A"
]

# =========================
# Feature Extractor
# =========================

def get_feature_extractor(model):

    # Force model build if Sequential model is not initialized
    if not model.built:
        model.build((None, *model.input_shape[1:]))

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output
    )

    return feature_model

# =========================
# Embedding Extraction
# =========================

def extract_embeddings(model, X, save_path=None):

    extractor = get_feature_extractor(model)

    embeddings = extractor.predict(X, verbose=1)

    print("Embedding shape:", embeddings.shape)

    if save_path is not None:
        np.save(save_path, embeddings)

    return embeddings

# =========================
# Evaluation Pipeline
# =========================

def evaluate(model, name):

    print("\n===============================")
    print("Evaluating:", name)
    print("===============================")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print("Test Accuracy:", acc)
    print("Test Loss:", loss)

    # Predictions
    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=1)

    y_true = y_test
    y_pred = preds

    # Confusion Matrix Visualization
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        RESULTS_DIR
    )

    # Misclassification Analysis
    analyze_confusions(
        y_true,
        y_pred,
        class_names
    )

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)

    # =========================
    # Save Metrics
    # =========================

    report_path = os.path.join(
        RESULTS_DIR,
        f"{name}_classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    np.savetxt(
        os.path.join(
            RESULTS_DIR,
            f"{name}_confusion_matrix.txt"
        ),
        cm,
        fmt="%d"
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    df = pd.DataFrame(report_dict).transpose()

    df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{name}_classification_report.csv"
        )
    )

    print("\nResults saved to results folder.")

    # =========================
    # CNN Embedding Extraction
    # =========================

    embedding_path = os.path.join(
        RESULTS_DIR,
        f"{name}_cnn_embeddings.npy"
    )

    label_path = os.path.join(
        RESULTS_DIR,
        f"{name}_embedding_labels.npy"
    )

    embeddings = extract_embeddings(
        model,
        X_test,
        save_path=embedding_path
    )

    np.save(label_path, y_test)

    print("Embeddings and labels saved.")

    return embeddings


# =========================
# Run Evaluation
# =========================

if __name__ == "__main__":

    evaluate(best_model, "best_model")
    evaluate(final_model, "final_model")
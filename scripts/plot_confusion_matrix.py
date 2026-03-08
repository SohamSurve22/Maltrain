import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, results_dir):

    cm = confusion_matrix(y_true, y_pred)

    # Normal matrix
    plt.figure(figsize=(14,10))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # Normalized matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14,10))

    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "confusion_matrix_normalized.png"))
    plt.close()

    print("Confusion matrix plots saved.")
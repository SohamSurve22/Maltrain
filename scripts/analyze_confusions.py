import numpy as np
from sklearn.metrics import confusion_matrix

def analyze_confusions(y_true, y_pred, class_names, top_k=10):

    cm = confusion_matrix(y_true, y_pred)

    confusions = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):

            if i != j and cm[i][j] > 0:

                confusions.append(
                    (class_names[i], class_names[j], cm[i][j])
                )

    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\nTop Confused Malware Families\n")
    print("="*40)

    for i in range(min(top_k, len(confusions))):

        true_label, pred_label, count = confusions[i]

        print(f"{true_label} -> {pred_label} : {count} samples")

    return confusions
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "Malimg")
SPLIT_PATH = os.path.join(BASE_DIR, "data", "splits")

os.makedirs(SPLIT_PATH, exist_ok=True)

IMAGE_SIZE = 64


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset():
    X = []
    y = []

    families = sorted(os.listdir(DATASET_PATH))
    label_map = {family: idx for idx, family in enumerate(families)}

    print("Loading images...\n")

    for family in families:
        family_path = os.path.join(DATASET_PATH, family)
        if not os.path.isdir(family_path):
            continue

        for file in os.listdir(family_path):
            img_path = os.path.join(family_path, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            X.append(img)
            y.append(label_map[family])

    X = np.array(X, dtype="float32") / 255.0
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = np.array(y)

    return X, y


# -----------------------------
# Controlled Oversampling
# -----------------------------
def oversample_training(X_train, y_train, max_multiplier=3):
    unique_classes, counts = np.unique(y_train, return_counts=True)

    max_count = max(counts)
    X_balanced = []
    y_balanced = []

    print("\nApplying controlled oversampling...\n")

    for cls, count in zip(unique_classes, counts):
        indices = np.where(y_train == cls)[0]

        multiplier = min(max_count // count, max_multiplier)

        X_cls = X_train[indices]
        y_cls = y_train[indices]

        X_oversampled = np.tile(X_cls, (multiplier, 1, 1, 1))
        y_oversampled = np.tile(y_cls, multiplier)

        X_balanced.append(X_oversampled)
        y_balanced.append(y_oversampled)

        print(f"Class {cls}: original={count}, multiplier={multiplier}")

    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)

    return X_balanced, y_balanced


# -----------------------------
# Main Pipeline
# -----------------------------
if __name__ == "__main__":

    X, y = load_dataset()

    print("\nPerforming stratified split...\n")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("Original training distribution:")
    print(np.unique(y_train, return_counts=True))

    # Hybrid: Oversample ONLY training data
    X_train_bal, y_train_bal = oversample_training(X_train, y_train)

    print("\nBalanced training distribution:")
    print(np.unique(y_train_bal, return_counts=True))

    # Compute class weights on ORIGINAL distribution
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = {
        i: class_weights[i] for i in range(len(class_weights))
    }

    print("\nClass Weights:")
    print(class_weight_dict)

    # Save everything
    np.save(os.path.join(SPLIT_PATH, "X_train.npy"), X_train_bal)
    np.save(os.path.join(SPLIT_PATH, "y_train.npy"), y_train_bal)
    np.save(os.path.join(SPLIT_PATH, "X_val.npy"), X_val)
    np.save(os.path.join(SPLIT_PATH, "y_val.npy"), y_val)
    np.save(os.path.join(SPLIT_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(SPLIT_PATH, "y_test.npy"), y_test)

    print("\nData preparation complete.")
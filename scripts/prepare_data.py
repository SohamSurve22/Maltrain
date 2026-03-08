import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "Malimg")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "splits")

os.makedirs(OUTPUT_PATH, exist_ok=True)

IMG_SIZE = 64

images = []
labels = []

print("Loading malware images...")

# Loop through malware family folders
for family in os.listdir(DATASET_PATH):

    family_path = os.path.join(DATASET_PATH, family)

    if not os.path.isdir(family_path):
        continue

    for file in os.listdir(family_path):

        file_path = os.path.join(family_path, file)

        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(family)

        except Exception:
            continue

print("Total images loaded:", len(images))

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Normalize pixel values
X = X.astype("float32") / 255.0

# Add channel dimension for CNN
X = np.expand_dims(X, axis=-1)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Number of malware families:", len(encoder.classes_))

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=42
)

print("Dataset splits:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Save splits
np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)

np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

print("Dataset preparation complete.")
print("Files saved in:", OUTPUT_PATH)
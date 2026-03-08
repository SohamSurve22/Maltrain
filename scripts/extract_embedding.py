import os
import numpy as np
from tensorflow.keras.models import load_model, Model

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "splits")
MODEL_PATH = os.path.join(BASE_DIR, "models")
RESULT_PATH = os.path.join(BASE_DIR, "results", "phase2")

os.makedirs(RESULT_PATH, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset splits...")

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_val   = np.load(os.path.join(DATA_PATH, "X_val.npy"))
X_test  = np.load(os.path.join(DATA_PATH, "X_test.npy"))

# LOAD LABELS (THIS WAS MISSING)
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_val   = np.load(os.path.join(DATA_PATH, "y_val.npy"))
y_test  = np.load(os.path.join(DATA_PATH, "y_test.npy"))

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# Load Best CNN Model
# -----------------------------
print("\nLoading trained CNN model...")

model = load_model(os.path.join(MODEL_PATH, "cnn_best_model.h5"))

# -----------------------------
# Create Embedding Extractor
# -----------------------------
embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding_layer").output
)

print("Embedding model ready.")

# -----------------------------
# Extract Embeddings
# -----------------------------
print("\nExtracting train embeddings...")
train_embeddings = embedding_model.predict(X_train, batch_size=64)

print("Extracting validation embeddings...")
val_embeddings = embedding_model.predict(X_val, batch_size=64)

print("Extracting test embeddings...")
test_embeddings = embedding_model.predict(X_test, batch_size=64)

# -----------------------------
# Save Embeddings
# -----------------------------
np.save(os.path.join(RESULT_PATH, "train_embeddings.npy"), train_embeddings)
np.save(os.path.join(RESULT_PATH, "val_embeddings.npy"), val_embeddings)
np.save(os.path.join(RESULT_PATH, "test_embeddings.npy"), test_embeddings)

# -----------------------------
# Save Labels (NEW)
# -----------------------------
np.save(os.path.join(RESULT_PATH, "train_labels.npy"), y_train)
np.save(os.path.join(RESULT_PATH, "val_labels.npy"), y_val)
np.save(os.path.join(RESULT_PATH, "test_labels.npy"), y_test)

# -----------------------------
# Print Results
# -----------------------------
print("\nEmbeddings saved successfully.")

print("Train embeddings shape:", train_embeddings.shape)
print("Validation embeddings shape:", val_embeddings.shape)
print("Test embeddings shape:", test_embeddings.shape)

print("\nLabels saved successfully.")
print("Train labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)
print("Test labels shape:", y_test.shape)

print("\nSaved to folder:")
print(RESULT_PATH)
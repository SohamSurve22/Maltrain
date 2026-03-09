import numpy as np

# File paths
embeddings_path = "results/phase2/all_embeddings.npy"
labels_path = "results/phase2/all_labels.npy"
classes_path = "results/phase2/label_encoder_classes.npy"

def load_npy_safe(path):
    try:
        arr = np.load(path, allow_pickle=True)
        return arr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Load files
embeddings = load_npy_safe(embeddings_path)
labels = load_npy_safe(labels_path)
classes = load_npy_safe(classes_path)

print("=== Embeddings ===")
if embeddings is not None:
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
else:
    print("Embeddings not loaded.")

print("\n=== Labels ===")
if labels is not None:
    print(f"Shape: {labels.shape}")
    print(f"Dtype: {labels.dtype}")
    print(f"First 20 values: {labels.flatten()[:20]}")
else:
    print("Labels not loaded.")

print("\n=== Label Encoder Classes ===")
if classes is not None:
    print(f"Type: {type(classes)}")
    try:
        print(f"Length: {len(classes)}")
    except Exception as e:
        print(f"Could not determine length: {e}")
    print(f"Sample values: {classes[:10]}")
else:
    print("Classes not loaded.")

# Flatten labels if needed
if labels is not None:
    labels_flat = labels.flatten()
else:
    labels_flat = None

# Check label mapping
print("\n=== Label Mapping Checks ===")
if labels_flat is not None and classes is not None:
    invalid_labels = []
    for idx, label in enumerate(labels_flat):
        if not (0 <= label < len(classes)):
            invalid_labels.append((idx, label))
    all_valid = len(invalid_labels) == 0
    print(f"All label indices valid: {all_valid}")
    print(f"Number of unique families in labels: {len(np.unique(labels_flat))}")
    print(f"Number of class names: {len(classes)}")
    if not all_valid:
        print(f"Invalid label indices found: {len(invalid_labels)}")
        for i, (idx, label) in enumerate(invalid_labels[:10]):
            print(f"  At index {idx}: label={label}")
        # Show embeddings for problematic labels
        if embeddings is not None:
            print("\nSample embeddings for problematic labels:")
            for idx, label in invalid_labels[:5]:
                print(f"Index {idx}, label {label}, embedding: {embeddings[idx]}")
    else:
        print("No invalid label indices found.")
else:
    print("Cannot check label mapping due to missing data.")

print("\n=== End of Report ===")
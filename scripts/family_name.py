import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Path to your original dataset labels (e.g., all malware families)
# For example, if you have a list of family names per sample:
all_families = np.load("results/phase2/all_labels.npy")  # or however you stored original labels

# Create LabelEncoder
le = LabelEncoder()
le.fit(all_families)

# This is the correct order of family names corresponding to numeric labels
family_names_in_order = list(le.classes_)
print(family_names_in_order)
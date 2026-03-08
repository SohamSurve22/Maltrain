import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load splits
import os
split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'splits'))
X_train = np.load(os.path.join(split_dir, 'X_train.npy'))
y_train = np.load(os.path.join(split_dir, 'y_train.npy'))
X_test = np.load(os.path.join(split_dir, 'X_test.npy'))
y_test = np.load(os.path.join(split_dir, 'y_test.npy'))

# Flatten images for classical ML baselines
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

models = {
    'SVM_RBF': SVC(kernel='rbf', random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNN_Cosine': KNeighborsClassifier(metric='cosine')
}

results = []

for name, model in models.items():

    print(f"Training {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Macro_Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('../results/baseline_comparison.csv', index=False)

print(results_df)
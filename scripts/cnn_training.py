import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "splits")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))

y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

num_classes = len(np.unique(y_train))

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)
print("Number of classes:", num_classes)

# -----------------------------
# CNN Architecture
# -----------------------------
print("Building CNN model...")

inputs = Input(shape=(64, 64, 1))

x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)

# Embedding layer (IMPORTANT for later phases)
embedding = Dense(256, activation='relu', name="embedding_layer")(x)

x = Dropout(0.5)(embedding)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Training Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_PATH, "cnn_best_model.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# -----------------------------
# Train Model
# -----------------------------
print("Training CNN...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# Evaluate Model
# -----------------------------
print("Evaluating on test set...")

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)

# -----------------------------
# Save Final Model
# -----------------------------
model.save(os.path.join(MODEL_PATH, "cnn_final_model.keras"))

print("Training complete.")
print("Best model saved as cnn_best_model.h5")
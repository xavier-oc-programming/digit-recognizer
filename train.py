import os
# Disable Apple Metal GPU so BatchNormalization statistics are computed in
# standard CPU float32 — required for portable deployment to Linux servers.
# Without this, moving_mean/moving_variance accumulate with Metal precision
# and produce shifted activations on CPU inference (Azure, HF Spaces, Docker).
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from config import (
    MODEL_DIR, PLOTS_DIR, RANDOM_STATE,
    IMG_SIZE, NUM_CLASSES, INPUT_SHAPE,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT
)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ── Data loading & preprocessing ──────────────────────────────────────────────

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# Add channel dimension: (n, 28, 28) → (n, 28, 28, 1)
X_train = X_train_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test  = X_test_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels: integer class → probability vector length 10
y_train = to_categorical(y_train_raw, NUM_CLASSES)
y_test  = to_categorical(y_test_raw,  NUM_CLASSES)

print("Dataset shapes")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print("\nClass distribution (train):")
for digit in range(NUM_CLASSES):
    count = np.sum(y_train_raw == digit)
    print(f"  Digit {digit}: {count:,}")


# ── Exploratory plots ──────────────────────────────────────────────────────────

# 01 — sample digits: one example per class
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit, ax in enumerate(axes.flat):
    idx = np.where(y_train_raw == digit)[0][0]
    ax.imshow(X_train_raw[idx], cmap='gray')
    ax.set_title(f'Digit: {digit}', fontsize=11)
    ax.axis('off')
fig.suptitle('Sample Digits — One per Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_sample_digits.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 01_sample_digits.png")

# 02 — pixel intensity distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(X_train_raw.flatten(), bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
ax.set_xlabel('Pixel intensity (0–255)')
ax.set_ylabel('Frequency')
ax.set_title('Pixel Intensity Distribution — Training Set', fontsize=13, fontweight='bold')
ax.annotate('Mostly black background', xy=(10, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > 0 else 1),
            fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_pixel_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_pixel_distribution.png")

# 03 — class distribution
train_counts = [np.sum(y_train_raw == d) for d in range(NUM_CLASSES)]
test_counts  = [np.sum(y_test_raw  == d) for d in range(NUM_CLASSES)]
x = np.arange(NUM_CLASSES)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 0.2, train_counts, 0.4, label='Train', color='steelblue')
ax.bar(x + 0.2, test_counts,  0.4, label='Test',  color='coral')
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in range(NUM_CLASSES)])
ax.set_xlabel('Digit')
ax.set_ylabel('Count')
ax.set_title('Class Distribution — MNIST is Approximately Balanced', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_class_distribution.png")


# ── Model architecture ─────────────────────────────────────────────────────────

model = Sequential([
    # Block 1: feature detection
    # Conv2D(32): learns 32 different feature detectors (edges, curves, corners)
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE, padding='same'),
    # BatchNormalization: normalises layer inputs, stabilises training, allows higher learning rates
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D: reduces spatial dimensions by half, retains dominant features,
    # adds translation invariance — a "7" drawn slightly left still looks like "7"
    MaxPooling2D((2, 2)),
    # Dropout(0.25): during training, randomly zeroes 25% of neurons to prevent
    # co-adaptation and overfitting
    Dropout(0.25),

    # Block 2: deeper feature detection
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Classifier
    # Flatten: converts 2D feature maps to 1D vector for the dense layers
    Flatten(),
    # Dense(512): learns non-linear combinations of the extracted features
    Dense(512, activation='relu'),
    BatchNormalization(),
    # Dropout(0.5): heavier dropout before the final layer — the most common
    # source of overfitting in image classifiers
    Dropout(0.5),
    # Dense(NUM_CLASSES, softmax): outputs a probability distribution over 10 digits
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ── Training ───────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_DIR / 'cnn_best.keras', save_best_only=True, monitor='val_accuracy'),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
print(f"\nBest epoch: {best_epoch}")


# ── Evaluation ─────────────────────────────────────────────────────────────────

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test loss:     {test_loss:.4f}")
print(f"Baseline (random): 10.0% — CNN improvement: {test_acc*100:.1f}%")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_raw, y_pred, target_names=[str(d) for d in range(NUM_CLASSES)]))

cm = confusion_matrix(y_test_raw, y_pred)
print("Confusion Matrix:")
print(cm)


# ── Save model artefacts ───────────────────────────────────────────────────────

history_dict = {
    'loss':         [float(v) for v in history.history['loss']],
    'accuracy':     [float(v) for v in history.history['accuracy']],
    'val_loss':     [float(v) for v in history.history['val_loss']],
    'val_accuracy': [float(v) for v in history.history['val_accuracy']],
}
with open(MODEL_DIR / 'training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

report = classification_report(y_test_raw, y_pred, output_dict=True)
metrics_dict = {
    'test_accuracy': float(test_acc),
    'test_loss':     float(test_loss),
    'best_epoch':    best_epoch,
    'per_class':     {str(d): report[str(d)] for d in range(NUM_CLASSES)},
}
with open(MODEL_DIR / 'model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print("\nSaved training_history.json and model_metrics.json")


# ── Remaining plots ────────────────────────────────────────────────────────────

# 04 — training curves
n_epochs_run = len(history.history['loss'])
early_stop_epoch = best_epoch  # EarlyStopping restores best weights at this epoch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, n_epochs_run + 1)

ax1.plot(epochs_range, history.history['loss'],     label='Train loss')
ax1.plot(epochs_range, history.history['val_loss'], label='Val loss')
ax1.axvline(x=early_stop_epoch, color='red', linestyle='--', alpha=0.7, label='Best epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss')
ax1.legend()

ax2.plot(epochs_range, history.history['accuracy'],     label='Train accuracy')
ax2.plot(epochs_range, history.history['val_accuracy'], label='Val accuracy')
ax2.axvline(x=early_stop_epoch, color='red', linestyle='--', alpha=0.7, label='Best epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')
ax2.legend()

fig.suptitle('CNN Training Curves — MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_training_curves.png")

# 05 — confusion matrix heatmap
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(range(NUM_CLASSES))
ax.set_yticklabels(range(NUM_CLASSES))
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix — Test Set (10,000 images)', fontsize=13, fontweight='bold')
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=7)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_confusion_matrix.png")

# 06 — misclassified examples
# Most misclassifications are genuinely ambiguous — a poorly written 4 that looks
# like a 9, or a 7 without the crossbar that looks like a 1. This is expected and honest.
misclassified_idx = np.where(y_pred != y_test_raw)[0]
fig, axes = plt.subplots(4, 4, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified_idx):
        idx = misclassified_idx[i]
        ax.imshow(X_test_raw[idx], cmap='gray')
        true_label = y_test_raw[idx]
        pred_label = y_pred[idx]
        conf = float(y_pred_probs[idx][pred_label])
        ax.set_title(f'True:{true_label} Pred:{pred_label}\n{conf*100:.0f}%', fontsize=8)
    ax.axis('off')
fig.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_misclassified.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_misclassified.png")


# ── Final summary ──────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Test accuracy:  {test_acc*100:.2f}%")
print(f"Best epoch:     {best_epoch}")
print(f"Model saved to: {MODEL_DIR / 'cnn_best.keras'}")

print("\nMost confused digit pairs (top 5):")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
flat_idx = np.argsort(cm_no_diag.flatten())[::-1][:5]
for idx in flat_idx:
    true_d = idx // NUM_CLASSES
    pred_d = idx %  NUM_CLASSES
    print(f"  True {true_d} → Predicted {pred_d}: {cm_no_diag[true_d, pred_d]} times")

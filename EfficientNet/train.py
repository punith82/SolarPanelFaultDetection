# =============================================================================
# train.py — EfficientNet Training Pipeline for Solar Panel Fault Detection
# =============================================================================
# HOW TO RUN:
#   cd SPFD/EfficientNet/
#   python train.py
# =============================================================================

import os
import random
import warnings
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on Mac, Windows, Linux
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              ConfusionMatrixDisplay)

# Import all settings from our config file
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    OUTPUT_DIR, MODEL_SAVE_PATH, FINAL_MODEL_PATH,
    HISTORY_PLOT_PATH, CONFUSION_MAT_PATH, REPORT_PATH,
    CLASS_NAMES, NUM_CLASSES, INPUT_SHAPE, IMG_SIZE,
    MODEL_VARIANT, BATCH_SIZE, EPOCHS, HEAD_EPOCHS,
    INITIAL_LR, FINETUNE_LR, UNFREEZE_LAYERS,
    DROPOUT_RATE, L2_REG, LABEL_SMOOTHING,
    PATIENCE_EARLY_STOP, PATIENCE_LR_REDUCE,
    LR_REDUCE_FACTOR, MIN_LR,
    AUGMENTATION_PARAMS, USE_CLASS_WEIGHTS,
    MIXED_PRECISION, SEED
)

# ── 0. REPRODUCIBILITY ────────────────────────────────────────────────────────
# Setting the same seed every run makes results reproducible (same random
# numbers → same weight initialisation → easier to debug)
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")

# ── 1. OPTIONAL: MIXED PRECISION ──────────────────────────────────────────────
if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy(MIXED_PRECISION)
    print(f"[INFO] Mixed precision enabled: {MIXED_PRECISION}")

# ── 2. DATA LOADING & AUGMENTATION ────────────────────────────────────────────
# WHY USE IMAGEDATAGENERATOR?
#   It loads images in batches from disk (memory-efficient) and applies
#   random transformations on-the-fly so we never store augmented copies.
#
# WHY preprocess_input?
#   EfficientNet expects pixel values in a specific range.
#   preprocess_input rescales from 0-255 → the range the model was pre-trained on.
#   NOT using it is a very common mistake that causes unstable training!

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("\n[INFO] Setting up data generators...")

# Training generator: augmentation + preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    **AUGMENTATION_PARAMS
)

# Validation & Test generators: ONLY preprocessing, NO augmentation
#   We want to evaluate on "real" images, not artificially modified ones.
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# flow_from_directory reads images from subfolders.
# class_mode="categorical" → one-hot labels like [0,0,1,0,0] for hotspot.
# classes=CLASS_NAMES ensures the order matches our config every single time.
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size    = IMG_SIZE,
    batch_size     = BATCH_SIZE,
    class_mode     = "categorical",
    classes        = CLASS_NAMES,
    shuffle        = True,
    seed           = SEED
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size    = IMG_SIZE,
    batch_size     = BATCH_SIZE,
    class_mode     = "categorical",
    classes        = CLASS_NAMES,
    shuffle        = False   # keep order consistent for evaluation
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size    = IMG_SIZE,
    batch_size     = BATCH_SIZE,
    class_mode     = "categorical",
    classes        = CLASS_NAMES,
    shuffle        = False
)

# Quick sanity check
print(f"\n[INFO] Class → index mapping: {train_gen.class_indices}")
print(f"[INFO] Training samples  : {train_gen.samples}")
print(f"[INFO] Validation samples: {val_gen.samples}")
print(f"[INFO] Test samples      : {test_gen.samples}")

# ── 3. CLASS WEIGHTS (fix imbalanced dataset) ─────────────────────────────────
# WHY CLASS WEIGHTS?
#   If "normal" has 500 images but "birddrop" has 80, the model learns to just
#   always predict "normal" — that's 500/580 = 86% accuracy without learning
#   anything useful. Class weights penalise wrong predictions on rare classes
#   more, forcing the model to pay attention to all classes equally.
#
# THIS IS THE #1 REASON predictions collapse to one class!

class_weights_dict = None

if USE_CLASS_WEIGHTS:
    labels = train_gen.classes   # integer label for every training image
    class_weights_array = compute_class_weight(
        class_weight = "balanced",
        classes      = np.unique(labels),
        y            = labels
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print("\n[INFO] Class weights (higher = rarer class):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"       {name:12s} → {class_weights_dict[i]:.4f}")

# ── 4. BUILD THE MODEL ────────────────────────────────────────────────────────
# TRANSFER LEARNING STRATEGY (2-phase approach):
#
#   Phase 1 — Train only the HEAD (new layers we add on top):
#     The EfficientNet backbone is frozen (weights don't change).
#     We only train our new Dense layers.
#     This is fast and prevents destroying pre-trained features early on.
#
#   Phase 2 — Fine-tune the backbone:
#     We unfreeze the top N layers of EfficientNet so they can adapt
#     to solar panel images specifically (they were trained on ImageNet).
#     Use a very small learning rate so we make gentle adjustments.

from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

print(f"\n[INFO] Building EfficientNet{MODEL_VARIANT} model...")

def build_efficientnet(variant="B0", input_shape=INPUT_SHAPE,
                       num_classes=NUM_CLASSES,
                       dropout_rate=DROPOUT_RATE,
                       l2_reg=L2_REG):
    """
    Build a transfer-learning EfficientNet model for SPFD.

    Architecture:
        EfficientNet backbone (pre-trained on ImageNet, frozen initially)
        → GlobalAveragePooling2D      (reduces spatial dimensions)
        → BatchNormalization          (stabilises training)
        → Dense(256, relu)            (learn fault-specific features)
        → Dropout                     (prevent overfitting)
        → Dense(NUM_CLASSES, softmax) (output probabilities for each fault)
    """
    # Select backbone
    if variant == "B0":
        base = EfficientNetB0(include_top=False,        # remove ImageNet head
                              weights="imagenet",        # use pretrained weights
                              input_shape=input_shape)
    elif variant == "B1":
        base = EfficientNetB1(include_top=False,
                              weights="imagenet",
                              input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported variant: {variant}. Use 'B0' or 'B1'.")

    # Freeze all backbone layers initially
    base.trainable = False
    print(f"[INFO] Backbone frozen — {len(base.layers)} layers")

    # Build the classification head
    reg = regularizers.l2(l2_reg)

    inputs  = tf.keras.Input(shape=input_shape, name="input_image")
    x       = base(inputs, training=False)  # training=False → BN uses stored stats
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.BatchNormalization(name="bn_head")(x)
    x       = layers.Dense(256, activation="relu",
                           kernel_regularizer=reg,
                           name="dense_head")(x)
    x       = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = Model(inputs, outputs,
                  name=f"EfficientNet{variant}_SPFD")
    return model, base

model, base_model = build_efficientnet(MODEL_VARIANT)
model.summary()

# ── 5. COMPILE PHASE 1 (head only) ───────────────────────────────────────────
# CategoricalCrossentropy with label_smoothing:
#   Instead of penalising the model for not being 100% confident, we target
#   0.9 for the correct class — this reduces overconfidence and improves accuracy.

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy

def compile_model(m, lr):
    m.compile(
        optimizer = Adam(learning_rate=lr),
        loss      = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics   = [
            "accuracy",
            TopKCategoricalAccuracy(k=2, name="top2_accuracy")
        ]
    )

compile_model(model, INITIAL_LR)

# ── 6. CALLBACKS ──────────────────────────────────────────────────────────────
# Callbacks are functions that Keras calls automatically at the end of each epoch.

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)

callbacks_phase1 = [
    # Save the model only when validation accuracy improves
    ModelCheckpoint(
        filepath       = MODEL_SAVE_PATH,
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1
    ),
    # Stop early if no improvement for PATIENCE epochs
    EarlyStopping(
        monitor              = "val_accuracy",
        patience             = PATIENCE_EARLY_STOP,
        restore_best_weights = True,   # go back to the best checkpoint
        verbose              = 1
    ),
    # Reduce learning rate when training plateaus
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = LR_REDUCE_FACTOR,
        patience = PATIENCE_LR_REDUCE,
        min_lr   = MIN_LR,
        verbose  = 1
    ),
]

# ── 7. PHASE 1 TRAINING: HEAD ONLY ───────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 1: Training classification head (backbone frozen)")
print(f"{'='*60}")

history1 = model.fit(
    train_gen,
    epochs           = HEAD_EPOCHS,
    validation_data  = val_gen,
    class_weight     = class_weights_dict,
    callbacks        = callbacks_phase1
)

# ── 8. PHASE 2: UNFREEZE & FINE-TUNE ─────────────────────────────────────────
# Now we "unfreeze" the top layers of EfficientNet so they can specialise
# to solar panel features. We use a MUCH smaller learning rate to avoid
# destroying the valuable ImageNet features.

print(f"\n{'='*60}")
print(f"PHASE 2: Fine-tuning top {UNFREEZE_LAYERS} backbone layers")
print(f"{'='*60}")

# Unfreeze the entire base first, then re-freeze from the beginning
base_model.trainable = True

if UNFREEZE_LAYERS > 0:
    # Freeze all layers EXCEPT the last UNFREEZE_LAYERS
    freeze_until = len(base_model.layers) - UNFREEZE_LAYERS
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] {unfrozen} backbone layers now trainable")

# Recompile with a much smaller learning rate
compile_model(model, FINETUNE_LR)

# New set of callbacks for phase 2
callbacks_phase2 = [
    ModelCheckpoint(
        filepath       = MODEL_SAVE_PATH,
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1
    ),
    EarlyStopping(
        monitor              = "val_accuracy",
        patience             = PATIENCE_EARLY_STOP,
        restore_best_weights = True,
        verbose              = 1
    ),
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = LR_REDUCE_FACTOR,
        patience = PATIENCE_LR_REDUCE,
        min_lr   = MIN_LR,
        verbose  = 1
    ),
]

history2 = model.fit(
    train_gen,
    epochs           = EPOCHS,
    initial_epoch    = HEAD_EPOCHS,   # continue from where phase 1 ended
    validation_data  = val_gen,
    class_weight     = class_weights_dict,
    callbacks        = callbacks_phase2
)

# Save the final model regardless (best checkpoint already saved above)
model.save(FINAL_MODEL_PATH)
print(f"\n[INFO] Final model saved → {FINAL_MODEL_PATH}")

# ── 9. MERGE HISTORIES & PLOT TRAINING CURVES ─────────────────────────────────
# We combine the two training phases into one continuous history for plotting.

def merge_histories(h1, h2):
    """Merge two Keras History objects into one dict."""
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_histories(history1, history2)

def plot_training_history(history, save_path):
    """Plot accuracy and loss curves for both train and validation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"EfficientNet{MODEL_VARIANT} — Training History", fontsize=14)

    epochs_range = range(1, len(history["accuracy"]) + 1)
    phase_split  = HEAD_EPOCHS   # vertical line where fine-tuning starts

    # ── Accuracy plot ──
    ax = axes[0]
    ax.plot(epochs_range, history["accuracy"],     label="Train Accuracy",      color="steelblue")
    ax.plot(epochs_range, history["val_accuracy"], label="Val Accuracy",         color="coral")
    ax.axvline(x=phase_split, color="gray", linestyle="--", label="Fine-tuning starts")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Loss plot ──
    ax = axes[1]
    ax.plot(epochs_range, history["loss"],     label="Train Loss",  color="steelblue")
    ax.plot(epochs_range, history["val_loss"], label="Val Loss",    color="coral")
    ax.axvline(x=phase_split, color="gray", linestyle="--", label="Fine-tuning starts")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")

plot_training_history(history, HISTORY_PLOT_PATH)

# ── 10. EVALUATE ON TEST SET ──────────────────────────────────────────────────
print("\n[INFO] Evaluating on test set...")

# Load the BEST checkpoint (not necessarily the last epoch)
from tensorflow.keras.models import load_model
best_model = load_model(MODEL_SAVE_PATH)

test_loss, test_acc, test_top2 = best_model.evaluate(test_gen, verbose=1)
print(f"\n[RESULT] Test Loss     : {test_loss:.4f}")
print(f"[RESULT] Test Accuracy : {test_acc*100:.2f}%")
print(f"[RESULT] Top-2 Accuracy: {test_top2*100:.2f}%")

# ── 11. CONFUSION MATRIX & CLASSIFICATION REPORT ─────────────────────────────
# Confusion matrix: rows = true class, columns = predicted class.
#   Diagonal = correct predictions. Off-diagonal = errors.
#   Helps you spot which classes are confused with each other.

print("\n[INFO] Generating confusion matrix and classification report...")

# Get predictions for all test images
test_gen.reset()   # rewind to beginning (important if shuffle=False)
y_pred_probs = best_model.predict(test_gen, verbose=1)   # shape: (N, 5)
y_pred       = np.argmax(y_pred_probs, axis=1)            # predicted class index
y_true       = test_gen.classes                            # true class index

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
ax.set_title(f"EfficientNet{MODEL_VARIANT} — Confusion Matrix (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(CONFUSION_MAT_PATH, dpi=150)
plt.close()
print(f"[INFO] Confusion matrix saved → {CONFUSION_MAT_PATH}")

# Classification report: precision, recall, F1 per class
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\n[CLASSIFICATION REPORT]\n")
print(report)

# Save to text file for comparison with other models later
with open(REPORT_PATH, "w") as f:
    f.write(f"EfficientNet{MODEL_VARIANT} — Solar Panel Fault Detection\n")
    f.write("="*60 + "\n")
    f.write(f"Test Loss     : {test_loss:.4f}\n")
    f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
    f.write(f"Top-2 Accuracy: {test_top2*100:.2f}%\n\n")
    f.write(report)
print(f"[INFO] Report saved → {REPORT_PATH}")

# ── 12. FINAL SUMMARY ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("TRAINING COMPLETE — Summary")
print(f"{'='*60}")
print(f"  Best model  : {MODEL_SAVE_PATH}")
print(f"  Final model : {FINAL_MODEL_PATH}")
print(f"  History plot: {HISTORY_PLOT_PATH}")
print(f"  Confusion mat: {CONFUSION_MAT_PATH}")
print(f"  Report      : {REPORT_PATH}")
print(f"{'='*60}\n")
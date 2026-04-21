"""
WHY THIS FILE?
This is the main training file.

It:
1. Loads dataset
2. Preprocesses images
3. Loads MobileNetV2 pretrained model
4. Trains model
5. Evaluates model
6. Saves graphs + model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import *

# ---------------------------------------------------
# Create folders if missing
# ---------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

# ---------------------------------------------------
# Image generators
# Train data gets augmentation
# ---------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------------------------------
# Handle class imbalance using class weights
# ---------------------------------------------------
labels = train_data.classes
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights))

# ---------------------------------------------------
# Load pretrained MobileNetV2
# ---------------------------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ---------------------------------------------------
# Compile model
# ---------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------
# Callbacks
# ---------------------------------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# ---------------------------------------------------
# Train model
# ---------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ---------------------------------------------------
# Evaluate model
# ---------------------------------------------------
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc:.4f}")

# ---------------------------------------------------
# Plot Accuracy Graph
# ---------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.savefig(RESULTS_DIR + "accuracy.png")
plt.close()

# ---------------------------------------------------
# Plot Loss Graph
# ---------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Graph")
plt.savefig(RESULTS_DIR + "loss.png")
plt.close()

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
pred_probs = model.predict(test_data)
preds = np.argmax(pred_probs, axis=1)
true_labels = test_data.classes

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.savefig(RESULTS_DIR + "confusion_matrix.png")
plt.close()

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------
report = classification_report(
    true_labels,
    preds,
    target_names=CLASS_NAMES,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR + "classification_report.csv"
)

print("Training Complete ✅")
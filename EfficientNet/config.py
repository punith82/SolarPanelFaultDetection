# =============================================================================
# config.py — EfficientNet Configuration for Solar Panel Fault Detection (SPFD)
# =============================================================================
# WHY A CONFIG FILE?
#   Instead of scattering numbers all over your code, we keep every setting
#   in ONE place. Change something here → it changes everywhere automatically.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1. PATHS  (relative so the project works on any machine / OS)
# -----------------------------------------------------------------------------
# os.path.dirname(__file__) = the folder where THIS file lives (SPFD/EfficientNet/)
# os.path.join goes up two levels (.., ..) to reach the project root, then down
# into faultsdataset/

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # SPFD/EfficientNet/
DATA_DIR   = os.path.join(BASE_DIR, "..", "..", "faultsdataset") # ../../faultsdataset/

TRAIN_DIR = "../faultsdataset/train"
VAL_DIR = "../faultsdataset/val"
TEST_DIR = "../faultsdataset/test"

# Where to save trained model, graphs, and reports
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)   # create folder if it doesn't exist

MODEL_SAVE_PATH     = os.path.join(OUTPUT_DIR, "efficientnet_spfd_best.keras")
FINAL_MODEL_PATH    = os.path.join(OUTPUT_DIR, "efficientnet_spfd_final.keras")
HISTORY_PLOT_PATH   = os.path.join(OUTPUT_DIR, "training_history.png")
CONFUSION_MAT_PATH  = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
REPORT_PATH         = os.path.join(OUTPUT_DIR, "classification_report.txt")

# -----------------------------------------------------------------------------
# 2. CLASSES  (order must match the folder names exactly, alphabetical is safe)
# -----------------------------------------------------------------------------
# WHY DEFINE CLASSES HERE?
#   Keras ImageDataGenerator auto-sorts folder names alphabetically.
#   By listing them explicitly we make sure predict.py uses the SAME order.
CLASS_NAMES = ["birddrop", "dusty", "hotspot", "normal", "physical"]
NUM_CLASSES = len(CLASS_NAMES)   # 5

# -----------------------------------------------------------------------------
# 3. IMAGE SETTINGS
# -----------------------------------------------------------------------------
# EfficientNetB0 was designed for 224×224.
# EfficientNetB1 uses 240×240 — slightly better accuracy, slightly slower.
# Change IMG_SIZE to (240, 240) and MODEL_VARIANT to "B1" to switch.
IMG_SIZE      = (224, 224)   # (height, width) in pixels
IMG_CHANNELS  = 3            # RGB → 3 channels
INPUT_SHAPE   = IMG_SIZE + (IMG_CHANNELS,)  # (224, 224, 3)
MODEL_VARIANT = "B0"         # "B0" or "B1"

# -----------------------------------------------------------------------------
# 4. TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
# BATCH_SIZE: how many images are fed to the model at once during one step.
#   - 16 or 32 works well on laptops / Macs with limited RAM.
#   - Larger batch → faster training, but needs more memory.
BATCH_SIZE = 32

# EPOCHS: maximum number of times we loop over the full training dataset.
#   Early stopping will halt training sooner if the model stops improving.
EPOCHS = 50

# LEARNING RATE for the FINAL fine-tuning phase (when we unfreeze the backbone).
#   Keep this VERY small — large LR destroys pre-trained weights.
FINETUNE_LR = 1e-5   # 0.00001

# INITIAL LR used when only the new classification head is being trained.
INITIAL_LR  = 1e-3   # 0.001

# How many epochs to train the HEAD before unfreezing the backbone.
HEAD_EPOCHS = 10

# How many top layers of EfficientNet to unfreeze for fine-tuning.
#   0   = freeze everything (fastest, less accurate)
#   20  = unfreeze last 20 layers (good balance)
#   -1  = unfreeze ALL layers (slowest, best accuracy but needs lots of data)
UNFREEZE_LAYERS = 20

# Dropout rate: randomly turns off this fraction of neurons during training.
#   Prevents the model from memorising training images (overfitting).
DROPOUT_RATE = 0.4

# L2 regularisation strength — another tool against overfitting.
L2_REG = 1e-4

# Label smoothing: instead of hard 0/1 targets, use 0.1/0.9.
#   Stops the model from being over-confident and improves generalisation.
LABEL_SMOOTHING = 0.1

# -----------------------------------------------------------------------------
# 5. EARLY STOPPING & CALLBACKS
# -----------------------------------------------------------------------------
# Stop training if val_accuracy doesn't improve for this many epochs.
PATIENCE_EARLY_STOP = 10

# Reduce learning rate if val_loss doesn't improve for this many epochs.
PATIENCE_LR_REDUCE  = 4
LR_REDUCE_FACTOR    = 0.3    # new_lr = old_lr * 0.3
MIN_LR              = 1e-7   # never go below this

# -----------------------------------------------------------------------------
# 6. DATA AUGMENTATION SETTINGS
# -----------------------------------------------------------------------------
# Augmentation creates fake variations of training images so the model sees
# more diversity and doesn't overfit to the exact images in the dataset.
AUGMENTATION_PARAMS = dict(
    rotation_range      = 20,     # rotate ± 20 degrees
    width_shift_range   = 0.15,   # shift left/right by up to 15 %
    height_shift_range  = 0.15,   # shift up/down by up to 15 %
    shear_range         = 0.10,   # slight shear distortion
    zoom_range          = 0.15,   # zoom in/out by up to 15 %
    horizontal_flip     = True,   # mirror image left-right
    vertical_flip       = False,  # usually NOT helpful for solar panels
    brightness_range    = [0.8, 1.2],  # vary brightness (dusty vs clear sky)
    fill_mode           = "nearest",   # fill empty pixels after transform
)

# -----------------------------------------------------------------------------
# 7. CLASS WEIGHTS (to handle imbalanced datasets)
# -----------------------------------------------------------------------------
# If some classes have fewer images the model tends to ignore them and just
# predict the most common class → this is why you see "predict only normal".
# Setting USE_CLASS_WEIGHTS=True makes the model pay MORE attention to rare classes.
USE_CLASS_WEIGHTS = True

# -----------------------------------------------------------------------------
# 8. MIXED PRECISION (optional speed boost on supported GPUs / Apple Silicon)
# -----------------------------------------------------------------------------
# "mixed_float16" uses 16-bit floats where possible → ~2× faster on modern GPUs.
# Set to None to disable if you get errors.
MIXED_PRECISION = None   # set to "mixed_float16" if you have a modern GPU

# -----------------------------------------------------------------------------
# 9. RANDOM SEED (for reproducibility)
# -----------------------------------------------------------------------------
SEED = 42
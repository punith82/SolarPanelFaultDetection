"""
WHY THIS FILE?
Stores all settings in one place:
- dataset paths
- image size
- epochs
- class names
- save paths
"""

TRAIN_DIR = "../faultsdataset/train"
VAL_DIR = "../faultsdataset/val"
TEST_DIR = "../faultsdataset/test"

CLASS_NAMES = ["birddrop", "dusty", "hotspot", "normal", "physical"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

MODEL_PATH = "saved_model/mobilenetv2_best.keras"
RESULTS_DIR = "results/"
LOGS_DIR = "logs/"
SEED = 42
# =============================================================================
# predict.py — EfficientNet Inference for Solar Panel Fault Detection (SPFD)
# =============================================================================
# HOW TO RUN:
#
#   # Predict a single image:
#   python predict.py --image path/to/panel.jpg
#
#   # Predict an entire folder:
#   python predict.py --folder path/to/test_images/
#
#   # Predict using the final model instead of best checkpoint:
#   python predict.py --image panel.jpg --model final
#
#   # Show the image with prediction overlay:
#   python predict.py --image panel.jpg --show
# =============================================================================

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # safe for servers; remove if you want interactive display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import (
    MODEL_SAVE_PATH, FINAL_MODEL_PATH,
    IMG_SIZE, CLASS_NAMES, NUM_CLASSES, OUTPUT_DIR
)

# ── 1. ARGUMENT PARSER ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="EfficientNet Solar Panel Fault Detector"
)
parser.add_argument("--image",  type=str, default=None,
                    help="Path to a single image file")
parser.add_argument("--folder", type=str, default=None,
                    help="Path to a folder of images")
parser.add_argument("--model",  type=str, default="best",
                    choices=["best", "final"],
                    help="Which saved model to use (default: best checkpoint)")
parser.add_argument("--show",   action="store_true",
                    help="Save prediction visualisation images")
parser.add_argument("--batch",  type=int, default=16,
                    help="Batch size for folder prediction (default: 16)")
args = parser.parse_args()

if args.image is None and args.folder is None:
    # Default: run on the test directory so you can quickly check the model
    print("[INFO] No --image or --folder specified. "
          "Running on test directory for a quick check.\n")
    args.folder = os.path.join(os.path.dirname(__file__),
                               "..", "..", "faultsdataset", "test")

# ── 2. LOAD MODEL ─────────────────────────────────────────────────────────────
# WHY IS THIS IMPORTANT?
#   A common mistake is loading the model but forgetting to use the SAME
#   preprocessing as during training. We always call preprocess_input() here.

model_path = MODEL_SAVE_PATH if args.model == "best" else FINAL_MODEL_PATH
print(f"[INFO] Loading model from: {model_path}")

if not os.path.exists(model_path):
    print(f"\n[ERROR] Model file not found: {model_path}")
    print("        Have you run train.py yet?")
    sys.exit(1)

model = load_model(model_path)
print(f"[INFO] Model loaded successfully.")
print(f"[INFO] Classes: {CLASS_NAMES}")

# ── 3. IMAGE LOADING & PREPROCESSING ──────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load one image from disk and prepare it for the model.

    Steps:
        1. Read file → decode as JPEG/PNG
        2. Resize to 224×224 (or whatever IMG_SIZE is)
        3. Add a batch dimension  → shape (1, 224, 224, 3)
        4. Apply EfficientNet's preprocess_input
           (THIS IS THE KEY STEP many people miss — it rescales pixels
            from [0,255] to [-1, 1] range, which is what the model expects)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {ext}")

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)            # (224, 224, 3)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)               # (1, 224, 224, 3)
    img = preprocess_input(img)                     # normalise for EfficientNet
    return img.numpy()

# ── 4. SINGLE IMAGE PREDICTION ────────────────────────────────────────────────
def predict_single(image_path: str, show: bool = False) -> dict:
    """
    Predict the fault class of one solar panel image.

    Returns a dict with:
        predicted_class  — name of the fault (e.g. "dusty")
        confidence       — probability of that class (0.0 – 1.0)
        all_probs        — dict of {class_name: probability} for all 5 classes
        image_path       — original file path
    """
    img_array = load_and_preprocess(image_path)

    # model.predict returns shape (1, 5) — probabilities for each class
    probs = model.predict(img_array, verbose=0)[0]   # shape: (5,)

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    result = dict(
        image_path      = image_path,
        predicted_class = pred_class,
        confidence      = confidence,
        all_probs       = all_probs
    )

    # ── Pretty print ──
    print(f"\n{'─'*50}")
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Prediction: {pred_class.upper()} ({confidence*100:.1f}% confident)")
    print(f"  All class probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar   = "█" * int(prob * 30)
        flag  = " ← TOP" if cls == pred_class else ""
        print(f"    {cls:12s}  {bar:<30}  {prob*100:5.1f}%{flag}")
    print(f"{'─'*50}")

    # ── Optional: save visualisation ──
    if show:
        _save_prediction_image(image_path, result)

    return result


def _save_prediction_image(image_path: str, result: dict):
    """Save a figure showing the image + bar chart of probabilities."""
    img_raw = tf.io.read_file(image_path)
    img_raw = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img_raw = tf.image.resize(img_raw, IMG_SIZE).numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"EfficientNet Prediction — {result['predicted_class'].upper()} "
        f"({result['confidence']*100:.1f}%)",
        fontsize=14, fontweight="bold"
    )

    # Left: image
    axes[0].imshow(img_raw)
    axes[0].axis("off")
    axes[0].set_title(os.path.basename(image_path), fontsize=10)

    # Right: probability bar chart
    classes = list(result["all_probs"].keys())
    probs   = [result["all_probs"][c] for c in classes]
    colors  = ["#2196F3" if c != result["predicted_class"] else "#FF5722"
               for c in classes]

    bars = axes[1].barh(classes, probs, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Class Probabilities")
    axes[1].invert_yaxis()

    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        axes[1].text(min(prob + 0.01, 0.95), bar.get_y() + bar.get_height()/2,
                     f"{prob*100:.1f}%", va="center", fontsize=9)

    red_patch  = mpatches.Patch(color="#FF5722", label="Predicted class")
    blue_patch = mpatches.Patch(color="#2196F3", label="Other classes")
    axes[1].legend(handles=[red_patch, blue_patch], loc="lower right")

    plt.tight_layout()

    # Save next to the model outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path  = os.path.join(OUTPUT_DIR, f"pred_{base_name}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Visualisation saved → {out_path}")


# ── 5. BATCH / FOLDER PREDICTION ─────────────────────────────────────────────
def predict_folder(folder_path: str, batch_size: int = 16) -> list:
    """
    Predict all images in a folder (supports nested subfolders).

    Returns a list of result dicts (same format as predict_single).
    Also prints a summary table and per-class statistics.
    """
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        sys.exit(1)

    # Collect all valid image paths recursively
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"[ERROR] No supported images found in {folder_path}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(image_paths)} images in {folder_path}")
    print("[INFO] Running batch prediction...\n")

    results = []

    # Process in batches for efficiency
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        # Stack preprocessed images into one array
        batch_arrays = []
        valid_paths  = []
        for p in batch_paths:
            try:
                arr = load_and_preprocess(p)
                batch_arrays.append(arr)
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

        if not batch_arrays:
            continue

        batch_input = np.vstack(batch_arrays)           # (B, 224, 224, 3)
        batch_probs = model.predict(batch_input, verbose=0)  # (B, 5)

        for i, path in enumerate(valid_paths):
            probs      = batch_probs[i]
            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

            results.append(dict(
                image_path      = path,
                predicted_class = pred_class,
                confidence      = confidence,
                all_probs       = {CLASS_NAMES[j]: float(probs[j])
                                   for j in range(NUM_CLASSES)}
            ))

        done = min(batch_start + batch_size, len(image_paths))
        print(f"  Processed {done}/{len(image_paths)} images...", end="\r")

    print()  # newline after progress

    # ── Print summary table ──
    print(f"\n{'='*65}")
    print(f"{'FILE':<30} {'PREDICTION':<12} {'CONFIDENCE':>10}")
    print(f"{'='*65}")
    for r in results:
        fname = os.path.basename(r["image_path"])[:29]
        print(f"{fname:<30} {r['predicted_class']:<12} {r['confidence']*100:>9.1f}%")

    # ── Per-class count summary ──
    from collections import Counter
    counts = Counter(r["predicted_class"] for r in results)
    avg_conf = {c: [] for c in CLASS_NAMES}
    for r in results:
        avg_conf[r["predicted_class"]].append(r["confidence"])

    print(f"\n{'='*45}")
    print(f"PREDICTION SUMMARY — {len(results)} images total")
    print(f"{'='*45}")
    for cls in CLASS_NAMES:
        n    = counts.get(cls, 0)
        mean = np.mean(avg_conf[cls]) * 100 if avg_conf[cls] else 0
        print(f"  {cls:12s} : {n:4d} images | avg confidence {mean:.1f}%")

    # ── Diagnosis: Warn if one class dominates ──
    dominant_cls   = counts.most_common(1)[0][0] if counts else None
    dominant_frac  = counts[dominant_cls] / len(results) if counts else 0

    if dominant_frac > 0.70:
        print(f"\n[WARNING] {dominant_frac*100:.0f}% of predictions are '{dominant_cls}'.")
        print("          This may indicate a class imbalance problem or that the")
        print("          model needs more fine-tuning. Consider:")
        print("          1. Checking class weights in config.py (USE_CLASS_WEIGHTS=True)")
        print("          2. Increasing UNFREEZE_LAYERS and retraining")
        print("          3. Collecting more data for underrepresented classes")

    return results


# ── 6. MAIN ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":

    if args.image:
        # ── Single image mode ──
        predict_single(args.image, show=args.show)

    elif args.folder:
        # ── Folder mode ──
        results = predict_folder(args.folder, batch_size=args.batch)

        if args.show:
            print("\n[INFO] Saving visualisations for all images...")
            for r in results:
                _save_prediction_image(r["image_path"], r)
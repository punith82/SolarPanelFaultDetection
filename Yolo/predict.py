"""
Inference script for panel detection using trained YOLOv8 model.
"""

from ultralytics import YOLO
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = ROOT_DIR / 'runs' / 'detect' / 'runs' / 'panel_detection' / 'yolov8n_panels' / 'weights' / 'best.pt'
DEFAULT_PREDICTION_PROJECT = ROOT_DIR / 'runs' / 'panel_detection'

def print_results(results, sources=None):
    """Print a readable summary of YOLO detection results."""
    for i, r in enumerate(results):
        source = sources[i] if sources is not None else None
        if source:
            print(f"\nPredictions for: {source}")

        print(f"  Result {i}: {len(r.boxes)} detections")
        for j, box in enumerate(r.boxes):
            xyxy = box.xyxy.tolist()[0]
            conf = float(box.conf.tolist()[0])
            cls = int(box.cls.tolist()[0])
            print(f"    Box {j}: class={cls}, conf={conf:.4f}, xyxy={xyxy}")


def predict_panels(model_path, image_path, conf_threshold=0.5):
    """
    Run panel detection on a single image.

    Args:
        model_path: Path to trained YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
    """

    model = YOLO(str(model_path))

    sources = [str(image_path)]
    results = model.predict(
        source=sources,
        conf=conf_threshold,
        save=True,
        project=str(DEFAULT_PREDICTION_PROJECT),
        name='predictions',
        exist_ok=True
    )

    print_results(results, sources=sources)
    print(f"\n✅ Annotated output saved to: {DEFAULT_PREDICTION_PROJECT / 'predictions'}")
    print(f"  - Total images tested: {len(sources)}")
    print(f"  - Self-annotating: yes (saved image with bounding boxes)")

    return results

def batch_predict(model_path, images_dir, conf_threshold=0.5):
    """
    Run panel detection on all images in a directory.
    """

    model = YOLO(str(model_path))

    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(Path(images_dir).glob(ext))

    print(f"Processing {len(image_files)} images...")

    sources = [str(p) for p in image_files]
    results = model.predict(
        source=sources,
        conf=conf_threshold,
        save=True,
        project=str(DEFAULT_PREDICTION_PROJECT),
        name='batch_predictions',
        exist_ok=True,
        verbose=False
    )

    print_results(results, sources=sources)
    print(f"✅ Predictions saved to: {DEFAULT_PREDICTION_PROJECT / 'batch_predictions'}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO panel detection test runner')
    parser.add_argument('--image', type=str, help='Path to a single image to test')
    parser.add_argument('--dir', type=str, help='Path to a directory of images for batch testing')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH), help='Path to the trained YOLO model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        predict_panels(model_path, image_path, conf_threshold=args.conf)
    else:
        images_dir = Path(args.dir) if args.dir else SCRIPT_DIR / 'YoloData' / 'images' / 'val'
        if not images_dir.exists():
            raise FileNotFoundError(f"Image directory not found at {images_dir}")
        batch_predict(model_path, images_dir, conf_threshold=args.conf)

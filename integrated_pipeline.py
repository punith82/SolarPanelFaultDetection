#!/usr/bin/env python3
"""Integrated YOLOv8 + EfficientNet pipeline for solar panel fault monitoring.

This script loads a trained YOLOv8 panel detector and a trained EfficientNet
fault classifier, then processes one image or a folder of images.

Usage:
    python integrated_pipeline.py --image /path/to/image.jpg
    python integrated_pipeline.py --dir /path/to/folder
    python integrated_pipeline.py --model final

Outputs:
    - Annotated images with panel boxes and fault labels
    - Console report of panel IDs, bounding boxes, and classification results
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Ensure the repo root is importable so we can read EfficientNet config.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from EfficientNet.config import (
    MODEL_SAVE_PATH,
    FINAL_MODEL_PATH,
    IMG_SIZE,
    CLASS_NAMES,
)

from ultralytics import YOLO

DEFAULT_YOLO_MODEL = (
    ROOT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "panel_detection"
    / "yolov8n_panels"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "runs"
    / "panel_detection"
    / "fault_reports"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run integrated panel fault detection with YOLOv8 + EfficientNet"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single top-view panel image",
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to a folder of images to process",
    )

    parser.add_argument(
        "--yolo",
        type=str,
        default=str(DEFAULT_YOLO_MODEL),
        help="Path to YOLOv8 weights file",
    )

    parser.add_argument(
        "--eff_model",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which EfficientNet model to use",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save annotated output images",
    )

    parser.add_argument(
        "--save_crops",
        action="store_true",
        help="Save each cropped panel image to disk",
    )

    parser.add_argument(
        "--crop_dir",
        type=str,
        default=None,
        help="Directory to save cropped panel images (default: output/crops)",
    )

    return parser.parse_args()


def load_efficientnet(model_name: str):
    model_path = MODEL_SAVE_PATH if model_name == "best" else FINAL_MODEL_PATH

    print(f"[INFO] Loading EfficientNet model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"EfficientNet model not found: {model_path}"
        )

    model = load_model(model_path)

    print("[INFO] EfficientNet model loaded successfully.")

    return model


def load_and_preprocess_panel(panel_bgr: np.ndarray) -> np.ndarray:
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    panel_resized = cv2.resize(panel_rgb, tuple(IMG_SIZE[::-1]))
    panel_float = panel_resized.astype(np.float32)
    panel_preprocessed = preprocess_input(panel_float)

    return np.expand_dims(panel_preprocessed, axis=0)


def xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int):
    return x1, y1, x2 - x1, y2 - y1


def build_panel_list(boxes, image_shape):
    panels = []

    height = image_shape[0]
    row_height = max(20, height / 10)

    for box in boxes:
        coords = box.xyxy.tolist()[0]

        x1, y1, x2, y2 = [
            int(max(0, round(v)))
            for v in coords
        ]

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        panels.append(
            {
                "box": box,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "xc": xc,
                "yc": yc,
                "row": int(yc // row_height),
            }
        )

    panels.sort(key=lambda p: (p["row"], p["xc"]))

    for idx, panel in enumerate(panels, start=1):
        panel["panel_id"] = f"P{idx}"

    return panels


def annotate_image(
    image: np.ndarray,
    detections: list,
    output_path: Path,
):
    annotated = image.copy()

    for detection in detections:
        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        label = (
            f"{detection['panel_id']}: "
            f"{detection['predicted_class']} "
            f"({detection['confidence'] * 100:.1f}%)"
        )

        color = (
            (0, 200, 0)
            if detection["predicted_class"] == "normal"
            else (0, 120, 255)
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text_y = max(y1 - 8, 12)

        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated)

    print(f"[INFO] Annotated image saved to: {output_path}")


def process_top_view_image(
    image_path: Path,
    yolo_model,
    eff_model,
    output_dir: Path,
    save_crops: bool = False,
    crop_dir: Path = None,
    conf_threshold: float = 0.5,
):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    crop_dir = crop_dir or output_dir / "crops"

    if save_crops and crop_dir.exists():
        shutil.rmtree(str(crop_dir))
        print(f"[INFO] Cleared previous crops from: {crop_dir}")

    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = yolo_model.predict(
        source=str(image_path),
        conf=conf_threshold,
        save=False,
        verbose=False,
    )

    if not results:
        raise RuntimeError("YOLO did not return any results.")

    yolo_result = results[0]
    boxes = yolo_result.boxes

    num_panels = len(boxes)

    print(f"\n[INFO] {image_path.name}: detected {num_panels} panel(s)")

    if num_panels == 0:
        return {
            "detections": [],
            "annotated_image": None,
        }

    panels = build_panel_list(boxes, image.shape)

    detections = []

    for panel in panels:
        x1 = panel["x1"]
        y1 = panel["y1"]
        x2 = panel["x2"]
        y2 = panel["y2"]

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            print(
                f'[WARN] Skipping empty crop for {panel["panel_id"]}'
            )
            continue

        if save_crops:
            crop_path = (
                crop_dir
                / f'{image_path.stem}_{panel["panel_id"]}.png'
            )

            cv2.imwrite(str(crop_path), crop)

        panel_input = load_and_preprocess_panel(crop)

        probs = eff_model.predict(
            panel_input,
            verbose=0,
        )[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        panel["predicted_class"] = pred_class
        panel["confidence"] = confidence
        panel["xywh"] = xyxy_to_xywh(x1, y1, x2, y2)

        panel["all_probs"] = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        detections.append(panel)

        print(
            f"  {panel['panel_id']}: "
            f"{pred_class.upper()} "
            f"({confidence * 100:.1f}%) | "
            f"xywh={panel['xywh']}"
        )

    output_image_path = (
        output_dir
        / f"{image_path.stem}_{uuid4().hex}_faults.png"
    )

    annotate_image(
        image,
        detections,
        output_image_path,
    )

    return {
        "detections": detections,
        "annotated_image": output_image_path,
    }


def collect_images(folder_path: Path):
    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    image_paths = [
        p
        for p in sorted(folder_path.rglob("*"))
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {folder_path}"
        )

    return image_paths


def main():
    args = parse_args()

    yolo_path = Path(args.yolo)

    if not yolo_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {yolo_path}"
        )

    eff_model = load_efficientnet(args.eff_model)
    yolo_model = YOLO(str(yolo_path))

    output_dir = Path(args.output)
    crop_dir = Path(args.crop_dir) if args.crop_dir else None

    if args.image:
        image_paths = [Path(args.image)]
    elif args.dir:
        image_paths = collect_images(Path(args.dir))
    else:
        raise ValueError("Please specify --image or --dir.")

    print(f"[INFO] Processing {len(image_paths)} image(s)")
    print(f"[INFO] Saving annotated outputs to: {output_dir}")

    if args.save_crops:
        print(
            f'[INFO] Saving cropped panels to: '
            f'{crop_dir or output_dir / "crops"}'
        )

    all_results = []

    for image_path in image_paths:
        result = process_top_view_image(
            image_path=image_path,
            yolo_model=yolo_model,
            eff_model=eff_model,
            output_dir=output_dir,
            save_crops=args.save_crops,
            crop_dir=crop_dir,
            conf_threshold=args.conf,
        )

        all_results.append(
            {
                "image": str(image_path),
                "detections": result["detections"],
                "annotated_image": str(result["annotated_image"])
                if result["annotated_image"]
                else None,
            }
        )

    print("\n[INFO] Completed integrated pipeline")
    print(f"[INFO] Processed {len(all_results)} image(s) total")


if __name__ == "__main__":
    main()
"""
Auto-generate YOLO labels for solar panel detection from images.
This script uses image processing to detect panels and create YOLO format annotations.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def detect_panels(image_path):
    """
    Detect solar panels in an image using contour detection.
    Returns list of bounding boxes in normalized YOLO format: [x_center, y_center, width, height]
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio and size (solar panels are typically rectangular)
        if w > 20 and h > 20:  # Minimum size
            aspect_ratio = w / h if h > 0 else 0
            
            # Solar panels are typically wider than tall or square-ish
            if 0.5 < aspect_ratio < 2.5:
                # Normalize to YOLO format [x_center, y_center, width, height] in [0, 1]
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                norm_w = w / width
                norm_h = h / height
                
                bboxes.append([x_center, y_center, norm_w, norm_h])
    
    return bboxes

def create_yolo_labels(images_dir, labels_dir):
    """
    Create YOLO format labels for all images in the directory.
    """
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        image_files.extend(Path(images_dir).glob(ext))
    
    print(f"Found {len(image_files)} images to process...")
    
    for image_path in tqdm(image_files, desc="Creating labels"):
        bboxes = detect_panels(str(image_path))
        
        # Create corresponding label file
        label_path = Path(labels_dir) / (image_path.stem + '.txt')
        
        if bboxes:
            with open(label_path, 'w') as f:
                # Class 0 = panel
                for bbox in bboxes:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        else:
            # Create empty file if no panels detected
            with open(label_path, 'w') as f:
                pass
        
        print(f"  {label_path.name}: {len(bboxes)} panels detected")

if __name__ == "__main__":
    base_path = "/Users/punithkumarr/Desktop/SPFD1/Yolo/YoloData"
    
    # Process train images
    print("\n=== Processing TRAIN images ===")
    create_yolo_labels(
        f"{base_path}/images/train",
        f"{base_path}/labels/train"
    )
    
    # Process val images
    print("\n=== Processing VAL images ===")
    create_yolo_labels(
        f"{base_path}/images/val",
        f"{base_path}/labels/val"
    )
    
    print("\n✅ Label generation complete!")

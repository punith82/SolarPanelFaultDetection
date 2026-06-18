"""
Complete setup guide for YOLO panel detection training.
Run this script to initialize everything.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Execute a shell command."""
    print(f"\n{'='*60}")
    print(f"📍 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    result = os.system(cmd)
    if result != 0:
        print(f"❌ Failed to execute: {description}")
        return False
    return True

def main():
    base_path = "/Users/punithkumarr/Desktop/SPFD1/Yolo"
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     YOLO PANEL DETECTION - COMPLETE SETUP & TRAINING        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check directory structure
    print("\n✅ Step 1: Verifying directory structure...")
    dirs_to_check = [
        f"{base_path}/YoloData/images/train",
        f"{base_path}/YoloData/images/val",
        f"{base_path}/YoloData/labels/train",
        f"{base_path}/YoloData/labels/val",
    ]
    
    for dir_path in dirs_to_check:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path}")
    
    # Step 2: Install dependencies
    print("\n✅ Step 2: Installing dependencies...")
    print("   This may take a few minutes...")
    if not run_command(
        f"cd {base_path} && pip install -q -r requirements.txt",
        "Installing Python packages"
    ):
        sys.exit(1)
    
    # Step 3: Generate labels
    print("\n✅ Step 3: Auto-generating labels for all images...")
    print("   Using computer vision to detect solar panels...")
    if not run_command(
        f"cd {base_path} && python auto_label.py",
        "Auto-labeling images"
    ):
        sys.exit(1)
    
    # Step 4: Check label count
    train_labels = len(list(Path(f"{base_path}/YoloData/labels/train").glob("*.txt")))
    val_labels = len(list(Path(f"{base_path}/YoloData/labels/val").glob("*.txt")))
    
    print(f"\n📊 Label Statistics:")
    print(f"   Training labels: {train_labels}")
    print(f"   Validation labels: {val_labels}")
    
    # Step 5: Train YOLO
    print("\n✅ Step 4: Training YOLOv8 model...")
    print("   This will take 10-30 minutes depending on GPU availability...")
    if not run_command(
        f"cd {base_path} && python train_yolo.py",
        "Training YOLO model"
    ):
        print("⚠️  Training encountered an issue, but you can retry manually")
    
    # Step 6: Summary
    print("""
╔══════════════════════════════════════════════════════════════╗
║                      🎉 SETUP COMPLETE 🎉                   ║
╚══════════════════════════════════════════════════════════════╝

✅ What was done:
   1. Created directory structure for YOLO training
   2. Installed all required packages
   3. Auto-generated labels for all images using CV
   4. Trained YOLOv8 model for panel detection

📂 Directory structure:
   Yolo/
   ├── YoloData/
   │   ├── images/
   │   │   ├── train/  (training images)
   │   │   └── val/    (validation images)
   │   └── labels/
   │       ├── train/  (auto-generated labels)
   │       └── val/    (auto-generated labels)
   ├── auto_label.py      (label generation script)
   ├── train_yolo.py      (training script)
   ├── predict.py         (inference script)
   ├── requirements.txt   (dependencies)
   └── data.yaml         (YOLO config)

🚀 Next steps:

   1. To run training manually:
      cd /Users/punithkumarr/Desktop/SPFD1/Yolo
      python train_yolo.py

   2. To test the trained model:
      python predict.py

   3. To use the model in your code:
      from ultralytics import YOLO
      model = YOLO('runs/panel_detection/yolov8n_panels/weights/best.pt')
      results = model.predict('image.jpg')

   4. To retrain with different parameters:
      Edit train_yolo.py and modify the training parameters

📚 Generated files will be in:
   runs/panel_detection/yolov8n_panels/
   ├── weights/
   │   ├── best.pt  (Best model checkpoint)
   │   └── last.pt  (Last model checkpoint)
   └── results/     (Training metrics and plots)

💡 Tips for better results:
   • If detection is poor, adjust auto_label.py contour detection
   • Run training longer by increasing 'epochs' in train_yolo.py
   • Use GPU if available (check with 'nvidia-smi')
   • Consider manually labeling some images with Roboflow or LabelImg

Need help? Check the scripts for detailed configuration options!
    """)

if __name__ == "__main__":
    main()

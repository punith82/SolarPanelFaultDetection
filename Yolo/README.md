# YOLO Panel Detection - Complete Guide

## Overview

This project sets up an automated pipeline for training a YOLOv8 model to detect solar panels in images. It includes automatic label generation and a complete training workflow.

## 📁 Project Structure

```
Yolo/
├── YoloData/
│   ├── images/
│   │   ├── train/     (your training images)
│   │   └── val/       (your validation images)
│   └── labels/
│       ├── train/     (auto-generated labels)
│       └── val/       (auto-generated labels)
├── auto_label.py      (Generates YOLO labels from images)
├── train_yolo.py      (Trains the YOLOv8 model)
├── predict.py         (Runs inference on images)
├── setup.py           (Complete setup script)
├── requirements.txt   (Python dependencies)
├── data.yaml          (YOLO dataset config)
└── README.md          (this file)
```

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
cd /Users/punithkumarr/Desktop/SPFD1/Yolo
python setup.py
```

This will:

1. Set up directories
2. Install all dependencies
3. Generate labels for all images
4. Train the YOLO model

### Option 2: Manual Steps

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Generate Labels

```bash
python auto_label.py
```

This uses computer vision to automatically detect panels and create YOLO format labels.

#### 3. Train the Model

```bash
python train_yolo.py
```

Training will take 10-30 minutes depending on GPU availability.

#### 4. Test the Model

```bash
python predict.py
```

## 📊 Data Format

### YOLO Label Format

Each image gets a corresponding `.txt` file with bounding boxes:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:

- `class_id`: 0 = panel
- Coordinates are normalized to [0, 1]

Example:

```
0 0.5 0.5 0.3 0.4
0 0.7 0.6 0.25 0.35
```

## 🎯 How Panel Detection Works

### Auto-Labeling Process

1. **Load Image** → Convert to grayscale
2. **Apply Filters** → Gaussian blur for smoothing
3. **Edge Detection** → Find panel edges using Canny
4. **Contour Detection** → Identify rectangular shapes
5. **Filter by Aspect Ratio** → Remove non-panel objects
6. **Normalize Coordinates** → Convert to YOLO format

### Configuration

In `auto_label.py`, you can adjust:

- `Canny edges`: 50, 150 thresholds
- `Min size`: 20 pixels (minimum panel size)
- `Aspect ratio`: 0.5-2.5 (panel width/height ratio)

## 🧠 Model Training

### Model Used: YOLOv8 Nano

- **Nano**: Fastest, smallest (lightweight for deployment)
- **Small**: Balanced speed and accuracy
- **Medium**: Better accuracy, slower
- **Large**: Best accuracy, slowest

Change in `train_yolo.py`:

```python
model = YOLO('yolov8n.pt')  # Change 'n' to 's', 'm', or 'l'
```

### Training Parameters (in `train_yolo.py`)

- **epochs**: 100 - Number of training cycles
- **batch**: 16 - Images per batch
- **imgsz**: 640 - Input image size
- **patience**: 20 - Early stopping patience
- **device**: 0 - GPU device (0 for first GPU, -1 for CPU)

## 💾 Inference / Predictions

### Single Image

```python
from ultralytics import YOLO

model = YOLO('runs/panel_detection/yolov8n_panels/weights/best.pt')
results = model.predict('path/to/image.jpg', conf=0.5)

# Visualize
for r in results:
    r.show()
```

### Batch Prediction

```bash
python predict.py
```

## 📈 Monitor Training

Training metrics and plots are saved in:

```
runs/panel_detection/yolov8n_panels/
```

Including:

- `results.csv` - Training metrics
- `confusion_matrix.png` - Confusion matrix
- Various accuracy plots

## 🔧 Troubleshooting

### Problem: Poor Detection Accuracy

**Solution 1**: Manually label some images using:

- [Roboflow](https://roboflow.com) (easiest)
- [LabelImg](https://github.com/heartexlabs/labelImg)
- [CVAT](https://cvat.org)

**Solution 2**: Adjust auto-labeling thresholds in `auto_label.py`

```python
# More aggressive edge detection
edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds

# Adjust aspect ratio to be more/less strict
if 0.3 < aspect_ratio < 3.0:  # Wider range
```

### Problem: Training is slow

- Use GPU: `device=0` in `train_yolo.py`
- Use smaller model: `YOLO('yolov8n.pt')`
- Reduce image size: `imgsz=416` (smaller = faster)
- Use fewer epochs: `epochs=50`

### Problem: CUDA/GPU not detected

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Not enough memory

- Reduce batch size: `batch=8` (in train_yolo.py)
- Reduce image size: `imgsz=416`
- Use CPU: `device=-1`

## 📚 Dataset Split

- **Train**: ~80% (training images)
- **Val**: ~20% (validation images)

Ensure validation images are placed in `YoloData/images/val/`

## 🎓 Learn More

- [YOLOv8 Docs](https://docs.ultralytics.com)
- [YOLO Format](https://docs.ultralytics.com/datasets/detect/#coco-dataset)
- [Computer Vision Basics](https://opencv.org)

## 📝 Next Steps

1. Run `python setup.py` to start training
2. Monitor training progress in `runs/` directory
3. Test predictions with `python predict.py`
4. Integrate model into your application
5. Fine-tune with manual labels if needed

## 🆘 Support

For issues:

1. Check the log files in `runs/panel_detection/`
2. Review training plots for convergence issues
3. Adjust parameters in config files
4. Consider using manual labeling for better accuracy

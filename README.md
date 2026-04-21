# ☀️ Solar Panel Fault Detection (SPFD)

An AI-powered deep learning project to automatically detect and classify solar panel faults using image data.
This project compares three state-of-the-art models:

* MobileNetV2
* EfficientNet
* Vision Transformer (ViT)

The system classifies solar panel images into 5 categories:

* birddrop
* dusty
* hotspot
* normal
* physical

---

# 📌 Project Objective

Manual inspection of solar panels is slow, costly, and inefficient.
This project uses Computer Vision + Deep Learning to identify faults automatically, improving maintenance speed and energy efficiency.

---

# 🧠 Models Used

## 1. MobileNetV2

Lightweight CNN optimized for mobile / embedded systems.

## 2. EfficientNet

High-performance CNN with compound scaling.

## 3. Vision Transformer

Attention-based modern architecture for image classification.

---

# 📊 Final Results

| Model             | Accuracy | Macro F1 | Best Use     |
| ----------------- | -------- | -------- | ------------ |
| EfficientNet      | 85.86%   | 87.39%   | Best overall |
| MobileNetV2       | 83.30%   | 84.35%   | Edge devices |
| VisionTransformer | 82.16%   | 84.79%   | Research     |

---

# 📁 Project Structure

```text
SPFD/
├── faultsdataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── MobileNetV2/
├── EfficientNet/
├── VisionTransformer/
├── comparison/
├── README.md
└── requirements.txt
```

---

# 🗂 Dataset Structure

Inside each folder (`train`, `val`, `test`):

```text
birddrop/
dusty/
hotspot/
normal/
physical/
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git lfs install
git clone <your-repo-url>
cd SPFD
```

---

## 2️⃣ Create Virtual Environment

## 💻 Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 🍎 Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

## Root (optional)

```bash
pip install -r requirements.txt
```

## OR Model Specific

### MobileNetV2

```bash
cd MobileNetV2
pip install -r requirements.txt
```

### EfficientNet

```bash
cd EfficientNet
pip install -r requirements.txt
```

### VisionTransformer

```bash
cd VisionTransformer
pip install -r requirements.txt
```

---

# 🚀 How to Train Models

## MobileNetV2

```bash
cd MobileNetV2
python train.py
```

## EfficientNet

```bash
cd EfficientNet
python train.py
```

## VisionTransformer

```bash
cd VisionTransformer
python train.py
```

---

# 🔍 How to Predict Single Image

## Example

```bash
python predict.py --image "/full/path/image.jpg"
```

---

# 📈 Comparison of All Models

Run:

```bash
cd comparison
python compare_results.py
```

Generated files:

* final_report.csv
* comparison_chart.png
* per_class_f1_heatmap.png
* radar_chart.png
* conclusion.txt

---

# 💾 Output Files

Each model folder contains:

## saved_model/

Trained model files.

## results/

* accuracy graph
* loss graph
* confusion matrix
* classification report

## logs/

Training logs.

---

# 🛠 Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Git / Git LFS

---

# 🎯 Real-World Applications

* Solar farms
* Rooftop solar monitoring
* Drone-based panel inspection
* Predictive maintenance
* Smart energy systems

---

# 🔮 Future Improvements

* Real-time camera detection
* Raspberry Pi deployment
* Drone integration
* Web dashboard
* Model ensemble
* Larger dataset

---

# 👨‍💻 Author

Punith Kumar R
Electrical Engineering + AI/Software Projects

---

# 📜 License

For academic and educational use.

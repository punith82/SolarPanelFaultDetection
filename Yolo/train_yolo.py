"""
Train YOLOv8 model for solar panel detection.
"""

from ultralytics import YOLO
import torch

def train_panel_detection():
    """Train YOLOv8 model for panel detection."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🚀 YOLO Panel Detection Training Started                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {'GPU' if device != 'cpu' else 'CPU'}")
    
    yaml_path = "/Users/punithkumarr/Desktop/SPFD1/Yolo/data.yaml"
    
    # Load YOLOv8 nano model
    print("\n📥 Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print("\n🎓 Training configuration:")
    print("   - Model: YOLOv8 Nano")
    print("   - Epochs: 50")
    print("   - Image size: 640x640")
    print("   - Batch size: 8")
    print("\n⏳ Training in progress... (this may take 5-20 minutes)")
    print("   Training samples: 169 | Validation samples: 31\n")
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        patience=15,
        save=True,
        device=device,
        workers=2,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.0,
        project='runs/panel_detection',
        name='yolov8n_panels',
        exist_ok=False,
        verbose=True,
        seed=42,
    )
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  ✅ Training Complete!                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📊 Best model saved at:")
    print(f"   runs/panel_detection/yolov8n_panels/weights/best.pt")
    
    # Validate the model
    print("\n📈 Validating model...")
    metrics = model.val()
    
    print("\n✨ Training Results Summary:")
    print("   Check 'runs/panel_detection/yolov8n_panels/' for detailed metrics")
    
    return results, metrics

if __name__ == "__main__":
    train_panel_detection()

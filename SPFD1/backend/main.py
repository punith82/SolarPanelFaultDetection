from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure the workspace root is importable so the existing pipeline can be reused.
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import integrated_pipeline as pipeline

app = FastAPI(
    title="Solar Panel Fault Detection API",
    description="Backend API for the Solar Panel Fault Detection System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model handles loaded once at app startup.
yolo_model = None
eff_model = None


@app.on_event("startup")
def startup_event():
    global yolo_model, eff_model
    eff_model = pipeline.load_efficientnet("best")
    yolo_model = pipeline.YOLO(str(pipeline.DEFAULT_YOLO_MODEL))


@app.get("/")
def root():
    return {"message": "Solar Panel Fault Detection API is running."}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    suffix = Path(image.filename).suffix or ".jpg"
    if suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(await image.read())

    output_dir = Path(tempfile.gettempdir()) / "spfd_backend_outputs"
    try:
        detections = pipeline.process_top_view_image(
            image_path=temp_path,
            yolo_model=yolo_model,
            eff_model=eff_model,
            output_dir=output_dir,
            save_crops=False,
            crop_dir=None,
            conf_threshold=0.5,
        )

        results = [
            {
                "panel_id": panel.get("panel_id"),
                "status": panel.get("predicted_class"),
                "confidence": round(panel.get("confidence", 0.0), 4),
                "bbox": panel.get("xywh", []),
            }
            for panel in detections
        ]

        return {"total_panels": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass

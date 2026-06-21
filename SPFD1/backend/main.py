from pathlib import Path
import sys
import tempfile
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure the workspace root is importable
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
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(tempfile.gettempdir()) / "spfd_backend_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/results",
    StaticFiles(directory=OUTPUT_DIR),
    name="results",
)

yolo_model = None
eff_model = None


@app.on_event("startup")
def startup_event():
    global yolo_model, eff_model

    eff_model = pipeline.load_efficientnet("best")
    yolo_model = pipeline.YOLO(
        str(pipeline.DEFAULT_YOLO_MODEL)
    )


@app.get("/")
def root():
    return {
        "message": "Solar Panel Fault Detection API is running."
    }


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    suffix = Path(image.filename).suffix or ".jpg"

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tiff",
    }

    if suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(await image.read())

    try:
        start_time = time.perf_counter()

        result = pipeline.process_top_view_image(
            image_path=temp_path,
            yolo_model=yolo_model,
            eff_model=eff_model,
            output_dir=OUTPUT_DIR,
            save_crops=False,
            crop_dir=None,
            conf_threshold=0.5,
        )

        inference_time = round(
            time.perf_counter() - start_time,
            2,
        )

        detections = result["detections"]
        annotated_path = result["annotated_image"]

        results = [
            {
                "panel_id": panel.get("panel_id"),
                "status": panel.get("predicted_class"),
                "confidence": round(
                    panel.get("confidence", 0.0),
                    4,
                ),
                "bbox": panel.get("xywh", []),
            }
            for panel in detections
        ]

        return {
            "total_panels": len(results),
            "results": results,
            "inference_time": inference_time,
            "annotated_image": (
                f"http://localhost:8000/results/{annotated_path.name}"
                if annotated_path
                else None
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass
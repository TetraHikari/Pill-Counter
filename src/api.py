"""Counting API for the Next.js app: POST a photo, get the count and boxes back.

    .venv/bin/uvicorn api:app --app-dir src --port 8000

POST /count                       fine-tuned model with the thresholds tuned on val
POST /count?expected=<pill type>  also flag pills whose type differs from the expected one

With results/models/pill_type.pt present, every box also gets a pill type from the
classifier (option B: detect, then classify on a masked full-resolution crop of each pill).
Boxes are in the pixel frame of the uploaded image, after applying its EXIF rotation.
"""
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from ultralytics import YOLO

from common import MODELS, RESULTS, device
from crops import square_crop

# thai_mix_s1024 (our photos + public multi-drug photos) is the best counter on every check, with thresholds tuned on
# validation one photo per call, as this server runs (docs/eval_mix_report.md): val 93% exact, mixed tray s20 95%,
# other medicines (held-out) far ahead of thai_v2, same speed. To go back:
# PILLTALLY_MODEL=results/models/thai_v2.pt PILLTALLY_THRESHOLDS=thresholds_thai_v2.json
MODEL_PATH = os.environ.get("PILLTALLY_MODEL", str(MODELS / "thai_mix_s1024.pt"))
THRESHOLDS = json.loads((RESULTS / os.environ.get("PILLTALLY_THRESHOLDS", "thresholds_thai_mix_s1024_single.json")).read_text())
REVIEW_MARGIN = 0.2  # boxes scoring less than conf + this are flagged for a second look
TYPE_MODEL = MODELS / "pill_type.pt"
TYPE_MIN_CONF = 0.6  # below this the pill type is reported as "unknown" (the classifier only knows PILL_TYPES)
# Save every photo + result locally, to find failure cases and turn them into new training data.
LOG_DIR = RESULTS / "app_uploads" if os.environ.get("PILLTALLY_LOG_UPLOADS", "1") == "1" else None

model = YOLO(MODEL_PATH)
dev = device()
type_model = YOLO(str(TYPE_MODEL)) if TYPE_MODEL.exists() else None
PILL_TYPES = sorted(type_model.names.values()) if type_model else []
app = FastAPI(title="PillTally")


@app.get("/health")
def health():
    return {"model": os.path.basename(MODEL_PATH), "conf": THRESHOLDS["conf"], "iou": THRESHOLDS["iou"],
            "pill_types": PILL_TYPES}


def count_ai(bgr):
    conf, iou = THRESHOLDS["conf"], THRESHOLDS["iou"]
    r = model.predict(bgr, imgsz=THRESHOLDS.get("imgsz", 1024), conf=conf, iou=iou, max_det=500, device=dev, verbose=False)[0]
    review = conf + REVIEW_MARGIN
    return [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(c, 3), "check": c < review}
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def add_types(bgr, boxes, expected):
    """Classify each detected pill from a masked crop of the uploaded (full-resolution) image."""
    if not type_model or not boxes:
        return
    H, W = bgr.shape[:2]
    crops = [square_crop(bgr, (b["x1"] + b["x2"]) / 2 / W, (b["y1"] + b["y2"]) / 2 / H,
                         (b["x2"] - b["x1"]) / W, (b["y2"] - b["y1"]) / H) for b in boxes]
    for b, r in zip(boxes, type_model.predict(crops, imgsz=224, device=dev, verbose=False)):
        b["type_guess"] = type_model.names[int(r.probs.top1)]   # closest known form, kept for debugging
        b["type_conf"] = round(float(r.probs.top1conf), 3)
        b["type_unsure"] = b["type_conf"] < TYPE_MIN_CONF
        b["type"] = "unknown" if b["type_unsure"] else b["type_guess"]
        b["mismatch"] = bool(expected) and (b["type"] != expected or b["type_unsure"])


@app.post("/count")
async def count(file: UploadFile = File(...), expected: str | None = None):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that image. Please send a JPG or PNG photo.")
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])  # BGR, as Ultralytics and OpenCV expect
    if expected and expected not in PILL_TYPES:
        raise HTTPException(400, f"Unknown pill type '{expected}'. Known: {', '.join(PILL_TYPES) or 'none'}.")
    t0 = time.perf_counter()
    boxes = count_ai(bgr)
    add_types(bgr, boxes, expected)
    types = {}
    for b in boxes:
        if "type" in b:
            types[b["type"]] = types.get(b["type"], 0) + 1
    result = {"count": len(boxes), "to_check": sum(b["check"] for b in boxes),
              "types": types, "expected": expected, "mismatches": sum(b.get("mismatch", False) for b in boxes),
              "width": img.width, "height": img.height, "boxes": boxes,
              "seconds": round(time.perf_counter() - t0, 2)}
    if LOG_DIR:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stem = LOG_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        img.save(f"{stem}.jpg", quality=92)
        Path(f"{stem}.json").write_text(json.dumps({**result, "model": os.path.basename(MODEL_PATH)}, indent=1))
    return result

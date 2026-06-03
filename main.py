import base64
import json
import io
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel, Field

from config import (
    MODEL_DIR, IMG_SIZE, NUM_CLASSES,
    CONFIDENCE_THRESHOLD, CANVAS_SIZE
)

app = FastAPI(
    title="Digit Recognizer API",
    description=(
        "Handwritten digit recognition using a CNN trained on MNIST. "
        "Draw a digit on the canvas and the model predicts it in real time. "
        "POST /predict accepts a base64-encoded PNG and returns the predicted "
        "digit with confidence scores."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

MODEL_LOADED = False
model = None
model_metrics = {}

_model_path = MODEL_DIR / 'cnn_best.keras'
if _model_path.exists():
    import tensorflow as tf
    model = tf.keras.models.load_model(_model_path)
    MODEL_LOADED = True
    _metrics_path = MODEL_DIR / 'model_metrics.json'
    if _metrics_path.exists():
        with open(_metrics_path) as f:
            model_metrics = json.load(f)

# ── Pydantic models ────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    image_data: str = Field(..., description="Base64-encoded PNG image of the drawn digit")

class DigitPrediction(BaseModel):
    digit: int
    confidence: float
    confidence_pct: int
    is_uncertain: bool
    probabilities: list[float]
    top3: list[dict]
    model_input_preview: str = ""  # base64 PNG of the 28x28 image the model actually saw

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: float = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_LOADED,
        model_accuracy=model_metrics.get("test_accuracy"),
    )


@app.post("/predict", response_model=DigitPrediction)
async def predict(body: PredictionRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 → bytes → PIL Image
        image_bytes = base64.b64decode(body.image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # MNIST is grayscale — convert canvas RGB/RGBA to single channel
        image = image.convert('L')

        # The canvas draws at CANVAS_SIZE (280px) for smooth user experience.
        # The model expects 28x28. LANCZOS gives the best quality for downscaling.
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        image_array = np.array(image, dtype='float32')
        image_array = image_array / 255.0

        # MNIST is white digits on black background; the canvas draws the opposite.
        # Invert pixel values so black-on-white canvas input becomes
        # white-on-black MNIST format. Without this, every prediction is wrong.
        image_array = 1.0 - image_array

        # Add batch and channel dimensions: (28, 28) → (1, 28, 28, 1)
        image_array = image_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or undecodable image data")

    probs = model.predict(image_array, verbose=0)[0].tolist()
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    sorted_idx = sorted(range(NUM_CLASSES), key=lambda i: probs[i], reverse=True)
    top3 = [
        {"digit": idx, "confidence": round(probs[idx], 4), "confidence_pct": int(probs[idx] * 100)}
        for idx in sorted_idx[:3]
    ]

    # Encode the 28x28 model input as a tiny PNG for client-side debugging
    preview_pixels = (image_array[0, :, :, 0] * 255).astype('uint8')
    preview_img = Image.fromarray(preview_pixels, mode='L')
    preview_buf = io.BytesIO()
    preview_img.save(preview_buf, format='PNG')
    preview_b64 = base64.b64encode(preview_buf.getvalue()).decode()

    return DigitPrediction(
        digit=digit,
        confidence=confidence,
        confidence_pct=int(confidence * 100),
        is_uncertain=confidence < CONFIDENCE_THRESHOLD,
        probabilities=[round(p, 4) for p in probs],
        top3=top3,
        model_input_preview=preview_b64,
    )


@app.get("/api/model-info")
async def model_info():
    path = MODEL_DIR / 'model_metrics.json'
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_metrics.json not found")
    with open(path) as f:
        return json.load(f)


@app.get("/api/training-history")
async def training_history():
    path = MODEL_DIR / 'training_history.json'
    if not path.exists():
        raise HTTPException(status_code=404, detail="training_history.json not found")
    with open(path) as f:
        return json.load(f)

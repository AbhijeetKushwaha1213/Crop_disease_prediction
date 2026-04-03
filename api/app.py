"""
Crop Disease Prediction API
----------------------------
A simple FastAPI server that takes a leaf image and predicts the disease.
Uses a TFLite model for fast inference.
"""

import os
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ──────────────────────────────────────────────
# Setup paths (relative to project root)
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # project root
MODEL_PATH = BASE_DIR / "crop_model.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create upload folder if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Load labels
# ──────────────────────────────────────────────
def load_labels(path: Path) -> list[str]:
    """Read labels.txt and return a clean list of class names."""
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

LABELS = load_labels(LABELS_PATH)

# ──────────────────────────────────────────────
# Load TFLite model
# ──────────────────────────────────────────────
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
except ImportError:
    # Fallback: try tflite_runtime (lighter package)
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(MODEL_PATH))

interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Figure out the expected image size from the model
INPUT_SHAPE = input_details[0]["shape"]        # e.g. [1, 224, 224, 3]
IMG_HEIGHT = INPUT_SHAPE[1]
IMG_WIDTH = INPUT_SHAPE[2]

print(f"✅ Model loaded successfully!")
print(f"   Input shape : {INPUT_SHAPE}")
print(f"   Image size  : {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"   Classes     : {len(LABELS)}")

# ──────────────────────────────────────────────
# Create the FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="🌾 Crop Disease Prediction API",
    description="Upload a leaf image → Get disease prediction",
    version="1.0.0",
)

# Enable CORS for the mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (uploaded images, etc.)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# HTML templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ──────────────────────────────────────────────
# Helper: preprocess image for the model
# ──────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Take raw image bytes, resize, normalize, and prepare for the model.
    
    Steps:
    1. Open the image
    2. Resize to model's expected size (e.g. 224x224)
    3. Convert to numpy array
    4. Normalize pixel values (0-255 → 0-1)
    5. Add batch dimension [1, H, W, 3]
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0               # normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim → [1, H, W, 3]
    
    return img_array


# ──────────────────────────────────────────────
# Helper: run prediction
# ──────────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    """
    Run the TFLite model on an image and return predictions.
    
    Returns a dict with:
      - disease: top predicted class name
      - confidence: confidence score (0-1)
      - top_3: list of top 3 predictions
    """
    # Preprocess
    img_array = preprocess_image(image_bytes)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]["index"])
    probabilities = output[0]  # shape: (num_classes,)
    
    # If output is logits (not probabilities), apply softmax
    if np.max(probabilities) > 1.0 or np.min(probabilities) < 0.0:
        exp_vals = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_vals / np.sum(exp_vals)
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    top_3 = []
    for idx in top_indices:
        top_3.append({
            "disease": LABELS[idx],
            "confidence": round(float(probabilities[idx]) * 100, 2)
        })
    
    return {
        "disease": LABELS[top_indices[0]],
        "confidence": round(float(probabilities[top_indices[0]]) * 100, 2),
        "top_3": top_3,
    }


# ══════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page — shows the upload form."""
    html_file = TEMPLATES_DIR / "index.html"
    html_content = html_file.read_text()
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a leaf image and get disease prediction.
    
    - Accepts: jpg, jpeg, png images
    - Returns: JSON with disease name, confidence, and top 3 predictions
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload a JPG or PNG image."
        )
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Check file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    # Save uploaded image (optional, for history)
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    
    # Run prediction
    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    return JSONResponse(content={
        "success": True,
        "filename": file.filename,
        "prediction": result,
    })


@app.get("/health")
async def health_check():
    """Check if the API and model are running fine."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_file": MODEL_PATH.name,
        "input_size": f"{IMG_WIDTH}x{IMG_HEIGHT}",
        "total_classes": len(LABELS),
        "version": "1.0.0",
    }


@app.get("/classes")
async def get_classes():
    """List all disease classes the model can predict."""
    return {
        "total_classes": len(LABELS),
        "classes": LABELS,
    }

# 🌾 Crop Disease Prediction

A simple machine learning project that predicts crop diseases from leaf images. Upload a photo of a crop leaf and the model tells you what disease it has (or if it's healthy).

---

## 📌 What Does This Project Do?

1. You give the model an **image of a crop leaf**
2. The model looks at the image and **predicts the disease**
3. It tells you the **disease name** and **confidence score** (how sure it is)

Think of it like a doctor for plants — you show it the leaf, it tells you what's wrong! 🩺🌿

---

## 📁 Project Structure (File/Folder Layout)

Here's how all the files and folders are organized:

```
Crop_disease_prediction/
│
├── README.md                  ← You are here! (Project guide)
├── requirements.txt           ← List of Python packages needed
├── .gitignore                 ← Files Git should ignore
│
├── data/                      ← All dataset stuff
│   ├── raw/                   ← Original downloaded images
│   │   ├── train/             ← Training images (model learns from these)
│   │   ├── valid/             ← Validation images (model checks itself)
│   │   └── test/              ← Test images (final exam for the model)
│   └── processed/             ← Cleaned/resized images (ready for training)
│
├── notebooks/                 ← Jupyter notebooks for experimenting
│   ├── 01_data_exploration.ipynb    ← Look at the data, understand it
│   ├── 02_data_preprocessing.ipynb  ← Clean and prepare the data
│   ├── 03_model_training.ipynb      ← Train the model
│   └── 04_model_evaluation.ipynb    ← Test how good the model is
│
├── src/                       ← Main Python source code
│   ├── __init__.py            ← Makes this folder a Python package
│   ├── data_loader.py         ← Code to load and prepare images
│   ├── model.py               ← The ML model architecture (CNN)
│   ├── train.py               ← Script to train the model
│   ├── predict.py             ← Script to predict disease from an image
│   └── utils.py               ← Helper functions (reusable small tools)
│
├── models/                    ← Saved trained models go here
│   └── best_model.pth         ← The best model after training
│
├── api/                       ← Backend API (web server)
│   ├── app.py                 ← Main Flask/FastAPI application
│   ├── routes.py              ← API endpoint definitions
│   └── schemas.py             ← Input/output data format definitions
│
├── static/                    ← Static files for the web interface
│   └── uploads/               ← User-uploaded images land here
│
├── templates/                 ← HTML pages (simple web UI)
│   └── index.html             ← Upload page for the user
│
├── tests/                     ← Unit tests
│   ├── test_model.py          ← Tests for the model
│   └── test_api.py            ← Tests for the API
│
└── config/                    ← Configuration files
    └── config.yaml            ← Settings like image size, batch size, etc.
```

---

## 📊 Dataset

### What Dataset Are We Using?

We use the **PlantVillage Dataset** — a free, popular dataset for crop disease detection.

- 🔗 **Download**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- 📦 **Size**: ~1 GB
- 🖼️ **Total Images**: ~54,000+ leaf images
- 🏷️ **Classes**: 38 different categories (healthy + diseased leaves)

### Sample Classes (What the Model Can Detect)

| # | Class Name | Example |
|---|-----------|---------|
| 1 | Tomato - Early Blight | 🍅 Brown spots on tomato leaves |
| 2 | Tomato - Healthy | 🍅 Clean green tomato leaves |
| 3 | Potato - Late Blight | 🥔 Dark patches on potato leaves |
| 4 | Apple - Apple Scab | 🍎 Scabby spots on apple leaves |
| 5 | Corn - Common Rust | 🌽 Rusty brown spots on corn leaves |
| ... | ... and 33 more classes | |

### Data Split (How We Divide the Data)

| Split | Purpose | % of Data | Approx Images |
|-------|---------|-----------|---------------|
| **Train** | Model learns from these | 70% | ~38,000 |
| **Validation** | Model checks itself during training | 15% | ~8,100 |
| **Test** | Final accuracy check (never seen before) | 15% | ~8,100 |

---

## ✨ Features

### Input Features (What Goes INTO the Model)

| Feature | Description |
|---------|------------|
| **Leaf Image** | A photo of a crop leaf (RGB, resized to 224×224 pixels) |
| **Pixel Values** | Each pixel has 3 color values (Red, Green, Blue), range 0–255 |
| **Normalized Pixels** | We scale pixel values to 0–1 range (helps model learn better) |

### Output (What Comes OUT of the Model)

| Output | Description |
|--------|------------|
| **Disease Name** | The predicted disease (e.g., "Tomato - Early Blight") |
| **Confidence** | How confident the model is (e.g., 95.3%) |
| **Top 3 Predictions** | The 3 most likely diseases with their confidence scores |

### Data Augmentation (Making More Training Data)

We create variations of existing images to help the model learn better:

| Technique | What It Does |
|-----------|-------------|
| **Random Flip** | Flips image left-right |
| **Random Rotation** | Rotates image up to 30° |
| **Color Jitter** | Slightly changes brightness/contrast |
| **Random Crop** | Zooms into random parts of the image |

---

## 🧠 Model Architecture

We use a **CNN (Convolutional Neural Network)** — a type of neural network that's really good at understanding images.

### Option A: Build From Scratch (Simple CNN)

```
Input Image (224×224×3)
    ↓
[Conv Layer 1] → 32 filters → ReLU → MaxPool
    ↓
[Conv Layer 2] → 64 filters → ReLU → MaxPool
    ↓
[Conv Layer 3] → 128 filters → ReLU → MaxPool
    ↓
[Flatten] → Convert to 1D
    ↓
[Dense Layer] → 512 neurons → ReLU → Dropout(0.5)
    ↓
[Output Layer] → 38 neurons (one per class) → Softmax
    ↓
Prediction: "Tomato - Early Blight" (95.3%)
```

### Option B: Transfer Learning (Recommended for Better Accuracy)

Use a pre-trained model like **ResNet50** that already knows how to see patterns in images, and we just teach it about crop diseases.

```
Pre-trained ResNet50 (frozen layers — already trained on ImageNet)
    ↓
[Our Custom Dense Layer] → 512 neurons → ReLU → Dropout
    ↓
[Output Layer] → 38 neurons → Softmax
    ↓
Prediction: "Tomato - Early Blight" (97.8%)
```

> **Why Transfer Learning?** It's like hiring someone who already knows photography and just teaching them about plant diseases — much faster and better results!

---

## 🔧 Tech Stack (Tools We Use)

| Tool | What It's For |
|------|-------------|
| **Python 3.9+** | Main programming language |
| **PyTorch** | Building and training the ML model |
| **torchvision** | Image transforms and pre-trained models |
| **FastAPI** | Building the web API (backend server) |
| **Pillow (PIL)** | Loading and processing images |
| **Matplotlib** | Plotting charts and showing images |
| **NumPy** | Math operations on arrays |
| **Uvicorn** | Running the FastAPI server |
| **Jinja2** | HTML templates for the web page |

---

## 🌐 API Endpoints

The API lets you interact with the model through HTTP requests (like a website talks to a server).

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. `GET /` — Home Page

Shows the upload form where you can upload a leaf image.

| Field | Value |
|-------|-------|
| **Method** | GET |
| **URL** | `http://localhost:8000/` |
| **Response** | HTML page with upload form |

---

#### 2. `POST /predict` — Predict Disease

Upload an image and get the prediction back.

| Field | Value |
|-------|-------|
| **Method** | POST |
| **URL** | `http://localhost:8000/predict` |
| **Content-Type** | multipart/form-data |
| **Body** | `file`: image file (jpg, png) |

**Request Example (using cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf_photo.jpg"
```

**Response Example (JSON):**
```json
{
  "success": true,
  "prediction": {
    "disease": "Tomato - Early Blight",
    "confidence": 0.953,
    "top_3": [
      {"disease": "Tomato - Early Blight", "confidence": 0.953},
      {"disease": "Tomato - Late Blight", "confidence": 0.032},
      {"disease": "Tomato - Healthy", "confidence": 0.015}
    ]
  }
}
```

---

#### 3. `GET /health` — Health Check

Check if the server is running and the model is loaded.

| Field | Value |
|-------|-------|
| **Method** | GET |
| **URL** | `http://localhost:8000/health` |

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

#### 4. `GET /classes` — List All Disease Classes

Get the list of all diseases the model can detect.

| Field | Value |
|-------|-------|
| **Method** | GET |
| **URL** | `http://localhost:8000/classes` |

**Response:**
```json
{
  "total_classes": 38,
  "classes": [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "...and 34 more"
  ]
}
```

---

## 🗃️ Database (Optional — SQLite)

A lightweight database to store prediction history. We use **SQLite** because it's simple, needs no setup, and stores everything in one file.

### Why a Database?

- Keep a **history** of all predictions made
- **Track** which diseases are most common
- Useful for **analytics** later

### Database File

```
data/predictions.db     ← Single SQLite file, no server needed!
```

### Table: `predictions`

| Column | Type | Description |
|--------|------|------------|
| `id` | INTEGER (Primary Key) | Unique ID, auto-incremented |
| `image_name` | TEXT | Name of the uploaded image file |
| `predicted_disease` | TEXT | What disease the model predicted |
| `confidence` | REAL (Float) | Confidence score (0.0 to 1.0) |
| `actual_disease` | TEXT (nullable) | Real disease (if user provides feedback) |
| `created_at` | DATETIME | When the prediction was made |

### Example Data

| id | image_name | predicted_disease | confidence | actual_disease | created_at |
|----|-----------|------------------|------------|---------------|------------|
| 1 | leaf_001.jpg | Tomato - Early Blight | 0.953 | NULL | 2026-04-03 12:00:00 |
| 2 | leaf_002.jpg | Potato - Healthy | 0.987 | NULL | 2026-04-03 12:05:00 |
| 3 | leaf_003.jpg | Corn - Common Rust | 0.891 | Corn - Common Rust | 2026-04-03 12:10:00 |

---

## 🚀 How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Crop_disease_prediction.git
cd Crop_disease_prediction
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

1. Go to [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Download and extract to `data/raw/` folder
3. You should see `data/raw/train/`, `data/raw/valid/`, `data/raw/test/`

### Step 5: Train the Model

```bash
python src/train.py
```

This will take some time (30 min to a few hours depending on your computer). The trained model will be saved to `models/best_model.pth`.

### Step 6: Start the API Server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### Step 7: Open in Browser

Go to `http://localhost:8000` — upload a leaf image and get predictions! 🎉

---

## 📈 Model Training Details

### Hyperparameters (Settings for Training)

| Parameter | Value | What It Means |
|-----------|-------|--------------|
| **Batch Size** | 32 | Process 32 images at a time |
| **Learning Rate** | 0.001 | How big of a step the model takes to learn |
| **Epochs** | 25 | How many times the model goes through all training data |
| **Optimizer** | Adam | Algorithm that helps the model learn efficiently |
| **Loss Function** | CrossEntropyLoss | Measures how wrong the model's predictions are |
| **Image Size** | 224 × 224 | All images resized to this size |

### Expected Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~96-98% |
| **Validation Accuracy** | ~93-95% |
| **Test Accuracy** | ~92-94% |

---

## 🛠️ Configuration (config/config.yaml)

```yaml
# Data Settings
data:
  image_size: 224
  batch_size: 32
  num_workers: 4              # How many CPU cores to use for loading data
  data_dir: "data/raw"

# Model Settings
model:
  architecture: "resnet50"    # Which model to use
  num_classes: 38             # Number of disease categories
  pretrained: true            # Use pre-trained weights?
  dropout: 0.5                # Randomly turn off 50% neurons (prevents overfitting)

# Training Settings
training:
  epochs: 25
  learning_rate: 0.001
  optimizer: "adam"
  save_dir: "models/"

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
  max_file_size: 10485760     # 10 MB max upload
  allowed_extensions: ["jpg", "jpeg", "png"]
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run only model tests
python -m pytest tests/test_model.py

# Run only API tests
python -m pytest tests/test_api.py
```

---

## 📋 requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.5
pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
jinja2>=3.1.0
pytest>=7.4.0
```

---

## 🗺️ Roadmap (What's Next?)

- [x] Project structure setup
- [x] README documentation
- [ ] Data download and exploration
- [ ] Data preprocessing pipeline
- [ ] Build CNN model
- [ ] Train and evaluate model
- [ ] Build FastAPI backend
- [ ] Create simple web UI
- [ ] Add SQLite prediction history
- [ ] Deploy to cloud (Render / Railway / AWS)

---

## 🤔 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | Run `pip install torch torchvision` |
| `CUDA out of memory` | GPU doesn't have enough memory | Reduce `batch_size` to 16 or 8 |
| `FileNotFoundError: data/raw/train` | Dataset not downloaded | Download from Kaggle and extract |
| `Model file not found` | Haven't trained the model yet | Run `python src/train.py` first |

---

## 📝 License

This project is for **learning purposes**. Feel free to use, modify, and share!

---

## 🙏 Credits

- **Dataset**: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **PyTorch**: [pytorch.org](https://pytorch.org)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

> **Made with ❤️ for learning Machine Learning**

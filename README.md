# HarvestLenz

Fruit quality grading system using computer vision and deep learning. Segments individual fruits from basket images, removes background, and grades quality using per-fruit MobileNetV2 classifiers.

## Architecture

```
User selects fruit type (dropdown)
  -> Upload image
  -> Fruit Type Verification (ClassificationService: Pre-check to prevent fruit mismatch)
  -> OpenCV segmentation (watershed + contour analysis)
  -> Background removal (mask-based crop extraction)
  -> MobileNetV2 / EfficientNet grading (per-fruit independent models)
  -> Shelf life + market recommendation
  -> Persist to SQLite
  -> Dashboard report
```

### Key Features & Design Decisions

- **Fruit Verification & Mismatch Protection** — Automatically checks uploaded crops against deep learning classifier models and OpenCV visual feature profiles (HSV color, skin texture, aspect ratio) to catch user dropdown mismatches (e.g., uploading a Pineapple when Grapes is selected).
- **Per-Fruit Dedicated Models** — Independent fine-tuned model weight files for maximum grading precision.
- **3-Class Fruit & 2-Class Vegetable Grading**: Better / Good / Reject for fruits, Good / Reject for vegetables.
- **Background Async Job Processing** — `/analyze` returns immediately with `session_id` while analysis runs asynchronously.

## Supported Fruits & Model Weights

See [`MODEL_PATHS.md`](MODEL_PATHS.md) for full model paths, file sizes, formats, and search resolution details.

| Fruit | Model Weight File | Format |
|:---|:---|:---|
| **Banana** | `backend/backend/app/models/weights/banana.h5` | HDF5 (`.h5`) |
| **Grapes** | `backend/backend/app/models/weights/grapes.keras` | Keras Native (`.keras`) |
| **Guava** | `backend/backend/app/models/weights/guava.keras` / `guava.h5` | Keras Native / HDF5 |
| **Mango** | `backend/backend/app/models/weights/mango.keras` | Keras Native (`.keras`) |
| **Orange** | `backend/backend/app/models/weights/orange.keras` / `orange.h5` | Keras Native / HDF5 |
| **Pineapple** | `backend/backend/app/models/weights/pineapple.keras` | Keras Native (`.keras`) |
| **Pomegranate** | `backend/backend/app/models/weights/pomegranate.keras` | Keras Native (`.keras`) |
| **Strawberry** | `backend/backend/app/models/weights/strawberry.keras` / `strawberry.h5` | Keras Native / HDF5 |

## Project Structure

```
HarvestLenz/
  backend/
    backend/
      app/
        api/
          analyze_basket.py      # POST /analyze, GET /analysis/{id}, GET /stats
        models/
          shared_mobilenet.py    # MobileNetV2 architecture definition
          model_loader.py        # Loads .keras weight files per fruit
          weights/               # Trained .keras weight files
            mango.keras
            pineapple.keras
            grapes.keras
            pomegranate.keras
        services/
          analysis_orchestrator.py  # Pipeline coordinator
          segmentation_service.py   # Fruit detection (OpenCV)
          background_removal.py     # Mask-based crop extraction
          grading_service.py        # MobileNetV2 inference
          shelf_service.py          # Shelf life prediction
          market_service.py         # Market recommendations
        database/
          models.py             # SQLAlchemy models (AnalysisSession, BasketFruit)
          connection.py         # Async SQLite engine
      training/
        train.py                # Unified training script
  frontend/
    index.html                  # Dashboard
    scan.html                   # Analysis results
    js/app.js                   # API client
    css/styles.css              # Styles
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Backend

```bash
cd backend/backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Server runs at `http://127.0.0.1:8001`

### Frontend

The backend serves static files from `frontend/`. Open `http://127.0.0.1:8001` in your browser.

## Training

```bash
cd backend/backend
python training/train.py --fruit mango --epochs 25
python training/train.py --fruit pineapple --epochs 25
python training/train.py --fruit grapes --epochs 25
python training/train.py --fruit pomegranate --epochs 25
```

### Training approach

Two-phase transfer learning:

1. **Phase 1** (10 epochs): Train only the classification head. MobileNetV2 base frozen. lr=1e-3
2. **Phase 2** (remaining epochs): Unfreeze from layer 100, fine-tune with lr=1e-4

Data augmentation: rotation, brightness, zoom, horizontal flip, hue shift.

### Dataset format

```
dataset-{fruit}/
  Better/     # High quality
  Good/       # Standard quality
  Reject/     # Damaged/poor quality
```

## API Endpoints & App Integration

For full step-by-step developer integration guides (Flutter, React Native, iOS, Android), payload schemas, and code samples, see [`APP_INTEGRATION_GUIDE.md`](APP_INTEGRATION_GUIDE.md).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Upload image + `fruit_type` (or `fruit_type=auto` for AI classification), returns `session_id` |
| GET | `/analysis/{session_id}` | Get full quality analysis report & mismatch warnings |
| GET | `/analysis/{session_id}/status` | Poll background analysis job status |
| POST | `/scan/classify` | Instant single-crop fruit classifier endpoint |
| GET | `/stats` | Dashboard aggregate statistics |
| GET | `/fruits/supported` | List all 21 supported fruit & vegetable types |

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy (async), SQLite
- **ML**: TensorFlow/Keras (MobileNetV2), OpenCV
- **Frontend**: Vanilla HTML/CSS/JS


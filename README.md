# HarvestLenz

Fruit quality grading system using computer vision and deep learning. Segments individual fruits from basket images, removes background, and grades quality using per-fruit MobileNetV2 classifiers.

## Architecture

```
User selects fruit type (dropdown)
  -> Upload image
  -> OpenCV segmentation (watershed + contour analysis)
  -> Background removal (mask-based crop extraction)
  -> MobileNetV2 grading (per-fruit independent models)
  -> Shelf life + market recommendation
  -> Persist to SQLite
  -> Dashboard report
```

### Key Design Decisions

- **No ML detector** for fruit detection — uses classical OpenCV (HSV thresholds + watershed + contour analysis)
- **No fruit classification** — user selects fruit from dropdown, so 4 independent binary classifiers
- **3-class grading**: Better / Good / Reject (alphabetical for TensorFlow class ordering)
- **Background task** — `/analyze` returns immediately with session_id, grading runs async

## Supported Fruits

| Fruit | Model |
|-------|-------|
| Mango | MobileNetV2 |
| Pineapple | MobileNetV2 |
| Grapes | MobileNetV2 |
| Pomegranate | MobileNetV2 |

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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Upload image + fruit type, returns session_id |
| GET | `/analysis/{session_id}` | Get full analysis results |
| GET | `/analysis/{session_id}/status` | Poll job status |
| GET | `/stats` | Dashboard aggregate statistics |
| GET | `/fruits/supported` | List supported fruit types |

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy (async), SQLite
- **ML**: TensorFlow/Keras (MobileNetV2), OpenCV
- **Frontend**: Vanilla HTML/CSS/JS

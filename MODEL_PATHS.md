# HarvestLenz - Trained Model Registry & Paths

This document provides a complete registry of all trained Machine Learning models in the HarvestLenz repository, including their exact relative file paths, model formats, file sizes, and loading precedence.

---

## 1. Primary Model Weights (`backend/backend/app/models/weights/`)

These are the primary weight files loaded dynamically by the runtime service via [`model_loader.py`](file:///c:/Users/sssha/harv/backend/backend/app/models/model_loader.py).

| Fruit | Relative Path | File Size | Format | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Banana** | `backend/backend/app/models/weights/banana.h5` | 13.13 MB | HDF5 (`.h5`) | Active |
| **Grapes** | `backend/backend/app/models/weights/grapes.keras` | 27.57 MB | Keras Native (`.keras`) | Active |
| **Guava** | `backend/backend/app/models/weights/guava.keras`<br>`backend/backend/app/models/weights/guava.h5` | 13.33 MB<br>13.29 MB | Keras Native (`.keras`)<br>HDF5 (`.h5`) | Active |
| **Mango** | `backend/backend/app/models/weights/mango.keras` | 27.57 MB | Keras Native (`.keras`) | Active |
| **Orange** | `backend/backend/app/models/weights/orange.keras`<br>`backend/backend/app/models/weights/orange.h5` | 13.34 MB<br>13.29 MB | Keras Native (`.keras`)<br>HDF5 (`.h5`) | Active |
| **Pineapple** | `backend/backend/app/models/weights/pineapple.keras` | 27.57 MB | Keras Native (`.keras`) | Active |
| **Pomegranate** | `backend/backend/app/models/weights/pomegranate.keras` | 27.57 MB | Keras Native (`.keras`) | Active |
| **Strawberry** | `backend/backend/app/models/weights/strawberry.keras`<br>`backend/backend/app/models/weights/strawberry.h5` | 13.33 MB<br>13.29 MB | Keras Native (`.keras`)<br>HDF5 (`.h5`) | Active |

---

## 2. Legacy & Additional Model Locations

| Model Description | Relative Path | File Size | Format |
| :--- | :--- | :--- | :--- |
| **Banana Fine-Tuned Model** | `backend/backend/models/banana_quality_ft.h5` | 27.38 MB | HDF5 (`.h5`) |
| **Banana Temp Model** | `backend/backend/models/banana_quality_temp.h5` | 13.13 MB | HDF5 (`.h5`) |
| **Banana AI Service Model** | `backend/backend/app/ai/models/banana_model.h5` | 13.13 MB | HDF5 (`.h5`) |
| **Banana AI Legacy Model** | `backend/app/ai/models/banana_model.h5` | 13.13 MB | HDF5 (`.h5`) |

---

## 3. Model Resolution Search Order

The application backend resolves weight files dynamically in the following candidate search order:

1. `backend/backend/app/models/weights/{fruit}.keras`
2. `backend/backend/app/models/weights/{fruit}.h5`
3. `backend/backend/models/{fruit}_quality_ft.h5`
4. `backend/backend/models/{fruit}_model.h5`
5. `backend/backend/app/ai/models/{fruit}_model.h5`

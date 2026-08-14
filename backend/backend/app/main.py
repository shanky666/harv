"""
HarvestLenz — AI-Powered Fruit Quality Intelligence System
FastAPI Backend  |  Production Ready

Redesigned Architecture:
    - User selects fruit type from dropdown
    - Instance segmentation replaces bounding boxes
    - Per-fruit MobileNetV2 grading models
    - Background removal before classification
    - No fruit type classification needed
"""
import os
import sys

if sys.prefix == sys.base_prefix:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(current_dir, ".."))
    workspace_root = os.path.abspath(os.path.join(backend_dir, ".."))
    possible_pythons = [
        os.path.join(backend_dir, "venv", "Scripts", "python.exe"),
        os.path.join(workspace_root, "venv", "Scripts", "python.exe"),
    ]
    for venv_python in possible_pythons:
        if os.path.exists(venv_python):
            if os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
                os.execv(venv_python, [venv_python] + sys.argv)

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
backend_dir = os.path.abspath(os.path.join(current_dir, ".."))

for path in [workspace_root, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from loguru import logger

from app.core.config import settings
from app.database.connection import engine
from app.database.models import Base
from app.middleware.logging import LoggingMiddleware
from app.services.grading_service import grading_service
from app.services.vegetable_grading_service import vegetable_grading_service

SUPPORTED_FRUITS = ["mango", "pineapple", "grapes", "pomegranate", "orange", "guava", "kiwi", "watermelon", "banana", "cocoa", "coffee", "strawberry", "plum", "peach", "pear"]
SUPPORTED_VEGETABLES = ["carrot", "tomato", "onion", "cucumber", "capsicum", "potato"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HarvestLenz backend (v2.0 - Segmentation Architecture)...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")

    try:
        import torch
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.warning(f"CUDA check failed: {e}")

    try:
        from app.utils.model_verifier import verify_all_models
        verify_all_models()
    except Exception as e:
        logger.error(f"Model verification error: {e}")

    grading_service.load_models()
    vegetable_grading_service.load_models(SUPPORTED_VEGETABLES)

    import time
    startup_start = time.time()

    os.makedirs(os.path.join(settings.STORAGE_BASE_PATH, "uploads", "original"), exist_ok=True)
    os.makedirs(os.path.join(settings.STORAGE_BASE_PATH, "uploads", "processed"), exist_ok=True)
    os.makedirs(os.path.join(settings.STORAGE_BASE_PATH, "uploads", "reports"), exist_ok=True)
    os.makedirs(os.path.join(settings.STORAGE_BASE_PATH, "crops"), exist_ok=True)
    os.makedirs(os.path.join("app", "models", "weights"), exist_ok=True)
    logger.info("Storage directories ready")

    try:
        from app.database.connection import AsyncSessionLocal
        from app.services.analysis_orchestrator import analysis_orchestrator

        possible_warmups = [
            "dataset_mango/dataset_mango/Better/alternaria_007.jpg",
            os.path.join(workspace_root, "dataset_mango/dataset_mango/Better/alternaria_007.jpg"),
        ]
        warmup_img = None
        for p in possible_warmups:
            if os.path.exists(p):
                warmup_img = p
                break

        if warmup_img:
            warmup_t0 = time.time()
            async with AsyncSessionLocal() as db_session:
                await analysis_orchestrator.analyze(
                    db_session, warmup_img, "mango", is_single=True
                )
            warmup_time = time.time() - warmup_t0
            logger.info(f"Warmup analysis completed in {warmup_time:.2f}s")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

    yield
    logger.info("HarvestLenz backend shutting down...")


app = FastAPI(
    title="HarvestLenz API",
    description=(
        "## HarvestLenz v2.0 — Segmentation-Based Fruit Quality System\n\n"
        "User selects fruit type -> Upload image -> Instance segmentation -> "
        "Background removal -> Per-fruit grading -> Quality report.\n\n"
        "**New workflow:** `POST /analyze?fruit_type=mango` -> `GET /analysis/{id}/status` "
        "-> `GET /analysis/{id}`"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

_frontend_dir = os.path.abspath(os.path.join(workspace_root, "frontend"))
if os.path.isdir(_frontend_dir):
    app.mount("/frontend", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


from app.api import auth, users, report, passport, analyze_basket

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(report.router)
app.include_router(passport.router)
app.include_router(analyze_basket.router)


@app.get("/", tags=["Health"])
async def root():
    return RedirectResponse(url="/frontend/index.html")


@app.get("/health", tags=["Health"])
async def health():
    from app.models.model_loader import _resolve_weight_path
    from app.services.vegetable_grading_service import get_vegetable_weight_path
    loaded_models = []
    for fruit in SUPPORTED_FRUITS:
        path = _resolve_weight_path(fruit)
        if path:
            loaded_models.append(fruit)
    loaded_veggies = []
    for veg in SUPPORTED_VEGETABLES:
        path = get_vegetable_weight_path(veg)
        if path:
            loaded_veggies.append(veg)

    return {
        "api": "ok",
        "version": "2.0.0",
        "architecture": "segmentation + per-fruit grading",
        "loaded_models": loaded_models,
        "supported_fruits": SUPPORTED_FRUITS,
        "supported_vegetables": SUPPORTED_VEGETABLES,
        "loaded_vegetable_models": loaded_veggies,
        "demo_mode": settings.DEMO_MODE,
    }

import os
import sys
import pytest

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai_models.model_registry.model_loader import resolve_active_model_path, load_registered_model

def test_resolve_active_model_path():
    # Verify that the active versions are mapped correctly in the registry.json
    yolo_path = resolve_active_model_path("yolo")
    assert yolo_path is not None
    assert "yolo" in yolo_path.lower()
    
    mango_path = resolve_active_model_path("mango_model")
    assert mango_path is not None
    assert "mango" in mango_path.lower()

def test_load_registered_model():
    # Verify loading registered models returns a model instance (real or mock fallback)
    yolo_model = load_registered_model("yolo", dry_run=True)
    assert yolo_model is not None
    
    mango_model = load_registered_model("mango_model", dry_run=True)
    assert mango_model is not None

import os
import shutil
import tempfile
import json
import pytest
from ai_models.model_registry.model_registry import register_model, rollback_model, get_latest_version
from ai_models.model_registry.model_loader import load_registered_model, resolve_active_model_path
from ai_models.model_registry.version_manager import list_model_versions, get_active_version, promote_model_version

# Use a temporary models folder for tests to avoid altering the production registry
@pytest.fixture(autouse=True)
def setup_temp_registry(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr("ai_models.model_registry.model_registry.REGISTRY_DIR", temp_dir)
    monkeypatch.setattr("ai_models.model_registry.model_registry.METADATA_PATH", os.path.join(temp_dir, "registry.json"))
    monkeypatch.setattr("ai_models.model_registry.model_loader.REGISTRY_DIR", temp_dir)
    monkeypatch.setattr("ai_models.model_registry.model_loader.METADATA_PATH", os.path.join(temp_dir, "registry.json"))

    monkeypatch.setattr("ai_models.model_registry.version_manager.REGISTRY_DIR", temp_dir)
    monkeypatch.setattr("ai_models.model_registry.version_manager.METADATA_PATH", os.path.join(temp_dir, "registry.json"))
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_model_registration_and_versioning(setup_temp_registry):
    temp_dir = setup_temp_registry
    
    # Create dummy source weight file
    src_weight = os.path.join(temp_dir, "best_weights.h5")
    with open(src_weight, "w") as f:
        f.write("weights")
        
    meta = {
        "architecture": "MobileNetV3",
        "dataset": "datasets/mango",
        "metrics": {"accuracy": 0.92, "f1_score": 0.91}
    }
    
    # 1. Register version v1
    dest_path1 = register_model("Mango_Model", src_weight, version="v1", metadata=meta)
    assert os.path.exists(dest_path1)
    assert "mango_model_v1.h5" in dest_path1
    
    # Check latest version
    assert get_latest_version("Mango_Model") == "v1"
    assert get_active_version("Mango_Model") == "v1"
    
    # 2. Register version v2 (auto-increment)
    dest_path2 = register_model("Mango_Model", src_weight, metadata=meta)
    assert os.path.exists(dest_path2)
    assert "mango_model_v2.h5" in dest_path2
    assert get_latest_version("Mango_Model") == "v2"
    # By default, registering sets active version to the newly registered version
    assert get_active_version("Mango_Model") == "v2"

def test_model_promotion_and_rollback(setup_temp_registry):
    temp_dir = setup_temp_registry
    src_weight = os.path.join(temp_dir, "best_weights.h5")
    with open(src_weight, "w") as f:
        f.write("weights")
        
    register_model("Orange_Model", src_weight, version="v1")
    register_model("Orange_Model", src_weight, version="v2")
    
    # Active version is v2 currently
    assert get_active_version("Orange_Model") == "v2"
    
    # Rollback to v1
    success = rollback_model("Orange_Model", "v1")
    assert success
    assert get_active_version("Orange_Model") == "v1"
    
    # Promote back to v2
    promo_success = promote_model_version("Orange_Model", "v2")
    assert promo_success
    assert get_active_version("Orange_Model") == "v2"

def test_model_loader(setup_temp_registry):
    temp_dir = setup_temp_registry
    src_weight = os.path.join(temp_dir, "yolo_best.pt")
    with open(src_weight, "w") as f:
        f.write("yolo")
        
    # Register Orange YOLO
    register_model("Orange_YOLO", src_weight, version="v1")
    
    active_path = resolve_active_model_path("Orange_YOLO")
    assert active_path is not None
    assert "orange_yolo_v1.pt" in active_path
    
    # Test loading using dry_run fallback
    model = load_registered_model("Orange_YOLO", dry_run=True)
    assert model is not None
    assert hasattr(model, "predict")

import os
import json
import shutil
import tempfile
import pytest

from ai_models.datasets.dataset_validator import validate_dataset
from ai_models.datasets.dataset_preparation import split_dataset, generate_data_yaml
from ai_models.datasets.augmentation import rotate_image, flip_image
from ai_models.datasets.check_dataset_balance import check_dataset_balance
from ai_models.datasets.check_annotation_quality import check_annotation_quality
from ai_models.datasets.dataset_statistics import generate_dataset_statistics
import numpy as np


def test_dataset_validator():
    # Run validator on a temp folder or defaults
    report = validate_dataset("non_existent_folder")
    assert report["status"] == "WARNING"
    assert "errors" in report
    assert len(report["supported_fruits"]) > 0

def test_generate_data_yaml():
    temp_dir = tempfile.mkdtemp()
    temp_yaml = os.path.join(temp_dir, "data.yaml")
    try:
        generate_data_yaml(temp_yaml)
        assert os.path.exists(temp_yaml)
        
        # Read file to verify structure
        import yaml
        with open(temp_yaml, "r") as f:
            data = yaml.safe_load(f)
        assert "train" in data
        assert "val" in data
        assert "names" in data
        assert "Mango" in data["names"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_split_dataset():
    # Create temp source directory and dummy images
    src_dir = tempfile.mkdtemp()
    dest_dir = tempfile.mkdtemp()
    try:
        # Create subdirectories simulating classes
        for cls in ["Good", "Reject"]:
            cls_path = os.path.join(src_dir, cls)
            os.makedirs(cls_path, exist_ok=True)
            # Create a couple of empty files
            for i in range(5):
                with open(os.path.join(cls_path, f"img_{i}.jpg"), "w") as f:
                    f.write("dummy")
                    
        # Split: 80% train, 20% val (ratios must sum to <= 1.0)
        metrics = split_dataset(src_dir, dest_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
        
        # Total files = 10. Train = 8, Val = 2.
        assert metrics["train"] == 8
        assert metrics["val"] == 2
        assert os.path.exists(os.path.join(dest_dir, "train", "Good"))
        assert os.path.exists(os.path.join(dest_dir, "val", "Reject"))
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(dest_dir, ignore_errors=True)

def test_image_augmentations():
    # Create a simple numpy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    rotated = rotate_image(img, 90)
    assert rotated.shape == (100, 100, 3)
    
    flipped = flip_image(img, 1)
    assert flipped.shape == (100, 100, 3)

def test_dataset_balance():
    # Run balance checker on non-existent folder
    report = check_dataset_balance("non_existent_folder")
    assert report["status"] == "NO_DATA"
    
    # Test real imbalance
    temp_dir = tempfile.mkdtemp()
    try:
        # Create mango class folder
        mango_path = os.path.join(temp_dir, "mango")
        os.makedirs(os.path.join(mango_path, "Good"), exist_ok=True)
        os.makedirs(os.path.join(mango_path, "Reject"), exist_ok=True)
        
        # Write 10 images in Good and 1 image in Reject
        for i in range(10):
            with open(os.path.join(mango_path, "Good", f"img_{i}.jpg"), "w") as f:
                f.write("dummy")
        with open(os.path.join(mango_path, "Reject", "img_0.jpg"), "w") as f:
            f.write("dummy")
            
        report_imbalanced = check_dataset_balance(temp_dir)
        assert report_imbalanced["status"] == "IMBALANCED"
        assert len(report_imbalanced["warnings"]) > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_annotation_quality():
    # Create a temp directory with a dummy annotation file
    temp_dir = tempfile.mkdtemp()
    try:
        # Write valid line
        with open(os.path.join(temp_dir, "001.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            
        report = check_annotation_quality(temp_dir)
        assert report["status"] == "VALID"
        assert report["total_files_checked"] == 1
        assert report["total_boxes_checked"] == 1
        
        # Write invalid class index line
        with open(os.path.join(temp_dir, "002.txt"), "w") as f:
            f.write("99 0.5 0.5 0.2 0.3\n")
            
        report_invalid = check_annotation_quality(temp_dir)
        assert report_invalid["status"] == "INVALID"
        assert len(report_invalid["invalid_class_ids"]) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_dataset_statistics():
    temp_src = tempfile.mkdtemp()
    temp_report = os.path.join(temp_src, "report.json")
    try:
        # Generate stats
        report = generate_dataset_statistics(temp_src, temp_report)
        assert report["pipeline"] == "HarvestLenz Dataset Summary"
        assert os.path.exists(temp_report)
        
        # Verify JSON content
        with open(temp_report, "r") as f:
            data = json.load(f)
        assert "grading_dataset" in data
    finally:
        shutil.rmtree(temp_src, ignore_errors=True)


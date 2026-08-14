import pytest
from app.services.passport_service import passport_service

def test_generate_fruit_passport():
    fruit = {
        "fruit_id": "FRUIT_0001",
        "fruit_type": "Mango",
        "grade": "Good",
        "confidence": 0.95,
        "defect_score": 0.05,
        "shelf_life_days": 7,
        "expiry_date": "2026-06-22",
        "recommended_market": "Export",
        "crop_path": "/storage/crops/scan_1/FRUIT_0001.jpg"
    }
    
    passport = passport_service.generate_fruit_passport(fruit, scan_date="2026-06-15")
    
    assert passport["fruit_id"] == "FRUIT_0001"
    assert passport["fruit_type"] == "Mango"
    assert passport["grade"] == "Good"
    assert passport["confidence"] == 0.95
    assert passport["defect_score"] == 0.05
    assert passport["shelf_life_days"] == 7
    assert passport["expiry_date"] == "2026-06-22"
    assert passport["recommended_market"] == "Export"
    assert passport["scan_date"] == "2026-06-15"
    assert passport["crop_path"] == "/storage/crops/scan_1/FRUIT_0001.jpg"
    assert "passport_id" in passport

def test_generate_all_passports():
    fruits = [
        {
            "fruit_id": "FRUIT_0001",
            "fruit_type": "Mango",
            "grade": "Good",
            "confidence": 0.95,
            "defect_score": 0.05,
            "shelf_life_days": 7,
            "expiry_date": "2026-06-22",
            "recommended_market": "Export",
            "crop_path": "/storage/crops/scan_1/FRUIT_0001.jpg"
        },
        {
            "fruit_id": "FRUIT_0002",
            "fruit_type": "Orange",
            "grade": "Reject",
            "confidence": 0.88,
            "defect_score": 0.65,
            "shelf_life_days": 2,
            "expiry_date": "2026-06-17",
            "recommended_market": "Processing",
            "crop_path": "/storage/crops/scan_1/FRUIT_0002.jpg"
        }
    ]
    
    passports = passport_service.generate_all_passports(fruits, scan_date="2026-06-15")
    
    assert len(passports) == 2
    assert passports[0]["fruit_id"] == "FRUIT_0001"
    assert passports[1]["fruit_id"] == "FRUIT_0002"
    assert passports[1]["recommended_market"] == "Processing"

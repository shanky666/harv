import pytest
from datetime import datetime, date, timedelta
from app.services.shelf_service import shelf_service

def test_predict_shelf_life_from_config():
    # Mango Good -> 7 days
    res = shelf_service.predict_shelf_life("Mango", "Good")
    assert res["shelf_life_days"] == 7
    assert "expiry_date" in res
    
    # Mango Reject -> 1 day
    res_reject = shelf_service.predict_shelf_life("Mango", "Reject")
    assert res_reject["shelf_life_days"] == 1
    
    # Pomegranate Good -> 30 days
    res_pom = shelf_service.predict_shelf_life("Pomegranate", "Good")
    assert res_pom["shelf_life_days"] == 30

def test_shelf_life_defect_adjustment():
    # Base Mango Good is 7 days.
    # Defect score 0.5 should reduce it by 50% -> 3 days
    res = shelf_service.predict_shelf_life("Mango", "Good", defect_score=0.5)
    assert res["shelf_life_days"] == 3
    
    # Defect score 0.9 should reduce it to at least 1 day
    res_min = shelf_service.predict_shelf_life("Mango", "Good", defect_score=0.9)
    assert res_min["shelf_life_days"] == 1

def test_calculate_expiry_and_remaining():
    today_str = date.today().strftime("%Y-%m-%d")
    expiry = shelf_service.calculate_expiry_date(5, start_date=today_str)
    
    expected_expiry = (date.today() + timedelta(days=5)).strftime("%Y-%m-%d")
    assert expiry == expected_expiry
    
    remaining = shelf_service.calculate_remaining_days(expiry, start_date=today_str)
    assert remaining == 5

def test_predict_all_fruits():
    fruits = [
        {"fruit_type": "Mango", "grade": "Good", "defect_score": 0.0},
        {"fruit_type": "Pomegranate", "grade": "Better", "defect_score": 0.2}
    ]
    updated = shelf_service.predict_all_fruits(fruits)
    assert len(updated) == 2
    assert updated[0]["shelf_life_days"] == 7
    # Pomegranate Better is 20. 20 * 0.8 = 16
    assert updated[1]["shelf_life_days"] == 16
    assert "expiry_date" in updated[0]

def test_aggregate_shelf_life():
    fruits = [
        {"shelf_life_days": 10},
        {"shelf_life_days": 20}
    ]
    agg = shelf_service.aggregate_shelf_life(fruits)
    assert agg["average_shelf_life"] == "15 days"
    assert agg["breakdown"]["10 days"] == 1
    assert agg["breakdown"]["20 days"] == 1

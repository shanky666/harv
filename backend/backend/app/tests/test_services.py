import pytest
from app.services.grading_service import grading_service, GRADES
from app.services.shelf_service import shelf_service
from app.services.market_service import market_service


def test_grading_heuristic():
    result = grading_service._heuristic_grade(None, "mango")
    assert result["grade"] in GRADES
    assert 0 < result["confidence"] <= 1


def test_shelf_life():
    sl = shelf_service.predict_shelf_life("mango", "Good")
    assert "shelf_life_days" in sl
    assert "expiry_date" in sl
    assert sl["shelf_life_days"] == 7



def test_shelf_aggregate():
    fruits = [
        {"fruit_type": "mango", "grade": "Good", "shelf_life": "5-7 days"},
        {"fruit_type": "mango", "grade": "Medium", "shelf_life": "2-4 days"},
    ]
    result = shelf_service.aggregate_shelf_life(fruits)
    assert "average_shelf_life" in result
    assert "breakdown" in result


def test_market_recommendation():
    rec = market_service.recommend_market("mango", "Good")
    assert "best_market" in rec
    assert "expected_price" in rec
    assert "₹" in rec["expected_price"]


def test_market_aggregate():
    fruits = [
        {"fruit_type": "mango", "grade": "Good"},
        {"fruit_type": "mango", "grade": "Better"},
    ]
    rec = market_service.aggregate_recommendation(fruits)
    assert rec["best_market"] != ""


def test_detect_defects():
    d = grading_service.detect_defects("", "Reject")
    assert "rot" in d.lower() or "damage" in d.lower()

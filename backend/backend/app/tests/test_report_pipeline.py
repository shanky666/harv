import pytest
from unittest.mock import patch
from app.services.report_service import report_service

def test_generate_final_report_stats():
    fruits = [
        {"fruit_type": "Mango", "grade": "Good", "shelf_life_days": 7, "recommended_market": "Export"},
        {"fruit_type": "Mango", "grade": "Better", "shelf_life_days": 5, "recommended_market": "Supermarket"},
        {"fruit_type": "Pineapple", "grade": "Better", "shelf_life_days": 15, "recommended_market": "Supermarket"},
        {"fruit_type": "Pomegranate", "grade": "Medium", "shelf_life_days": 10, "recommended_market": "Local Market"},
        {"fruit_type": "Grapes", "grade": "Reject", "shelf_life_days": 1, "recommended_market": "Winery/Juice"}
    ]
    
    report = report_service.generate_final_report("SCAN_001", fruits)
    
    assert report["scan_id"] == "SCAN_001"
    assert report["total_fruits"] == 5
    assert report["good"] == 1
    assert report["better"] == 2
    assert report["medium"] == 1
    assert report["reject"] == 1
    # Average shelf life: (7+5+15+10+1)/5 = 38/5 = 7.6 -> round to 8 days
    assert report["average_shelf_life"] == "8 days"
    
    # Fruit distribution
    assert report["fruit_distribution"]["Mango"] == 2
    assert report["fruit_distribution"]["Pineapple"] == 1
    assert report["fruit_distribution"]["Pomegranate"] == 1
    assert report["fruit_distribution"]["Grapes"] == 1
    
    # Market distribution
    assert report["markets"]["Export"] == 1
    assert report["markets"]["Supermarket"] == 2
    assert report["markets"]["Local Market"] == 1
    assert report["markets"]["Winery/Juice"] == 1

def test_pdf_generation():
    fruits = [
        {"fruit_type": "Mango", "grade": "Good", "shelf_life_days": 7, "recommended_market": "Export"},
        {"fruit_type": "Pomegranate", "grade": "Reject", "shelf_life_days": 2, "recommended_market": "Processing"}
    ]
    
    grade_counts = {"good": 1, "better": 0, "medium": 0, "reject": 1}
    market = {"best_market": "Export", "expected_price": "₹150-₹200/kg"}
    
    with patch("app.services.report_service.get_report_path", return_value="test_report.pdf"), \
         patch("app.services.report_service.SimpleDocTemplate") as doc_mock:
         
        pdf_path = report_service.generate_pdf("SCAN_TEST", None, fruits, grade_counts, "4 days", market)
        assert pdf_path == "test_report.pdf"
        assert doc_mock.called

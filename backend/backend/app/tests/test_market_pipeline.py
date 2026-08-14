import pytest
from app.services.market_service import market_service

def test_recommend_market_from_config():
    # Mango Good -> Export
    rec_mango = market_service.recommend_market("Mango", "Good")
    assert rec_mango["recommended_market"] == "Export"
    assert "₹150-₹200/kg" in rec_mango["estimated_price"]
    
    # Pomegranate Reject -> Juice Industry
    rec_pom = market_service.recommend_market("Pomegranate", "Reject")
    assert rec_pom["recommended_market"] == "Juice Industry"
    
    # Pomegranate Better -> Retail
    rec_pom2 = market_service.recommend_market("Pomegranate", "Better")
    assert rec_pom2["recommended_market"] == "Retail"

def test_recommend_all_fruits():
    fruits = [
        {"fruit_type": "Mango", "grade": "Good"},
        {"fruit_type": "Pomegranate", "grade": "Reject"}
    ]
    updated = market_service.recommend_all_fruits(fruits)
    assert len(updated) == 2
    assert updated[0]["recommended_market"] == "Export"
    assert updated[1]["recommended_market"] == "Juice Industry"

def test_estimate_price():
    price_mango = market_service.estimate_price("Mango", "Good")
    assert price_mango == "₹150-₹200/kg"
    
    price_pom = market_service.estimate_price("Pomegranate", "Better")
    assert price_pom == "₹120-₹160/kg"

def test_aggregate_recommendation():
    # Basket with mostly Mango, predominantly Good grade
    fruits = [
        {"fruit_type": "Mango", "grade": "Good"},
        {"fruit_type": "Mango", "grade": "Better"},
        {"fruit_type": "Pomegranate", "grade": "Reject"}
    ]
    agg = market_service.aggregate_recommendation(fruits)
    # Dominant type: Mango. Dominant grade: Good (highest priority grade present)
    assert agg["recommended_market"] == "Export"

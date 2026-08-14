"""
Market Recommendation Service
Recommends the best market and estimates price based on fruit type, grade, and YAML configurations.
"""
from typing import Dict, List, Any, Optional
from loguru import logger
from app.services.fruit_registry_service import fruit_registry

class MarketRecommendationService:
    def recommend_market(self, fruit_type: str, grade: str, defect_score: float = 0.0) -> Dict[str, Any]:
        """
        Recommends best market and estimates price range.
        Grade classes: Better, Good, Reject (alphabetical for TF class ordering).
        """
        rules = fruit_registry.get_market_rules(fruit_type)
        if not rules:
            logger.warning(f"No market rules found in config for {fruit_type}. Using defaults.")
            rules = {
                "Better": {"market": "Retail", "price_range": "₹80-₹100/kg"},
                "Good": {"market": "Export", "price_range": "₹100-₹150/kg"},
                "Reject": {"market": "Processing", "price_range": "₹10-₹25/kg"}
            }
            
        rule_val = rules.get(grade, rules.get("Good", {"market": "Retail", "price_range": "₹60-₹80/kg"}))
        
        if isinstance(rule_val, dict):
            market = rule_val.get("market", "Retail")
            price = rule_val.get("price_range", "Unknown")
        else:
            market = str(rule_val)
            price = self.estimate_price(fruit_type, grade)
            
        return {
            "recommended_market": market,
            "estimated_price": price,
            "best_market": market,
            "expected_price": price,
            "alternatives": []
        }

    def recommend_all_fruits(self, fruits: List[Dict]) -> List[Dict]:
        for f in fruits:
            rec = self.recommend_market(
                fruit_type=f.get("fruit_type", "default"),
                grade=f.get("grade", "Good"),
                defect_score=f.get("defect_score", 0.0)
            )
            f["recommended_market"] = rec["recommended_market"]
            f["estimated_price"] = rec["estimated_price"]
            f["market_recommendation"] = rec["recommended_market"]
        return fruits

    def estimate_price(self, fruit_type: str, grade: str) -> str:
        rules = fruit_registry.get_market_rules(fruit_type)
        if rules:
            rule_val = rules.get(grade)
            if isinstance(rule_val, dict):
                return rule_val.get("price_range", "Unknown")
                
        default_prices = {
            "Better": "₹80-₹100/kg",
            "Good": "₹100-₹150/kg",
            "Reject": "₹10-₹25/kg"
        }
        return default_prices.get(grade, "₹30-₹50/kg")

    def get_dominant_grade(self, grade_counts: Dict[str, int]) -> str:
        priority = ["Good", "Better", "Reject"]
        for g in priority:
            if grade_counts.get(g, 0) > 0:
                return g
        return "Good"

    def aggregate_recommendation(self, fruits: List[Dict]) -> Dict[str, Any]:
        from collections import Counter
        fruit_types = [f.get("fruit_type", "default") for f in fruits]
        dominant_fruit = Counter(fruit_types).most_common(1)[0][0] if fruit_types else "default"
        grades = [f.get("grade", "Good") for f in fruits]
        grade_counts = Counter(grades)
        dominant_grade = self.get_dominant_grade(dict(grade_counts))
        return self.recommend_market(dominant_fruit, dominant_grade)

market_service = MarketRecommendationService()

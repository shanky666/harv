"""
Shelf Life Prediction Service
Estimates shelf life based on fruit type, grade, and defect score.
"""
import re
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from loguru import logger
from app.services.fruit_registry_service import fruit_registry

class ShelfLifeService:
    def predict_shelf_life(self, fruit_type: str, grade: str, defect_score: float = 0.0) -> Dict[str, Any]:
        """
        Predicts base and adjusted shelf life days and expiry date.
        Grade classes: Better, Good, Reject (alphabetical for TF class ordering).
        """
        rules = fruit_registry.get_shelf_life_rules(fruit_type)
        if not rules:
            logger.warning(f"No shelf life rules found in config for {fruit_type}. Using defaults.")
            rules = {"Better": 4, "Good": 5, "Reject": 1}
            
        base_days_val = rules.get(grade, rules.get("Good", 3))
        
        if isinstance(base_days_val, str):
            match = re.search(r'\d+', base_days_val)
            base_days = int(match.group()) if match else 3
        else:
            base_days = int(base_days_val)
            
        adjusted_days = base_days
        if defect_score > 0.0:
            adjusted_days = max(1, int(base_days * (1.0 - defect_score)))
            
        expiry_date = self.calculate_expiry_date(adjusted_days)
        
        return {
            "shelf_life_days": adjusted_days,
            "expiry_date": expiry_date
        }

    def calculate_remaining_days(self, expiry_date: str, start_date: str = None) -> int:
        if not start_date:
            start = date.today()
        else:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                start = date.today()
        try:
            expiry = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            return max(0, (expiry - start).days)
        except Exception:
            return 0

    def calculate_expiry_date(self, shelf_life_days: int, start_date: str = None) -> str:
        if not start_date:
            start = date.today()
        else:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                start = date.today()
        expiry = start + timedelta(days=shelf_life_days)
        return expiry.strftime("%Y-%m-%d")

    def predict_all_fruits(self, fruits: List[Dict]) -> List[Dict]:
        for f in fruits:
            pred = self.predict_shelf_life(
                fruit_type=f.get("fruit_type", "default"),
                grade=f.get("grade", "Good"),
                defect_score=f.get("defect_score", 0.0)
            )
            f["shelf_life_days"] = pred["shelf_life_days"]
            f["expiry_date"] = pred["expiry_date"]
            f["shelf_life"] = f"{pred['shelf_life_days']} days"
        return fruits

    def aggregate_shelf_life(self, fruits: List[Dict]) -> Dict[str, Any]:
        if not fruits:
            return {"average_shelf_life": "N/A", "breakdown": {}}
            
        total_days = 0
        valid_count = 0
        breakdown = {}
        
        for f in fruits:
            days = f.get("shelf_life_days")
            if days is None:
                sl_str = f.get("shelf_life", "")
                if sl_str:
                    match = re.search(r'\d+', str(sl_str))
                    if match:
                        days = int(match.group())
            
            if days is not None:
                total_days += days
                valid_count += 1
                lbl = f"{days} days"
                breakdown[lbl] = breakdown.get(lbl, 0) + 1
                
        avg_days = round(total_days / valid_count) if valid_count > 0 else 0
        avg_sl_str = f"{avg_days} days" if avg_days > 0 else "N/A"
        
        return {
            "average_shelf_life": avg_sl_str,
            "breakdown": breakdown
        }

shelf_service = ShelfLifeService()

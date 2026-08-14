"""
Fruit Passport Service
Generates digital passports for individual fruits.
"""
from typing import Dict, List
from datetime import datetime

class PassportService:
    def generate_passport(self, fruit: Dict, grade: str, defects: str,
                           shelf_life: str, market: str) -> Dict:
        """
        Legacy passport generation method.
        """
        return {
            "passport_id": str(__import__("uuid").uuid4()),
            "fruit_id": str(fruit.get("fruit_id", "")),
            "fruit_type": fruit.get("fruit_type", "Unknown"),
            "grade": grade,
            "defects": defects,
            "shelf_life": shelf_life,
            "market": market,
            "bounding_box": fruit.get("bbox", {}),
            "confidence": fruit.get("confidence", 0),
            "issued_at": datetime.utcnow().isoformat(),
            "issued_by": "HarvestLenz AI v1.0"
        }

    def generate_fruit_passport(self, fruit: Dict, scan_date: str = None) -> Dict:
        """
        Generates a standardized AI Fruit Passport for a single fruit.
        """
        if not scan_date:
            scan_date = datetime.utcnow().strftime("%Y-%m-%d")
            
        return {
            "passport_id": str(__import__("uuid").uuid4()),
            "fruit_id": fruit.get("fruit_id", ""),
            "fruit_type": fruit.get("fruit_type", "Unknown").capitalize(),
            "grade": fruit.get("grade", "Medium"),
            "confidence": float(fruit.get("confidence", 0.0)),
            "defect_score": float(fruit.get("defect_score", 0.0)),
            "shelf_life_days": int(fruit.get("shelf_life_days", 3)),
            "expiry_date": fruit.get("expiry_date", ""),
            "recommended_market": fruit.get("recommended_market", "Local Market"),
            "scan_date": scan_date,
            "crop_path": fruit.get("crop_path", "")
        }

    def generate_all_passports(self, fruits: List[Dict], scan_date: str = None) -> List[Dict]:
        """
        Generates AI Fruit Passports for all detected fruits.
        """
        return [self.generate_fruit_passport(f, scan_date) for f in fruits]

passport_service = PassportService()

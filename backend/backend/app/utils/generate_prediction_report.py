import os
import sys
import json
import uuid
from datetime import datetime

# Add directory paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend")))

from app.core.config import settings
from app.services.yolo_service import yolo_service
from app.services.crop_service import crop_service
from app.services.grading_service import grading_service
from app.services.shelf_service import shelf_service
from app.services.market_service import market_service
from app.services.report_service import report_service

def generate_report():
    print("Running prediction pipeline for production validation...")
    
    # Initialize services
    yolo_service.load_model()
    grading_service.load_cnn_model()
    
    sample_dir = "sample_data"
    img1_path = os.path.join(sample_dir, "sample_basket_01.jpg")
    img2_path = os.path.join(sample_dir, "sample_basket_02.jpg")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: Sample images not found in {sample_dir}. Please run generate_samples.py first.")
        return

    # Process Crate 1
    scan1_id = str(uuid.uuid4())
    print(f"Processing Crate 1: {img1_path}...")
    det1 = yolo_service.detect_fruits(img1_path, scan1_id)
    det1 = crop_service.crop_fruits(img1_path, det1, scan1_id)
    
    fruit_dicts1 = []
    for d in det1:
        grade_info = grading_service.grade_fruit(d["crop_path"], d["fruit_type"])
        shelf_res = shelf_service.predict_shelf_life(d["fruit_type"], grade_info["grade"], grade_info["defect_score"])
        market_res = market_service.recommend_market(d["fruit_type"], grade_info["grade"], grade_info["defect_score"])
        
        fruit_dicts1.append({
            "fruit_id": d["fruit_id"],
            "fruit_type": d["fruit_type"],
            "grade": grade_info["grade"],
            "confidence": grade_info["confidence"],
            "defect_score": grade_info["defect_score"],
            "defects": grade_info["defects"],
            "shelf_life_days": shelf_res["shelf_life_days"],
            "recommended_market": market_res["recommended_market"],
            "estimated_price": market_res["estimated_price"],
            "bbox": d["bbox"]
        })
        
    rep1 = report_service.generate_final_report(scan1_id, fruit_dicts1)
    
    # Process Crate 2
    scan2_id = str(uuid.uuid4())
    print(f"Processing Crate 2: {img2_path}...")
    det2 = yolo_service.detect_fruits(img2_path, scan2_id)
    det2 = crop_service.crop_fruits(img2_path, det2, scan2_id)
    
    fruit_dicts2 = []
    for d in det2:
        grade_info = grading_service.grade_fruit(d["crop_path"], d["fruit_type"])
        shelf_res = shelf_service.predict_shelf_life(d["fruit_type"], grade_info["grade"], grade_info["defect_score"])
        market_res = market_service.recommend_market(d["fruit_type"], grade_info["grade"], grade_info["defect_score"])
        
        fruit_dicts2.append({
            "fruit_id": d["fruit_id"],
            "fruit_type": d["fruit_type"],
            "grade": grade_info["grade"],
            "confidence": grade_info["confidence"],
            "defect_score": grade_info["defect_score"],
            "defects": grade_info["defects"],
            "shelf_life_days": shelf_res["shelf_life_days"],
            "recommended_market": market_res["recommended_market"],
            "estimated_price": market_res["estimated_price"],
            "bbox": d["bbox"]
        })
        
    rep2 = report_service.generate_final_report(scan2_id, fruit_dicts2)
    
    # Write prediction_report.md
    markdown_content = f"""# Production Model Pipeline Prediction Report

This report documents the results of executing the complete localized YOLOv8 and CNN grading pipeline on the two sample fruit crate scans.

---

## 1. Pipeline Execution Metadata
* **Timestamp**: {datetime.utcnow().isoformat()}Z
* **YOLOv8 Active Registry Version**: `v1` (`yolov8m` base)
* **CNN Classifier Registry Version**: `v1` (`MobileNetV3` transfer learning base)
* **Execution Status**: `SUCCESS`

---

## 2. Scan Analysis - Crate 01 (`sample_basket_01.jpg`)
* **Generated Scan ID**: `{scan1_id}`
* **Total Fruits Detected**: `{rep1['total_fruits']}`
* **Average Shelf Life**: `{rep1['average_shelf_life']}`

### A. Grade Distribution
| Grade | Count | Percentage |
|---|---|---|
| Good | {rep1['good']} | {round(rep1['good']/rep1['total_fruits']*100, 1)}% |
| Better | {rep1['better']} | {round(rep1['better']/rep1['total_fruits']*100, 1)}% |
| Medium | {rep1['medium']} | {round(rep1['medium']/rep1['total_fruits']*100, 1)}% |
| Reject | {rep1['reject']} | {round(rep1['reject']/rep1['total_fruits']*100, 1)}% |

### B. Detected Fruit Breakdown
| Fruit ID | Class Type | Bounding Box | Predicted Grade | Conf | Shelf Life | Target Market |
|---|---|---|---|---|---|---|
"""
    for f in fruit_dicts1:
        markdown_content += f"| `{f['fruit_id']}` | {f['fruit_type'].capitalize()} | `{f['bbox']}` | **{f['grade']}** | {round(f['confidence']*100, 1)}% | {f['shelf_life_days']} days | {f['recommended_market']} ({f['estimated_price']}) |\n"
        
    markdown_content += f"""
---

## 3. Scan Analysis - Crate 02 (`sample_basket_02.jpg`)
* **Generated Scan ID**: `{scan2_id}`
* **Total Fruits Detected**: `{rep2['total_fruits']}`
* **Average Shelf Life**: `{rep2['average_shelf_life']}`

### A. Grade Distribution
| Grade | Count | Percentage |
|---|---|---|
| Good | {rep2['good']} | {round(rep2['good']/rep2['total_fruits']*100, 1)}% |
| Better | {rep2['better']} | {round(rep2['better']/rep2['total_fruits']*100, 1)}% |
| Medium | {rep2['medium']} | {round(rep2['medium']/rep2['total_fruits']*100, 1)}% |
| Reject | {rep2['reject']} | {round(rep2['reject']/rep2['total_fruits']*100, 1)}% |

### B. Detected Fruit Breakdown
| Fruit ID | Class Type | Bounding Box | Predicted Grade | Conf | Shelf Life | Target Market |
|---|---|---|---|---|---|---|
"""
    for f in fruit_dicts2:
        markdown_content += f"| `{f['fruit_id']}` | {f['fruit_type'].capitalize()} | `{f['bbox']}` | **{f['grade']}** | {round(f['confidence']*100, 1)}% | {f['shelf_life_days']} days | {f['recommended_market']} ({f['estimated_price']}) |\n"

    report_path = "prediction_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Successfully generated prediction report under {report_path}!")

if __name__ == "__main__":
    generate_report()

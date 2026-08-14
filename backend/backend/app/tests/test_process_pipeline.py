import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from app.database.models import User, FruitScan, FruitType, DetectedFruit
from app.api.process import process_scan
from app.database.schemas import ProcessResponse

@pytest.mark.asyncio
async def test_process_scan_pipeline():
    db_mock = AsyncMock()
    current_user_mock = MagicMock(spec=User)
    scan_id = uuid.uuid4()
    
    # Mock scan model in DB
    mock_scan = MagicMock(spec=FruitScan)
    mock_scan.scan_id = scan_id
    mock_scan.image_path = "storage/uploads/test.jpg"
    
    # Mock fruit type in DB
    mock_fruit_type = MagicMock(spec=FruitType)
    mock_fruit_type.id = 1
    mock_fruit_type.name = "Mango"
    
    # Mock YOLO detections
    yolo_detections = [
        {"fruit_type": "mango", "bbox": [10, 20, 100, 110], "confidence": 0.92}
    ]
    
    # Mock Crop service detections output
    crop_detections = [
        {
            "fruit_type": "mango",
            "bbox": [10, 20, 100, 110],
            "confidence": 0.92,
            "crop_path": f"storage/crops/{scan_id}/FRUIT_0001.jpg"
        }
    ]
    
    # Mock Grading service output
    grade_info = {
        "grade": "Good",
        "confidence": 0.95,
        "defect_score": 0.05,
        "defects": [],
        "predicted_at": "2026-06-16T01:00:00Z"
    }

    with patch("app.api.process.get_scan", return_value=mock_scan), \
         patch("app.api.process.yolo_service.detect_fruits", return_value=yolo_detections), \
         patch("app.api.process.crop_service.crop_fruits", return_value=crop_detections), \
         patch("app.api.process.get_fruit_type_by_name", return_value=mock_fruit_type), \
         patch("app.api.process.bulk_create_fruits", return_value=None), \
         patch("app.api.process.update_scan_total", return_value=None), \
         patch("app.api.process.grading_service.grade_fruit", return_value=grade_info), \
         patch("app.api.process.update_fruit_grade", return_value=None), \
         patch("app.api.process.create_passport", return_value=None), \
         patch("app.api.process.create_report", return_value=None), \
         patch("app.api.process.report_service.generate_pdf", return_value="storage/reports/test.pdf"), \
         patch("os.path.exists", return_value=True):
             
        response = await process_scan(
            scan_id=scan_id,
            db=db_mock,
            current_user=current_user_mock
        )
        
        # Verify response dictionary values
        assert response["scan_id"] == str(scan_id)
        assert response["total_fruits"] == 1
        assert response["good"] == 1
        assert response["better"] == 0
        assert response["medium"] == 0
        assert response["reject"] == 0
        assert response["average_shelf_life"] == "6 days"

        assert response["fruit_distribution"]["Mango"] == 1
        assert response["markets"]["Export"] == 1
        
        # Verify db mock had expected calls
        assert db_mock.execute.called

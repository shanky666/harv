import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the database engine and sessionmaker before importing app
mock_engine = AsyncMock()
mock_sessionmaker = MagicMock()

with patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine), \
     patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=mock_sessionmaker):
    from app.main import app

import httpx
from app.database.models import User, FruitScan, DetectedFruit

@pytest.mark.asyncio
async def test_scan_pipeline_data_mapping():
    user_id = uuid.uuid4()
    scan_id = uuid.uuid4()
    
    mock_user = User(id=user_id, name="Farmer", email="farmer@example.com")
    mock_scan = FruitScan(scan_id=scan_id, user_id=user_id, image_path="path/to/img.jpg")
    
    # Simulate a basket of 3 fruits (2 Mangoes, 1 Pomegranate)
    mock_fruits = [
        DetectedFruit(
            fruit_id=uuid.uuid4(), scan_id=scan_id, fruit_type="Mango",
            bbox_x1=50.0, bbox_y1=50.0, bbox_x2=150.0, bbox_y2=150.0,
            confidence=0.94, crop_path="crop1.jpg", grade="Good",
            shelf_life="7 days", market_recommendation="Local Mandi"
        ),
        DetectedFruit(
            fruit_id=uuid.uuid4(), scan_id=scan_id, fruit_type="Mango",
            bbox_x1=200.0, bbox_y1=50.0, bbox_x2=300.0, bbox_y2=150.0,
            confidence=0.91, crop_path="crop2.jpg", grade="Reject",
            shelf_life="1 days", market_recommendation="Compost/Reject"
        ),
        DetectedFruit(
            fruit_id=uuid.uuid4(), scan_id=scan_id, fruit_type="Pomegranate",
            bbox_x1=100.0, bbox_y1=200.0, bbox_x2=200.0, bbox_y2=300.0,
            confidence=0.88, crop_path="crop3.jpg", grade="Better",
            shelf_life="14 days", market_recommendation="Premium Export"
        )
    ]

    with patch("app.middleware.authentication.decode_token", return_value={"sub": str(user_id), "type": "access"}), \
         patch("app.middleware.authentication.get_user_by_id", return_value=mock_user), \
         patch("app.api.report.get_scan", return_value=mock_scan), \
         patch("app.api.report.get_fruits_by_scan", return_value=mock_fruits), \
         patch("app.api.report.get_report_by_scan", return_value=None), \
         patch("app.api.report.report_service.generate_pdf", return_value="storage/uploads/reports/test_report.pdf"), \
         patch("app.api.report.create_report", return_value=None):
             
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            headers = {"Authorization": "Bearer fake_token"}

            # Fetch the aggregate report and verify counts
            resp = await client.get(f"/scan/report/{scan_id}", headers=headers)
            assert resp.status_code == 200
            
            data = resp.json()
            assert data["scan_id"] == str(scan_id)
            assert data["total_fruits"] == 3
            
            # Grade Counter assertions
            assert data["grades"]["good"] == 1
            assert data["grades"]["better"] == 1
            assert data["grades"]["reject"] == 1
            assert "medium" not in data["grades"] or data["grades"]["medium"] == 0

            # Shelf life aggregation verification
            assert "shelf_life" in data
            # Market recommendation verification
            assert "market" in data

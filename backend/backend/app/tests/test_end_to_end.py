import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import ExitStack

# Mock the database engine and sessionmaker before importing app
mock_engine = AsyncMock()
mock_sessionmaker = MagicMock()

with patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine), \
     patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=mock_sessionmaker):
    from app.main import app

import httpx
from app.database.models import User, FruitScan, DetectedFruit, FruitPassport, Report

@pytest.mark.asyncio
async def test_complete_end_to_end_flow():
    user_id = uuid.uuid4()
    scan_id = uuid.uuid4()
    fruit_id = uuid.uuid4()
    passport_id = uuid.uuid4()
    
    mock_user = User(
        id=user_id,
        name="John Doe",
        email="farmer.john@example.com",
        phone="+919876543210",
        password_hash="hashed_password",
        location="Nashik",
        created_at=datetime.utcnow()
    )
    
    mock_scan = FruitScan(
        scan_id=scan_id,
        user_id=user_id,
        image_path="storage/uploads/original/test_basket.jpg",
        total_fruits=0,
        scan_date=datetime.utcnow()
    )
    
    mock_detected_fruit = DetectedFruit(
        fruit_id=fruit_id,
        scan_id=scan_id,
        fruit_type="Mango",
        bbox_x1=10.0,
        bbox_y1=20.0,
        bbox_x2=100.0,
        bbox_y2=120.0,
        confidence=0.95,
        crop_path=f"storage/crops/{scan_id}/FRUIT_0001.jpg",
        grade="Good",
        shelf_life="8 days",
        market_recommendation="Export Hub"
    )
    
    mock_passport = FruitPassport(
        passport_id=passport_id,
        fruit_id=fruit_id,
        grade="Good",
        defects="None",
        shelf_life="8 days",
        market="Export Hub",
        created_at=datetime.utcnow()
    )
    
    mock_report = Report(
        report_id=uuid.uuid4(),
        scan_id=scan_id,
        pdf_path="storage/uploads/reports/test_report.pdf",
        summary='{"good": 1, "better": 0, "medium": 0, "reject": 0}',
        created_at=datetime.utcnow()
    )

    # Mock service responses
    yolo_detections = [{"fruit_type": "Mango", "bbox": [10, 20, 100, 120], "confidence": 0.95}]
    crop_detections = [{"fruit_type": "Mango", "bbox": [10, 20, 100, 120], "confidence": 0.95, "crop_path": f"storage/crops/{scan_id}/FRUIT_0001.jpg"}]
    grade_info = {"grade": "Good", "confidence": 0.95, "defect_score": 0.0, "defects": [], "predicted_at": "2026-06-16T01:00:00Z"}
    shelf_life_res = {"shelf_life_days": 8, "expiry_date": "2026-06-24"}
    market_rec = {
        "recommended_market": "Export Hub",
        "estimated_price": "₹1500",
        "best_market": "Export Hub",
        "expected_price": "₹1500",
        "alternatives": []
    }

    with ExitStack() as stack:
        stack.enter_context(patch("app.api.auth.get_user_by_email", return_value=None))
        stack.enter_context(patch("app.api.auth.create_user", return_value=mock_user))
        stack.enter_context(patch("app.middleware.authentication.decode_token", return_value={"sub": str(user_id), "type": "access"}))
        stack.enter_context(patch("app.middleware.authentication.get_user_by_id", return_value=mock_user))
        stack.enter_context(patch("app.api.upload.save_upload", return_value="storage/uploads/original/test_basket.jpg"))
        stack.enter_context(patch("app.api.upload.create_scan", return_value=mock_scan))
        stack.enter_context(patch("app.api.process.get_scan", return_value=mock_scan))
        stack.enter_context(patch("app.api.process.get_fruits_by_scan", return_value=[mock_detected_fruit]))
        stack.enter_context(patch("app.api.process.get_passport_by_fruit", return_value=None))
        stack.enter_context(patch("app.api.process.get_report_by_scan", return_value=None))
        stack.enter_context(patch("app.api.process.yolo_service.detect_fruits", return_value=yolo_detections))
        stack.enter_context(patch("app.api.process.crop_service.crop_fruits", return_value=crop_detections))
        stack.enter_context(patch("app.api.process.get_fruit_type_by_name", return_value=MagicMock()))
        stack.enter_context(patch("app.api.process.bulk_create_fruits", return_value=None))
        stack.enter_context(patch("app.api.process.update_scan_total", return_value=None))
        stack.enter_context(patch("app.api.process.grading_service.grade_fruit", return_value=grade_info))
        stack.enter_context(patch("app.api.process.shelf_service.predict_shelf_life", return_value=shelf_life_res))
        stack.enter_context(patch("app.api.process.market_service.recommend_market", return_value=market_rec))
        stack.enter_context(patch("app.api.process.update_fruit_grade", return_value=None))
        stack.enter_context(patch("app.api.process.create_passport", return_value=mock_passport))
        stack.enter_context(patch("app.api.process.report_service.generate_pdf", return_value="storage/uploads/reports/test_report.pdf"))
        stack.enter_context(patch("app.api.process.create_report", return_value=mock_report))
        stack.enter_context(patch("app.api.report.get_scan", return_value=mock_scan))
        stack.enter_context(patch("app.api.report.get_fruits_by_scan", return_value=[mock_detected_fruit]))
        stack.enter_context(patch("app.api.report.get_report_by_scan", return_value=mock_report))
        stack.enter_context(patch("app.api.passport.get_fruit_by_id", return_value=mock_detected_fruit))
        stack.enter_context(patch("app.api.passport.get_passport_by_fruit", return_value=mock_passport))
        stack.enter_context(patch("app.api.users.update_user", return_value=mock_user))
        stack.enter_context(patch("os.path.exists", return_value=True))

        async with httpx.AsyncClient(app=app, base_url="http://test") as client:

            headers = {"Authorization": "Bearer fake_token"}

            # 1. Register
            reg_resp = await client.post("/auth/register", json={
                "name": "John Doe",
                "email": "farmer.john@example.com",
                "password": "securepassword123",
                "phone": "+919876543210",
                "location": "Nashik"
            })
            assert reg_resp.status_code == 201
            assert reg_resp.json()["email"] == "farmer.john@example.com"

            # 2. Login
            with patch("app.api.auth.get_user_by_email", return_value=mock_user), \
                 patch("app.api.auth.verify_password", return_value=True):
                login_resp = await client.post("/auth/login", json={
                    "email": "farmer.john@example.com",
                    "password": "securepassword123"
                })
                assert login_resp.status_code == 200
                assert "access_token" in login_resp.json()

            # 3. Get Current User Info
            me_resp = await client.get("/auth/me", headers=headers)
            assert me_resp.status_code == 200
            assert me_resp.json()["name"] == "John Doe"

            # 4. Upload Image
            upload_resp = await client.post(
                "/scan/upload",
                headers=headers,
                files={"file": ("basket.jpg", b"imagebytes", "image/jpeg")}
            )
            assert upload_resp.status_code == 201
            assert upload_resp.json()["scan_id"] == str(scan_id)

            # 5. Process Scan
            proc_resp = await client.post(f"/scan/process/{scan_id}", headers=headers)
            assert proc_resp.status_code == 200
            data = proc_resp.json()
            assert data["scan_id"] == str(scan_id)
            assert data["total_fruits"] == 1
            assert data["good"] == 1

            # 6. Get Quality Report
            report_resp = await client.get(f"/scan/report/{scan_id}", headers=headers)
            assert report_resp.status_code == 200
            assert report_resp.json()["total_fruits"] == 1
            assert report_resp.json()["pdf_url"] == f"/scan/report/{scan_id}/pdf"

            # 7. Get Fruit Passport
            passport_resp = await client.get(f"/scan/passport/{fruit_id}", headers=headers)
            assert passport_resp.status_code == 200
            assert passport_resp.json()["fruit_type"] == "Mango"
            assert passport_resp.json()["grade"] == "Good"

            # 8. Get Profile
            profile_resp = await client.get("/users/profile", headers=headers)
            assert profile_resp.status_code == 200
            assert profile_resp.json()["email"] == "farmer.john@example.com"

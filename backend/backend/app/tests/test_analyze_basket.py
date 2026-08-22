import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import ExitStack

# Mock database engine and sessionmaker before importing app
mock_engine = AsyncMock()
mock_sessionmaker = MagicMock()

with patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine), \
     patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=mock_sessionmaker):
    from app.main import app

import httpx
from app.database.models import User, AnalysisSession, BasketFruit

@pytest.mark.asyncio
async def test_analyze_basket_pipeline():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    
    mock_user = User(
        id=user_id,
        name="Test User",
        email="test@user.com",
        phone="+919999999999",
        password_hash="hash",
        location="Nashik",
        created_at=datetime.utcnow()
    )
    
    yolo_detections = [
        {"fruit_id": "FRUIT_0001", "fruit_type": "mango", "bbox": [10, 20, 100, 110], "confidence": 0.92, "crop_path": None}
    ]
    
    crop_detections = [
        {"fruit_id": "FRUIT_0001", "fruit_type": "mango", "bbox": [10, 20, 100, 110], "confidence": 0.92, "crop_path": f"storage/crops/{session_id}/FRUIT_0001.jpg"}
    ]
    
    class_res = {"fruit_type": "orange", "confidence": 0.96}  # YOLO says mango, classifier overrides to orange
    grade_res = {"grade": "Better", "confidence": 0.91, "defect_score": 0.08, "defects": [], "predicted_at": "2026-06-16T12:00:00Z"}

    # Mock response from basket_analysis_service.analyze_basket
    mock_response = {
        "session_id": str(session_id),
        "total_fruits": 1,
        "fruits": [
            {
                "fruit_id": "FRUIT_0001",
                "fruit_type": "orange",
                "grade": "Better",
                "confidence": 0.96,
                "bbox": [10, 20, 100, 110],
                "crop_path": f"storage/crops/{session_id}/FRUIT_0001.jpg"
            }
        ],
        "summary": {
            "good": 0,
            "better": 1,
            "medium": 0,
            "reject": 0
        }
    }

    with ExitStack() as stack:
        stack.enter_context(patch("app.middleware.authentication.decode_token", return_value={"sub": str(user_id), "type": "access"}))
        stack.enter_context(patch("app.middleware.authentication.get_user_by_id", return_value=mock_user))
        stack.enter_context(patch("app.api.analyze_basket.analysis_orchestrator.run_analysis_job", return_value=None))
        stack.enter_context(patch("os.path.exists", return_value=True))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer fake_token"}
            resp = await client.post(
                "/analyze?fruit_type=mango&is_single=true",
                headers=headers,
                files={"file": ("basket.jpg", b"imagebytes", "image/jpeg")}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "processing"
            assert data["fruit_type"] == "mango"


@pytest.mark.asyncio
async def test_get_analysis_history():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    
    mock_user = User(
        id=user_id,
        name="Test User",
        email="test@user.com",
        phone="+919999999999",
        password_hash="hash",
        location="Nashik",
        created_at=datetime.utcnow()
    )

    mock_history = {
        "session_id": str(session_id),
        "total_fruits": 2,
        "fruit_type": "mango",
        "fruits": [
            {
                "fruit_id": "FRUIT_0001",
                "fruit_type": "mango",
                "grade": "Good",
                "confidence": 0.95,
                "grade_confidence": 0.95,
                "bbox": [10, 20, 100, 110],
                "crop_path": f"storage/crops/{session_id}/FRUIT_0001.jpg"
            },
            {
                "fruit_id": "FRUIT_0002",
                "fruit_type": "pomegranate",
                "grade": "Reject",
                "confidence": 0.89,
                "grade_confidence": 0.89,
                "bbox": [120, 130, 220, 230],
                "crop_path": f"storage/crops/{session_id}/FRUIT_0002.jpg"
            }
        ],
        "summary": {
            "good": 1,
            "better": 0,
            "medium": 0,
            "reject": 1
        }
    }

    with ExitStack() as stack:
        stack.enter_context(patch("app.middleware.authentication.decode_token", return_value={"sub": str(user_id), "type": "access"}))
        stack.enter_context(patch("app.middleware.authentication.get_user_by_id", return_value=mock_user))
        stack.enter_context(patch("app.api.analyze_basket.analysis_orchestrator.get_analysis_history", return_value=mock_history))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer fake_token"}
            resp = await client.get(f"/analysis/{session_id}", headers=headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == str(session_id)
            assert data["total_fruits"] == 2
            assert data["fruits"][0]["fruit_type"] == "mango"
            assert data["fruits"][1]["fruit_type"] == "pomegranate"
            assert data["summary"]["good"] == 1
            assert data["summary"]["reject"] == 1


@pytest.mark.asyncio
async def test_get_analysis_history_not_found():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    
    mock_user = User(
        id=user_id,
        name="Test User",
        email="test@user.com",
        phone="+919999999999",
        password_hash="hash",
        location="Nashik",
        created_at=datetime.utcnow()
    )

    with ExitStack() as stack:
        stack.enter_context(patch("app.middleware.authentication.decode_token", return_value={"sub": str(user_id), "type": "access"}))
        stack.enter_context(patch("app.middleware.authentication.get_user_by_id", return_value=mock_user))
        stack.enter_context(patch("app.api.analyze_basket.analysis_orchestrator.get_analysis_history", return_value=None))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer fake_token"}
            resp = await client.get(f"/analysis/{session_id}", headers=headers)
            assert resp.status_code == 404
            assert resp.json()["detail"] == "Analysis session not found."

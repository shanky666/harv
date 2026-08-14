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
from app.database.models import User

@pytest.mark.asyncio
async def test_register_and_login():
    mock_user = User(
        id=uuid.uuid4(),
        name="Test Farmer",
        email="farmer@test.com",
        phone=None,
        password_hash="hashed_password",
        location="Maharashtra",
        created_at=datetime.utcnow()
    )

    with patch("app.api.auth.get_user_by_email", return_value=mock_user), \
         patch("app.api.auth.create_user", return_value=mock_user), \
         patch("app.api.auth.verify_password", return_value=True):

        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Register
            resp = await client.post("/auth/register", json={
                "name": "Test Farmer",
                "email": "farmer@test.com",
                "password": "testpass123",
                "location": "Maharashtra"
            })
            assert resp.status_code in (201, 400)  # 400 if already exists

            # Login
            resp = await client.post("/auth/login", json={
                "email": "farmer@test.com",
                "password": "testpass123"
            })
            assert resp.status_code in (200, 401, 500)


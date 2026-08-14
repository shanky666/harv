import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import get_fruits_by_scan, get_scan
from app.database.schemas import MarketResponse
from app.middleware.authentication import get_current_user
from app.database.models import User
from app.services.market_service import market_service

router = APIRouter(prefix="/scan", tags=["Market"])

@router.post("/market/{scan_id}", response_model=MarketResponse,
             summary="Get best market recommendation for this scan")
async def market_recommendation(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    scan = await get_scan(db, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found")
    fruits = await get_fruits_by_scan(db, scan_id)
    fruit_dicts = [{"fruit_type": f.fruit_type, "grade": f.grade} for f in fruits]
    rec = market_service.aggregate_recommendation(fruit_dicts)
    return MarketResponse(scan_id=scan_id, **rec)

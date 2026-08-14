import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import get_fruits_by_scan, get_scan
from app.database.schemas import ShelfLifeResponse
from app.middleware.authentication import get_current_user
from app.database.models import User
from app.services.shelf_service import shelf_service

router = APIRouter(prefix="/scan", tags=["Shelf Life"])

@router.post("/shelf-life/{scan_id}", response_model=ShelfLifeResponse,
             summary="Predict aggregated shelf life for the scan")
async def predict_shelf_life(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    scan = await get_scan(db, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found")
    fruits = await get_fruits_by_scan(db, scan_id)
    fruit_dicts = [{"fruit_type": f.fruit_type, "grade": f.grade,
                    "shelf_life": f.shelf_life} for f in fruits]
    result = shelf_service.aggregate_shelf_life(fruit_dicts)
    return ShelfLifeResponse(scan_id=scan_id, **result)

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import get_fruit_by_id, create_passport, get_passport_by_fruit
from app.database.schemas import PassportResponse
from app.middleware.authentication import get_current_user
from app.database.models import User
from app.services.grading_service import grading_service
from app.services.passport_service import passport_service

router = APIRouter(prefix="/scan", tags=["Passport"])

@router.get("/passport/{fruit_id}", response_model=PassportResponse,
            summary="Get AI Fruit Passport for a single detected fruit")
async def get_passport(
    fruit_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    fruit = await get_fruit_by_id(db, fruit_id)
    if not fruit:
        raise HTTPException(404, "Fruit not found")

    existing = await get_passport_by_fruit(db, fruit_id)
    if existing:
        return PassportResponse(
            passport_id=existing.passport_id, fruit_id=fruit_id,
            fruit_type=fruit.fruit_type, grade=existing.grade,
            defects=existing.defects, shelf_life=existing.shelf_life,
            market=existing.market, created_at=existing.created_at
        )

    grade = fruit.grade or "Medium"
    defects = grading_service.detect_defects(fruit.crop_path or "", grade)
    shelf_life = fruit.shelf_life or "3-5 days"
    market = fruit.market_recommendation or "Local Mandi"

    passport = await create_passport(db, fruit_id, grade, defects, shelf_life, market)
    return PassportResponse(
        passport_id=passport.passport_id, fruit_id=fruit_id,
        fruit_type=fruit.fruit_type, grade=grade,
        defects=defects, shelf_life=shelf_life,
        market=market, created_at=passport.created_at
    )

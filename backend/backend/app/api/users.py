from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.schemas import UserOut, UserUpdate
from app.database.crud import update_user
from app.middleware.authentication import get_current_user
from app.database.models import User

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/profile", response_model=UserOut)
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/profile", response_model=UserOut)
async def update_profile(body: UserUpdate, db: AsyncSession = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    updated = await update_user(db, current_user.id, **body.model_dump(exclude_none=True))
    return updated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.schemas import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserOut
from app.database.crud import create_user, get_user_by_email, get_user_by_id
from app.core.security import verify_password, create_access_token, create_refresh_token, decode_token
from app.middleware.authentication import get_current_user
from app.database.models import User
import uuid

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserOut, status_code=201,
             summary="Register a new farmer account")
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = await create_user(db, body.name, body.email, body.password, body.phone, body.location)
    return user

@router.post("/login", response_model=TokenResponse, summary="Login and get JWT tokens")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access = create_access_token({"sub": str(user.id)})
    refresh = create_refresh_token({"sub": str(user.id)})
    return TokenResponse(access_token=access, refresh_token=refresh)

@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh_token(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user_id = payload.get("sub")
    access = create_access_token({"sub": user_id})
    refresh = create_refresh_token({"sub": user_id})
    return TokenResponse(access_token=access, refresh_token=refresh)

@router.get("/me", response_model=UserOut, summary="Get current user info")
async def me(current_user: User = Depends(get_current_user)):
    return current_user

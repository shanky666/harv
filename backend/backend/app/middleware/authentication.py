import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.security import decode_token
from app.database.connection import get_db
from app.database.crud import get_user_by_id
from app.database.models import User

bearer_scheme = HTTPBearer()

class OptionalHTTPBearer(HTTPBearer):
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        if settings.DEMO_MODE:
            try:
                return await super().__call__(request)
            except HTTPException:
                return None
        return await super().__call__(request)

optional_bearer_scheme = OptionalHTTPBearer(auto_error=False)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    token = credentials.credentials
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = await get_user_by_id(db, uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

async def get_current_user_demo(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    if settings.DEMO_MODE and not credentials:
        return None
        
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )
        
    token = credentials.credentials
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = await get_user_by_id(db, uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

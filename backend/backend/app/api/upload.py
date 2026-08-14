from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import create_scan
from app.database.schemas import ScanOut
from app.middleware.authentication import get_current_user
from app.database.models import User
from app.utils.file_utils import save_upload

router = APIRouter(prefix="/scan", tags=["Scan"])

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

@router.post("/upload", response_model=ScanOut, status_code=201,
             summary="Upload a basket/crate image for scanning")
async def upload_image(
    file: UploadFile = File(..., description="Fruit basket image (JPEG/PNG)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    image_path = await save_upload(file, "original")
    scan = await create_scan(db, current_user.id, image_path)
    return scan

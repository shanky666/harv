import os
import shutil
import aiofiles
from app.core.config import settings

async def save_upload(file, subfolder: str = "original") -> str:
    """Save an uploaded file. `file` can be a FastAPI UploadFile or any file-like with .filename and .read()."""
    ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "jpg"
    filename = f"{__import__('uuid').uuid4().hex}.{ext}"
    dest_dir = os.path.join(settings.STORAGE_BASE_PATH, "uploads", subfolder)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    async with aiofiles.open(dest_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    from app.utils.image_utils import resize_image_to_limit
    resize_image_to_limit(dest_path, max_size=1280)
    
    return dest_path

def get_crop_dir(scan_id: str) -> str:
    path = os.path.join(settings.STORAGE_BASE_PATH, "crops", scan_id)
    os.makedirs(path, exist_ok=True)
    return path

def get_report_path(scan_id: str) -> str:
    path = os.path.join(settings.STORAGE_BASE_PATH, "uploads", "reports")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"report_{scan_id}.pdf")

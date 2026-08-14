import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import get_scan, bulk_create_fruits, update_scan_total
from app.database.models import DetectedFruit
from app.database.schemas import DetectionResponse
from app.middleware.authentication import get_current_user
from app.database.models import User
from app.services.yolo_service import yolo_service
from app.services.crop_service import crop_service

router = APIRouter(prefix="/scan", tags=["Detection"])

@router.post("/detect/{scan_id}", response_model=DetectionResponse,
             summary="Run YOLOv8 detection and crop all fruits in a basket/crate")
async def detect_fruits(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Fetch scan information
    scan = await get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # 2. Run YOLOv8 detection
    detections = yolo_service.detect_fruits(scan.image_path, str(scan_id))

    # 3. Crop every detected fruit (ROI extraction)
    detections = crop_service.crop_fruits(scan.image_path, detections, str(scan_id))

    # 4. Save metadata for each fruit in PostgreSQL database
    db_fruits = []
    for det in detections:
        bb = det["bbox"]
        # Generate a new UUID for database integrity (primary key)
        db_fruit_id = uuid.uuid4()
        
        # Fetch fruit type ID from DB
        from app.database.crud import get_fruit_type_by_name
        db_fruit_type = await get_fruit_type_by_name(db, det["fruit_type"])
        fruit_type_id = db_fruit_type.id if db_fruit_type else None
        
        fruit = DetectedFruit(
            fruit_id=db_fruit_id,
            scan_id=scan_id,
            fruit_type_id=fruit_type_id,
            fruit_type=det["fruit_type"],
            bbox_x1=float(bb[0]), 
            bbox_y1=float(bb[1]),
            bbox_x2=float(bb[2]), 
            bbox_y2=float(bb[3]),
            confidence=det["confidence"],
            crop_path=det.get("crop_path")  # Storing absolute crop path
        )
        db_fruits.append(fruit)

    await bulk_create_fruits(db, db_fruits)
    await update_scan_total(db, scan_id, len(db_fruits))

    # 5. Format and return the JSON response using yolo_service
    response_data = yolo_service.create_detection_response(str(scan_id), detections)
    return response_data

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import get_scan, get_fruits_by_scan, create_report, get_report_by_scan
from app.database.schemas import ReportResponse
from typing import Optional
from app.middleware.authentication import get_current_user, get_current_user_demo
from app.database.models import User
from app.services.report_service import report_service
from app.services.shelf_service import shelf_service
from app.services.market_service import market_service
from collections import Counter

router = APIRouter(prefix="/scan", tags=["Report"])


@router.get("/report/{scan_id}", response_model=ReportResponse,
            summary="Get full quality report for a scan")
async def get_report(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = await get_scan(db, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found")
    fruits = await get_fruits_by_scan(db, scan_id)
    fruit_dicts = [
        {
            "fruit_type": f.fruit_type, "grade": f.grade,
            "shelf_life": f.shelf_life, "market_recommendation": f.market_recommendation,
            "fruit_id": str(f.fruit_id),
            "bbox": [f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2],
        }
        for f in fruits
    ]
    grade_counts = dict(Counter(f["grade"].lower() for f in fruit_dicts if f["grade"]))
    shelf = shelf_service.aggregate_shelf_life(fruit_dicts)
    market = market_service.aggregate_recommendation(fruit_dicts)

    existing = await get_report_by_scan(db, scan_id)
    if not existing:
        annotated_path = None
        processed_dir = "storage/uploads/processed"
        candidate = f"{processed_dir}/{scan_id}_annotated.jpg"
        if os.path.exists(candidate):
            annotated_path = candidate

        pdf_path = report_service.generate_pdf(
            str(scan_id), scan.image_path, fruit_dicts, grade_counts,
            shelf["average_shelf_life"], market, annotated_path=annotated_path,
        )
        summary = str(report_service.generate_json(
            str(scan_id), fruit_dicts, grade_counts,
            shelf["average_shelf_life"], market,
        ))
        await create_report(db, scan_id, pdf_path, summary)

    pdf_url = f"/scan/report/{scan_id}/pdf"
    return ReportResponse(
        scan_id=scan_id, total_fruits=len(fruits),
        grades=grade_counts, shelf_life=shelf["average_shelf_life"],
        market=market["best_market"], pdf_url=pdf_url,
        created_at=datetime.utcnow(),
    )


@router.get("/report/{scan_id}/pdf", summary="Download PDF report")
async def download_pdf(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_demo),
):
    report = await get_report_by_scan(db, scan_id)
    if not report or not report.pdf_path:
        raise HTTPException(404, "PDF not generated yet — call GET /report first")
    return FileResponse(
        report.pdf_path, media_type="application/pdf",
        filename=f"harvestlenz_report_{scan_id}.pdf",
    )


import os
import io
import csv
from fastapi.responses import StreamingResponse


@router.get("/report/{scan_id}/csv", summary="Download CSV report")
async def download_csv(
    scan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_demo),
):
    scan = await get_scan(db, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found")
    fruits = await get_fruits_by_scan(db, scan_id)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Fruit ID", "Fruit Type", "Grade", "Confidence %",
        "Shelf Life (Days)", "Market Recommendation",
        "Estimated Price (INR)", "BBox X1", "BBox Y1", "BBox X2", "BBox Y2",
    ])

    from app.services.shelf_service import shelf_service as shelf_svc
    from app.services.market_service import market_service as market_svc

    for f in fruits:
        grade = f.grade or "Good"
        shelf_info = shelf_svc.predict_shelf_life(f.fruit_type, grade)
        market_info = market_svc.recommend_market(f.fruit_type, grade)

        writer.writerow([
            f.fruit_id, f.fruit_type.capitalize(), grade,
            round(f.confidence * 100.0, 1) if f.confidence else 0.0,
            shelf_info["shelf_life_days"], market_info["recommended_market"],
            market_info["estimated_price"],
            f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=harvestlenz_report_{scan_id}.csv"},
    )

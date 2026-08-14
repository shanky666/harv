import uuid
from typing import Optional, List, Union, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from .models import User, FruitScan, DetectedFruit, Report, FruitPassport, FruitType
from app.core.security import hash_password


# ─── Users ───────────────────────────────────────────────
async def create_user(db: AsyncSession, name: str, email: str, password: str,
                      phone: Optional[str] = None, location: Optional[str] = None) -> User:
    user = User(name=name, email=email, password_hash=hash_password(password),
                phone=phone, location=location)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def update_user(db: AsyncSession, user_id: uuid.UUID, **kwargs) -> Optional[User]:
    await db.execute(update(User).where(User.id == user_id).values(**kwargs))
    await db.commit()
    return await get_user_by_id(db, user_id)


# ─── Scans ───────────────────────────────────────────────
async def create_scan(db: AsyncSession, user_id: uuid.UUID, image_path: str) -> FruitScan:
    scan = FruitScan(user_id=user_id, image_path=image_path)
    db.add(scan)
    await db.commit()
    await db.refresh(scan)
    return scan

async def get_scan(db: AsyncSession, scan_id: uuid.UUID) -> Optional[FruitScan]:
    result = await db.execute(select(FruitScan).where(FruitScan.scan_id == scan_id))
    return result.scalar_one_or_none()

async def update_scan_total(db: AsyncSession, scan_id: uuid.UUID, total: int) -> None:
    await db.execute(update(FruitScan).where(FruitScan.scan_id == scan_id).values(total_fruits=total))
    await db.commit()


# ─── Detected Fruits ─────────────────────────────────────
async def bulk_create_fruits(db: AsyncSession, fruits: list) -> List[DetectedFruit]:
    db.add_all(fruits)
    await db.commit()
    return fruits

async def get_fruits_by_scan(db: AsyncSession, scan_id: uuid.UUID) -> List[DetectedFruit]:
    result = await db.execute(select(DetectedFruit).where(DetectedFruit.scan_id == scan_id))
    return list(result.scalars().all())

async def get_fruit_by_id(db: AsyncSession, fruit_id: Union[uuid.UUID, str]) -> Optional[DetectedFruit]:
    if isinstance(fruit_id, uuid.UUID):
        result = await db.execute(select(DetectedFruit).where(DetectedFruit.fruit_id == fruit_id))
        return result.scalar_one_or_none()
    
    try:
        val = uuid.UUID(str(fruit_id))
        result = await db.execute(select(DetectedFruit).where(DetectedFruit.fruit_id == val))
        return result.scalar_one_or_none()
    except ValueError:
        # Fallback to look up by sequential ID mapped to crop_path suffix (e.g. /FRUIT_0001.jpg)
        result = await db.execute(
            select(DetectedFruit)
            .where(DetectedFruit.crop_path.like(f"%/{fruit_id}.jpg"))
            .order_by(DetectedFruit.created_at.desc())
        )
        return result.scalars().first()

async def update_fruit_grade(db: AsyncSession, fruit_id: Union[uuid.UUID, str], grade: str,
                              grade_confidence: float, defect_score: float,
                              shelf_life: str, market: str, predicted_at: Any) -> None:
    fruit = await get_fruit_by_id(db, fruit_id)
    if fruit:
        await db.execute(
            update(DetectedFruit).where(DetectedFruit.fruit_id == fruit.fruit_id).values(
                grade=grade,
                grade_confidence=grade_confidence,
                defect_score=defect_score,
                predicted_at=predicted_at,
                shelf_life=shelf_life,
                market_recommendation=market
            )
        )
        await db.commit()


# ─── Reports ─────────────────────────────────────────────
async def create_report(db: AsyncSession, scan_id: uuid.UUID, pdf_path: str, summary: str) -> Report:
    report = Report(scan_id=scan_id, pdf_path=pdf_path, summary=summary)
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return report

async def get_report_by_scan(db: AsyncSession, scan_id: uuid.UUID) -> Optional[Report]:
    result = await db.execute(select(Report).where(Report.scan_id == scan_id))
    return result.scalar_one_or_none()


# ─── Passports ───────────────────────────────────────────
async def create_passport(db: AsyncSession, fruit_id: Union[uuid.UUID, str], grade: str,
                           defects: str, shelf_life: str, market: str) -> FruitPassport:
    fruit = await get_fruit_by_id(db, fruit_id)
    resolved_id = fruit.fruit_id if fruit else (uuid.UUID(str(fruit_id)) if isinstance(fruit_id, uuid.UUID) else uuid.uuid4())
    passport = FruitPassport(fruit_id=resolved_id, grade=grade, defects=defects,
                              shelf_life=shelf_life, market=market)
    db.add(passport)
    await db.commit()
    await db.refresh(passport)
    return passport

async def get_passport_by_fruit(db: AsyncSession, fruit_id: Union[uuid.UUID, str]) -> Optional[FruitPassport]:
    fruit = await get_fruit_by_id(db, fruit_id)
    if not fruit:
        return None
    result = await db.execute(select(FruitPassport).where(FruitPassport.fruit_id == fruit.fruit_id))
    return result.scalar_one_or_none()


# ─── Fruit Types ─────────────────────────────────────────
async def create_fruit_type(db: AsyncSession, name: str, scientific_name: str,
                            cnn_model_path: str, config_path: str, dataset_path: str) -> FruitType:
    existing = await get_fruit_type_by_name(db, name)
    if existing:
        return existing
        
    fruit_type = FruitType(
        name=name,
        scientific_name=scientific_name,
        cnn_model_path=cnn_model_path,
        config_path=config_path,
        dataset_path=dataset_path
    )
    db.add(fruit_type)
    await db.commit()
    await db.refresh(fruit_type)
    return fruit_type

async def get_fruit_type_by_name(db: AsyncSession, name: str) -> Optional[FruitType]:
    result = await db.execute(select(FruitType).where(FruitType.name.ilike(name)))
    return result.scalar_one_or_none()

async def list_fruit_types(db: AsyncSession) -> List[FruitType]:
    result = await db.execute(select(FruitType).order_by(FruitType.id))
    return list(result.scalars().all())

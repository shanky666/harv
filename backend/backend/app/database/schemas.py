import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# ─── Auth ────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: Optional[str] = None
    password: str = Field(..., min_length=6)
    location: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str


# ─── User ────────────────────────────────────────────────
class UserOut(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    phone: Optional[str]
    location: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


# ─── Scan ────────────────────────────────────────────────
class ScanOut(BaseModel):
    scan_id: uuid.UUID
    user_id: uuid.UUID
    image_path: str
    total_fruits: int
    scan_date: datetime
    class Config:
        from_attributes = True


# ─── Detection ───────────────────────────────────────────
class DetectionItem(BaseModel):
    fruit_id: str
    fruit_type: Optional[str] = None
    bbox: List[int]
    confidence: float
    crop_path: Optional[str]

class DetectionResponse(BaseModel):
    scan_id: str
    total_fruits: int
    detections: List[DetectionItem]


# ─── Grading ─────────────────────────────────────────────
class GradedFruitItem(BaseModel):
    fruit_id: str
    grade: str
    confidence: float
    defects: List[str]

class GradeCountResponse(BaseModel):
    scan_id: str
    total_fruits: int
    better: int
    good: int
    reject: int
    fruits: List[GradedFruitItem]


# ─── Shelf Life ──────────────────────────────────────────
class ShelfLifeResponse(BaseModel):
    scan_id: uuid.UUID
    average_shelf_life: str
    breakdown: dict


# ─── Market ──────────────────────────────────────────────
class MarketResponse(BaseModel):
    scan_id: uuid.UUID
    best_market: str
    expected_price: str
    alternatives: List[dict]


# ─── Report ──────────────────────────────────────────────
class ReportResponse(BaseModel):
    scan_id: uuid.UUID
    total_fruits: int
    grades: dict
    shelf_life: str
    market: str
    pdf_url: Optional[str]
    created_at: datetime


# ─── Passport ────────────────────────────────────────────
class PassportResponse(BaseModel):
    passport_id: uuid.UUID
    fruit_id: str
    fruit_type: str
    grade: str
    defects: Optional[str]
    shelf_life: str
    market: str
    created_at: datetime


# ─── Fruit Types ─────────────────────────────────────────
class FruitTypeOut(BaseModel):
    id: int
    name: str
    scientific_name: str
    cnn_model_path: str
    config_path: str
    dataset_path: str
    created_at: datetime

    class Config:
        from_attributes = True

class FruitTypeRegisterRequest(BaseModel):
    name: str
    scientific_name: str
    cnn_model_path: str
    config_path: str
    dataset_path: str

class FruitRegistryListResponse(BaseModel):
    supported_fruits: List[str]


# ─── Process ─────────────────────────────────────────────
class ProcessResponse(BaseModel):
    scan_id: str
    total_fruits: int
    better: int
    good: int
    reject: int
    average_shelf_life: str
    fruit_distribution: dict
    markets: dict


# ─── Basket Analysis ─────────────────────────────────────
class BasketFruitItem(BaseModel):
    fruit_id: str
    fruit_type: str
    grade: str
    grade_confidence: float
    defect_score: Optional[float] = None
    defects: List[str] = []
    crop_path: Optional[str] = None
    mask_path: Optional[str] = None
    shelf_life: Optional[str] = None
    market_recommendation: Optional[str] = None

    class Config:
        from_attributes = True

class BasketSummary(BaseModel):
    better: int
    good: int
    reject: int

class BasketAnalysisResponse(BaseModel):
    session_id: str
    total_fruits: int
    fruit_type: str
    is_single: bool = False
    fruits: List[BasketFruitItem]
    summary: BasketSummary
    demo_mode: bool = False
    original_image_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    
    overall_grade: Optional[str] = None
    score: Optional[float] = None
    total_price: Optional[float] = None
    estimated_selling_price: Optional[float] = None
    average_shelf_life: Optional[float] = None
    recommended_market: Optional[str] = None
    ai_recommendations: Optional[List[str]] = []

    class Config:
        from_attributes = True



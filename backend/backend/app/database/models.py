import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import String, Integer, Float, ForeignKey, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from .connection import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(20), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scans: Mapped[list["FruitScan"]] = relationship("FruitScan", back_populates="user")


class FruitScan(Base):
    __tablename__ = "fruit_scans"

    scan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    image_path: Mapped[str] = mapped_column(String(500))
    total_fruits: Mapped[int] = mapped_column(Integer, default=0)
    scan_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="scans")
    detected_fruits: Mapped[list["DetectedFruit"]] = relationship("DetectedFruit", back_populates="scan")
    reports: Mapped[list["Report"]] = relationship("Report", back_populates="scan")


class FruitType(Base):
    __tablename__ = "fruit_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    scientific_name: Mapped[str] = mapped_column(String(100))
    cnn_model_path: Mapped[str] = mapped_column(String(200))
    config_path: Mapped[str] = mapped_column(String(200))
    dataset_path: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DetectedFruit(Base):
    __tablename__ = "detected_fruits"

    fruit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("fruit_scans.scan_id"))
    fruit_type_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("fruit_types.id"), nullable=True)
    fruit_type: Mapped[str] = mapped_column(String(50))
    bbox_x1: Mapped[float] = mapped_column(Float)
    bbox_y1: Mapped[float] = mapped_column(Float)
    bbox_x2: Mapped[float] = mapped_column(Float)
    bbox_y2: Mapped[float] = mapped_column(Float)
    grade: Mapped[str] = mapped_column(String(20), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    grade_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    defect_score: Mapped[float] = mapped_column(Float, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    shelf_life: Mapped[str] = mapped_column(String(50), nullable=True)
    market_recommendation: Mapped[str] = mapped_column(String(200), nullable=True)
    crop_path: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scan: Mapped["FruitScan"] = relationship("FruitScan", back_populates="detected_fruits")
    passport: Mapped["FruitPassport"] = relationship("FruitPassport", back_populates="fruit", uselist=False)
    fruit_type_rel: Mapped[Optional["FruitType"]] = relationship("FruitType")


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("fruit_scans.scan_id"))
    pdf_path: Mapped[str] = mapped_column(String(500), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scan: Mapped["FruitScan"] = relationship("FruitScan", back_populates="reports")


class FruitPassport(Base):
    __tablename__ = "fruit_passports"

    passport_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fruit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("detected_fruits.fruit_id"), unique=True)
    grade: Mapped[str] = mapped_column(String(20))
    defects: Mapped[str] = mapped_column(Text, nullable=True)
    shelf_life: Mapped[str] = mapped_column(String(50))
    market: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    fruit: Mapped["DetectedFruit"] = relationship("DetectedFruit", back_populates="passport")


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    image_path: Mapped[str] = mapped_column(String(500))
    fruit_type: Mapped[str] = mapped_column(String(50))
    is_single: Mapped[bool] = mapped_column(default=False)
    total_fruits: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    fruits: Mapped[list["BasketFruit"]] = relationship("BasketFruit", back_populates="session", cascade="all, delete-orphan")


class BasketFruit(Base):
    __tablename__ = "basket_fruits"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("analysis_sessions.session_id"))
    fruit_id: Mapped[str] = mapped_column(String(50))
    fruit_type: Mapped[str] = mapped_column(String(50))
    grade: Mapped[str] = mapped_column(String(20))
    grade_confidence: Mapped[float] = mapped_column(Float)
    defect_score: Mapped[float] = mapped_column(Float, nullable=True)
    defects: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bbox_x1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_x2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    crop_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    mask_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    shelf_life: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    market_recommendation: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["AnalysisSession"] = relationship("AnalysisSession", back_populates="fruits")


"""
Database models and session management.
ALL DATA IS SYNTHETIC - NO REAL PERSONAL DATA.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey,
    Text, Boolean, Index, Enum as SQLEnum, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import enum

from app.config import settings, DATA_DIR

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Session:
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, enum.Enum):
    i4c_admin = "i4c_admin"
    state_officer = "state_officer"
    district_officer = "district_officer"
    bank_nodal = "bank_nodal"


class ChannelType(str, enum.Enum):
    ATM = "ATM"
    AePS = "AePS"
    Crypto = "Crypto"


class FraudType(str, enum.Enum):
    UPI_Fraud = "UPI Fraud"
    Investment_Scam = "Investment Scam"
    Digital_Arrest = "Digital Arrest"
    Loan_App_Harassment = "Loan App Harassment"
    Phishing = "Phishing"
    QR_Code_Scam = "QR Code Scam"


class AlertSeverity(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class InterventionStatus(str, enum.Enum):
    PENDING = "PENDING"
    DEPLOYED = "DEPLOYED"
    INTERCEPTED = "INTERCEPTED"
    MISSED = "MISSED"
    CANCELLED = "CANCELLED"


# =============================================================================
# Models
# =============================================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    state = Column(String(50), nullable=True, index=True)
    district = Column(String(50), nullable=True)
    bank = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class Complaint(Base):
    __tablename__ = "complaints"
    
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(String(50), unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    fraud_type = Column(SQLEnum(FraudType), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    victim_bank = Column(String(50), nullable=False)
    hop_count = Column(Integer, nullable=False)
    mule_account_chain = Column(SQLiteJSON, nullable=False)  # List of dicts
    victim_city = Column(String(50), nullable=True)
    victim_state = Column(String(50), nullable=True)
    
    # Ground truth (filled after cash-out)
    withdrawal_city = Column(String(50), nullable=True, index=True)
    withdrawal_channel = Column(SQLEnum(ChannelType), nullable=True, index=True)
    withdrawal_timestamp = Column(DateTime, nullable=True)
    minutes_to_cashout = Column(Integer, nullable=True)
    
    # Prediction results
    predicted_city = Column(String(50), nullable=True)
    predicted_channel = Column(SQLEnum(ChannelType), nullable=True)
    predicted_eta_min = Column(Integer, nullable=True)
    prediction_probability = Column(Float, nullable=True)
    prediction_features = Column(SQLiteJSON, nullable=True)  # For explainability
    model_version = Column(String(50), nullable=True)
    
    # Metadata
    is_synthetic = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    alerts = relationship("Alert", back_populates="complaint")
    interventions = relationship("Intervention", back_populates="complaint")


class MuleAccount(Base):
    __tablename__ = "mule_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(50), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), nullable=True, index=True)
    device_id = Column(String(50), nullable=True, index=True)
    bank = Column(String(50), nullable=False)
    city = Column(String(50), nullable=True, index=True)
    state = Column(String(50), nullable=True)
    gang_id = Column(String(50), ForeignKey("gangs.gang_id"), nullable=True, index=True)
    is_synthetic = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    complaint_links = relationship("ComplaintMuleLink", back_populates="mule_account")
    gang = relationship("Gang", back_populates="mules")


class ComplaintMuleLink(Base):
    __tablename__ = "complaint_mule_links"
    
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False, index=True)
    mule_account_id = Column(Integer, ForeignKey("mule_accounts.id"), nullable=False, index=True)
    hop_position = Column(Integer, nullable=False)  # 1 = first hop, etc.
    amount_received = Column(Float, nullable=True)
    amount_forwarded = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=True)
    
    complaint = relationship("Complaint", backref="mule_links")
    mule_account = relationship("MuleAccount", back_populates="complaint_links")


class Gang(Base):
    __tablename__ = "gangs"
    
    id = Column(Integer, primary_key=True, index=True)
    gang_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    corridor = Column(SQLiteJSON, nullable=False)  # List of city names
    corridor_weights = Column(SQLiteJSON, nullable=False)
    channel_mix = Column(SQLiteJSON, nullable=False)
    mule_count = Column(Integer, nullable=False)
    base_city = Column(String(50), nullable=True)
    total_complaints = Column(Integer, default=0)
    total_amount = Column(Float, default=0.0)
    is_synthetic = Column(Boolean, default=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    mules = relationship("MuleAccount", back_populates="gang")


class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False, index=True)
    predicted_city = Column(String(50), nullable=False, index=True)
    predicted_channel = Column(SQLEnum(ChannelType), nullable=False)
    predicted_eta_min = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)  # 1, 2, 3 for top-3
    features_used = Column(SQLiteJSON, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    complaint = relationship("Complaint")


class Intervention(Base):
    __tablename__ = "interventions"
    
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False, index=True)
    team_id = Column(String(50), nullable=True)  # Null for crypto (exchange freeze)
    team_name = Column(String(100), nullable=True)
    target_city = Column(String(50), nullable=False)
    target_channel = Column(SQLEnum(ChannelType), nullable=False)
    predicted_eta_min = Column(Integer, nullable=False)
    travel_time_min = Column(Integer, nullable=True)
    status = Column(SQLEnum(InterventionStatus), default=InterventionStatus.PENDING)
    deployed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    amount_recovered = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    complaint = relationship("Complaint", back_populates="interventions")


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False, index=True)
    severity = Column(SQLEnum(AlertSeverity), nullable=False, index=True)
    channel = Column(SQLEnum(ChannelType), nullable=False)
    city = Column(String(50), nullable=False, index=True)
    message_en = Column(Text, nullable=False)
    message_hi = Column(Text, nullable=False)
    probability = Column(Float, nullable=False)
    eta_min = Column(Integer, nullable=False)
    amount = Column(Float, nullable=False)
    gang_id = Column(String(50), nullable=True)
    dispatched_sms = Column(Boolean, default=False)
    dispatched_email = Column(Boolean, default=False)
    dispatched_webhook = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    complaint = relationship("Complaint", back_populates="alerts")


class AuditLog(Base):
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    actor = Column(String(100), nullable=False)  # username or system
    action = Column(String(100), nullable=False)
    payload = Column(SQLiteJSON, nullable=True)
    prev_hash = Column(String(64), nullable=False)
    current_hash = Column(String(64), nullable=False, index=True)
    
    __table_args__ = (Index("ix_audit_timestamp_hash", "timestamp", "current_hash"),)


class FederatedRound(Base):
    __tablename__ = "federated_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, nullable=False, unique=True)
    participating_banks = Column(SQLiteJSON, nullable=False)
    global_weights = Column(SQLiteJSON, nullable=True)
    status = Column(String(20), default="pending")  # pending, aggregating, completed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    raw_rows_shared = Column(Integer, default=0)  # Should always be 0


class DisplacementPressure(Base):
    __tablename__ = "displacement_pressure"
    
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String(50), nullable=False, index=True)
    channel = Column(SQLEnum(ChannelType), nullable=False, index=True)
    pressure_score = Column(Float, default=0.0)  # 0-1, higher = more policing
    gang_id = Column(String(50), nullable=True, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (Index("ix_displacement_city_channel_gang", "city", "channel", "gang_id"),)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50), nullable=False, unique=True)
    train_start_date = Column(DateTime, nullable=False)
    train_end_date = Column(DateTime, nullable=False)
    test_start_date = Column(DateTime, nullable=False)
    test_end_date = Column(DateTime, nullable=False)
    top1_accuracy = Column(Float, nullable=False)
    top3_accuracy = Column(Float, nullable=False)
    precision_at_3 = Column(Float, nullable=False)
    mean_lead_time_min = Column(Float, nullable=False)
    mae_minutes = Column(Float, nullable=False)
    rmse_minutes = Column(Float, nullable=False)
    num_train_samples = Column(Integer, nullable=False)
    num_test_samples = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
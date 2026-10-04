"""
Pydantic schemas for API requests and responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, Enum):
    i4c_admin = "i4c_admin"
    state_officer = "state_officer"
    district_officer = "district_officer"
    bank_nodal = "bank_nodal"


class ChannelType(str, Enum):
    ATM = "ATM"
    AePS = "AePS"
    Crypto = "Crypto"


class FraudType(str, Enum):
    UPI_Fraud = "UPI Fraud"
    Investment_Scam = "Investment Scam"
    Digital_Arrest = "Digital Arrest"
    Loan_App_Harassment = "Loan App Harassment"
    Phishing = "Phishing"
    QR_Code_Scam = "QR Code Scam"


class AlertSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class InterventionStatus(str, Enum):
    PENDING = "PENDING"
    DEPLOYED = "DEPLOYED"
    INTERCEPTED = "INTERCEPTED"
    MISSED = "MISSED"
    CANCELLED = "CANCELLED"


# =============================================================================
# Auth Schemas
# =============================================================================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: UserRole
    state: Optional[str] = None
    district: Optional[str] = None
    bank: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Complaint Schemas
# =============================================================================

class MuleHop(BaseModel):
    account_id: str
    bank: str
    city: str
    hop_position: int
    amount_received: float
    amount_forwarded: float
    phone_number: Optional[str] = None
    device_id: Optional[str] = None


class ComplaintCreate(BaseModel):
    complaint_id: str
    timestamp: datetime
    fraud_type: FraudType
    amount: float
    victim_bank: str
    hop_count: int
    mule_account_chain: List[MuleHop]
    victim_city: Optional[str] = None
    victim_state: Optional[str] = None


class ComplaintResponse(BaseModel):
    id: int
    complaint_id: str
    timestamp: datetime
    fraud_type: FraudType
    amount: float
    victim_bank: str
    hop_count: int
    mule_account_chain: List[Dict]
    victim_city: Optional[str] = None
    victim_state: Optional[str] = None
    withdrawal_city: Optional[str] = None
    withdrawal_channel: Optional[ChannelType] = None
    withdrawal_timestamp: Optional[datetime] = None
    minutes_to_cashout: Optional[int] = None
    predicted_city: Optional[str] = None
    predicted_channel: Optional[ChannelType] = None
    predicted_eta_min: Optional[int] = None
    prediction_probability: Optional[float] = None
    model_version: Optional[str] = None
    is_synthetic: bool
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Prediction Schemas
# =============================================================================

class PredictionItem(BaseModel):
    rank: int
    city: str
    channel: ChannelType
    probability: float
    eta_min: int
    explanation: str


class PredictionResponse(BaseModel):
    complaint_id: int
    predictions: List[PredictionItem]
    model_version: str
    generated_at: datetime


class HeatmapRequest(BaseModel):
    channel: Optional[ChannelType] = None
    fraud_type: Optional[FraudType] = None
    hours: int = 24


class HeatmapCell(BaseModel):
    city: str
    lat: float
    lon: float
    risk_score: float
    complaint_count: int
    avg_amount: float
    top_channel: ChannelType


class HeatmapResponse(BaseModel):
    cells: List[HeatmapCell]
    generated_at: datetime
    filters: Dict[str, Any]


# =============================================================================
# Gang Schemas
# =============================================================================

class GangSummary(BaseModel):
    gang_id: str
    name: str
    corridor: List[str]
    corridor_weights: List[float]
    channel_mix: Dict[str, float]
    mule_count: int
    base_city: Optional[str] = None
    total_complaints: int
    total_amount: float


class GangDetail(GangSummary):
    mules: List[Dict]


# =============================================================================
# Optimizer Schemas
# =============================================================================

class OptimizeRequest(BaseModel):
    complaint_id: str
    amount: float
    predictions: List[PredictionItem]
    team_count: int = 10


class InterventionPlan(BaseModel):
    complaint_id: str
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    target_city: str
    target_channel: ChannelType
    predicted_eta_min: int
    travel_time_min: int
    arrival_before_cashout: bool
    probability: float
    expected_recovery: float
    reason: str
    action_type: str  # "physical" or "exchange_freeze"


class OptimizeResponse(BaseModel):
    complaint_id: str
    plans: List[InterventionPlan]
    total_expected_recovery: float
    teams_used: int
    unassigned_predictions: List[PredictionItem]


# =============================================================================
# Outcome/Displacement Schemas
# =============================================================================

class OutcomeRequest(BaseModel):
    complaint_id: str
    intervention_id: int
    intercepted: bool
    amount_recovered: float = 0.0
    notes: str = ""


class OutcomeResponse(BaseModel):
    intervention_id: int
    intercepted: bool
    amount_recovered: float
    displacement: Optional[Dict] = None
    corridor_updated: bool = False


# =============================================================================
# Federated Learning Schemas
# =============================================================================

class FederatedRoundResponse(BaseModel):
    round_number: int
    participating_banks: List[str]
    status: str
    aggregated_at: Optional[datetime] = None
    raw_rows_shared: int = 0


class FederatedStatusResponse(BaseModel):
    current_round: int
    total_rounds_completed: int
    recent_rounds: List[Dict]
    global_prior_available: bool
    privacy_guarantee: str


# =============================================================================
# Alerts Schemas
# =============================================================================

class AlertResponse(BaseModel):
    id: int
    complaint_id: int
    severity: AlertSeverity
    channel: ChannelType
    city: str
    message: str
    probability: float
    eta_min: int
    amount: float
    gang_id: Optional[str] = None
    dispatched: Dict[str, bool]
    created_at: datetime


# =============================================================================
# Audit Schemas
# =============================================================================

class AuditEntry(BaseModel):
    id: int
    timestamp: datetime
    actor: str
    action: str
    payload: Optional[Dict] = None
    prev_hash: str
    current_hash: str


class AuditVerifyResponse(BaseModel):
    valid: bool
    total_entries: int
    errors: List[Dict]
    last_hash: str


# =============================================================================
# Metrics Schemas
# =============================================================================

class ModelMetricsResponse(BaseModel):
    model_version: str
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None
    test_end_date: Optional[datetime] = None
    top1_accuracy: float
    top3_accuracy: float
    precision_at_3: float
    mean_lead_time_min: float
    mae_minutes: float
    rmse_minutes: float
    num_train_samples: int
    num_test_samples: int
    notes: Optional[str] = None
    trained_at: datetime


# =============================================================================
# Generic Response
# =============================================================================

class ErrorResponse(BaseModel):
    detail: str


class SuccessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


# =============================================================================
# Complaint Import Schemas (CFCFRMS-style)
# =============================================================================

class ComplaintImportRow(BaseModel):
    """Single complaint row for bulk import."""
    transaction_id: str
    amount: float
    timestamp: datetime
    victim_bank: str
    mode: str  # UPI, NEFT, IMPS, etc.
    account_chain: List[Dict]  # [{account_id, bank, city, hop_position, amount_received, amount_forwarded}]
    withdrawal_lat: Optional[float] = None
    withdrawal_lon: Optional[float] = None
    withdrawal_channel: Optional[ChannelType] = None
    fraud_type: Optional[FraudType] = None
    victim_city: Optional[str] = None
    victim_state: Optional[str] = None


class ComplaintImportRequest(BaseModel):
    """Bulk import request."""
    complaints: List[ComplaintImportRow]
    source: str = "CFCFRMS"
    validate_only: bool = False


class ComplaintImportError(BaseModel):
    """Error for a single failed row."""
    row_index: int
    transaction_id: Optional[str]
    errors: List[str]


class ComplaintImportResponse(BaseModel):
    """Response for bulk import."""
    success: bool
    total_rows: int
    imported: int
    failed: int
    errors: List[ComplaintImportError]
    message: str


# =============================================================================
# Model Retrain Schemas
# =============================================================================

class RetrainRequest(BaseModel):
    force_retrain: bool = False
    test_size: float = 0.2


class RetrainResponse(BaseModel):
    success: bool
    model_version: str
    metrics: Dict[str, Any]
    message: str


# =============================================================================
# Real Graph Metrics Schemas
# =============================================================================

class RealGraphMetricsResponse(BaseModel):
    dataset: str
    data_type: str
    n_transactions: int
    n_edges: int
    n_labeled: int
    n_communities: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_samples: int
    test_samples: int
    note: str
    generated_at: datetime


# =============================================================================
# Generic Response
# =============================================================================

class ErrorResponse(BaseModel):
    detail: str


class SuccessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
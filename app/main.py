"""
Main FastAPI application with all API endpoints.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.config import settings
from app.db import (
    get_db, init_db, Base, engine, SessionLocal,
    Complaint, MuleAccount, Gang, User, Alert, Intervention,
    DisplacementPressure, FederatedRound, ModelMetrics,
    ChannelType, FraudType, AlertSeverity, InterventionStatus, UserRole
)
from app.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_current_active_user,
    require_admin, require_state_or_admin, require_district_or_above,
    require_bank_or_admin, require_any_role, filter_by_user_scope
)
from app.schemas import *
from app.real_graph import run_real_graph_analysis
from app.synth import generate_all_synthetic_data, get_complaints_dataframe
from app.graph import run_gang_detection, save_detected_gangs, assign_complaints_to_detected_gangs, list_all_gangs, get_gang_summary
from app.model import WithdrawalPredictor, train_model, get_predictor
from app.optimizer import optimize_intervention, optimize_for_multiple_complaints
from app.displacement import record_outcome, get_pressure_map, get_displacement_summary
from app.federated import run_federated_round, get_federated_status
from app.alerts import (
    create_alerts_from_predictions, get_alerts, dispatch_alerts,
    log_audit, verify_audit_chain, get_audit_log, determine_severity
)


# Global predictor instance
predictor_instance: Optional[WithdrawalPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor_instance
    
    # Startup
    print("Starting application...")
    init_db()
    
    # Try to load existing model
    predictor_instance = get_predictor()
    
    yield
    
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Cybercrime Withdrawal Predictor API",
    description="Predictive analytics framework for cybercrime complaints to forecast likely cash withdrawal locations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Auth Endpoints
# =============================================================================

@app.post("/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login with username/password, returns JWT token."""
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info."""
    return current_user


# =============================================================================
# Complaint Endpoints
# =============================================================================

@app.post("/complaints", response_model=Dict[str, Any])
async def create_complaint(
    complaint: ComplaintCreate,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """
    Ingest a complaint, run prediction, create intervention plan, and generate alerts.
    This is the main entry point for the frontend.
    """
    global predictor_instance
    
    # Check if complaint already exists
    existing = db.query(Complaint).filter(
        Complaint.complaint_id == complaint.complaint_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Complaint ID already exists")
    
    # Convert mule chain to dict format
    mule_chain = [hop.model_dump() for hop in complaint.mule_account_chain]
    
    # Create complaint record
    db_complaint = Complaint(
        complaint_id=complaint.complaint_id,
        timestamp=complaint.timestamp,
        fraud_type=complaint.fraud_type,
        amount=complaint.amount,
        victim_bank=complaint.victim_bank,
        hop_count=complaint.hop_count,
        mule_account_chain=mule_chain,
        victim_city=complaint.victim_city,
        victim_state=complaint.victim_state,
        is_synthetic=False,  # Real complaints from frontend
    )
    db.add(db_complaint)
    db.commit()
    db.refresh(db_complaint)
    
    # Run prediction
    if predictor_instance is None:
        predictor_instance = get_predictor()
    
    # Prepare complaint data for prediction
    first_mule_city = mule_chain[0].get("city") if mule_chain else None
    first_mule_bank = mule_chain[0].get("bank") if mule_chain else None
    
    # Convert datetime to ISO format string for JSON serialization
    timestamp_str = complaint.timestamp.isoformat() if isinstance(complaint.timestamp, datetime) else str(complaint.timestamp)
    
    prediction_input = {
        "timestamp": timestamp_str,
        "fraud_type": complaint.fraud_type.value,
        "amount": complaint.amount,
        "victim_bank": complaint.victim_bank,
        "hop_count": complaint.hop_count,
        "victim_city": complaint.victim_city,
        "first_mule_city": first_mule_city,
        "first_mule_bank": first_mule_bank,
        "detected_gang_id": "unknown",  # Would come from graph detection
    }
    
    try:
        predictions = predictor_instance.predict(prediction_input, top_k=3)
    except Exception as e:
        # Fallback if model not trained
        predictions = [
            {"rank": 1, "city": "Delhi", "channel": "ATM", "probability": 0.5, "eta_min": 45, 
             "explanation": "Model not trained, using fallback"},
            {"rank": 2, "city": "Mumbai", "channel": "AePS", "probability": 0.3, "eta_min": 120,
             "explanation": "Model not trained, using fallback"},
            {"rank": 3, "city": "Bengaluru", "channel": "Crypto", "probability": 0.2, "eta_min": 15,
             "explanation": "Model not trained, using fallback"},
        ]
    
    # Update complaint with top prediction
    top_pred = predictions[0]
    db_complaint.predicted_city = top_pred["city"]
    db_complaint.predicted_channel = ChannelType(top_pred["channel"])
    db_complaint.predicted_eta_min = top_pred["eta_min"]
    db_complaint.prediction_probability = top_pred["probability"]
    db_complaint.model_version = predictor_instance.model_version if predictor_instance else "unknown"
    db_complaint.prediction_features = prediction_input
    
    # Save individual predictions
    from app.db import Prediction
    for pred in predictions:
        pred_record = Prediction(
            complaint_id=db_complaint.id,
            predicted_city=pred["city"],
            predicted_channel=ChannelType(pred["channel"]),
            predicted_eta_min=pred["eta_min"],
            probability=pred["probability"],
            rank=pred["rank"],
            explanation=pred["explanation"],
        )
        db.add(pred_record)
    
    # Create intervention plan
    opt_result = optimize_intervention(
        complaint_id=db_complaint.complaint_id,
        amount=complaint.amount,
        predictions=predictions,
        team_count=10,
    )
    
    # Save interventions
    for plan in opt_result.plans:
        intervention = Intervention(
            complaint_id=db_complaint.id,
            team_id=plan.team_id,
            team_name=plan.team_name,
            target_city=plan.target_city,
            target_channel=ChannelType(plan.target_channel),
            predicted_eta_min=plan.predicted_eta_min,
            travel_time_min=plan.travel_time_min,
            status=InterventionStatus.PENDING,
            notes=plan.reason,
        )
        db.add(intervention)
    
    # Create alerts
    gang_id = prediction_input.get("detected_gang_id")
    alerts = create_alerts_from_predictions(db, db_complaint.id, predictions, gang_id)
    
    db.commit()
    
    # Log audit
    log_audit(db, current_user.username, "complaint_created", {
        "complaint_id": db_complaint.complaint_id,
        "amount": complaint.amount,
        "predictions": len(predictions),
    })
    
    return {
        "complaint": ComplaintResponse.model_validate(db_complaint),
        "predictions": [PredictionItem(**p) for p in predictions],
        "intervention_plan": {
            "plans": [InterventionPlan(**p.__dict__) for p in opt_result.plans],
            "total_expected_recovery": opt_result.total_expected_recovery,
            "teams_used": opt_result.teams_used,
        },
        "alerts": [
            AlertResponse(
                id=a.id,
                complaint_id=a.complaint_id,
                severity=a.severity,
                channel=a.channel,
                city=a.city,
                message=a.message_en,  # Default to English
                probability=a.probability,
                eta_min=a.eta_min,
                amount=a.amount,
                gang_id=a.gang_id,
                dispatched={
                    "sms": a.dispatched_sms,
                    "email": a.dispatched_email,
                    "webhook": a.dispatched_webhook,
                },
                created_at=a.created_at,
            )
            for a in alerts
        ],
    }


@app.get("/predictions/{complaint_id}", response_model=PredictionResponse)
async def get_predictions(
    complaint_id: int,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get predictions for a specific complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    # Check scope
    complaint = filter_by_user_scope(db.query(Complaint).filter(Complaint.id == complaint_id), Complaint, current_user).first()
    if not complaint:
        raise HTTPException(status_code=403, detail="Not authorized to view this complaint")
    
    predictions = db.query(Prediction).filter(
        Prediction.complaint_id == complaint_id
    ).order_by(Prediction.rank).all()
    
    return PredictionResponse(
        complaint_id=complaint_id,
        predictions=[PredictionItem(
            rank=p.rank,
            city=p.predicted_city,
            channel=p.predicted_channel,
            probability=p.probability,
            eta_min=p.predicted_eta_min,
            explanation=p.explanation or "",
        ) for p in predictions],
        model_version=complaint.model_version or "unknown",
        generated_at=datetime.utcnow(),
    )


# =============================================================================
# Heatmap Endpoint
# =============================================================================

@app.get("/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    channel: Optional[ChannelType] = None,
    fraud_type: Optional[FraudType] = None,
    hours: int = 24,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get risk heatmap for cities."""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(Complaint).filter(Complaint.timestamp >= since)
    
    if channel:
        query = query.filter(Complaint.withdrawal_channel == channel)
    if fraud_type:
        query = query.filter(Complaint.fraud_type == fraud_type)
    
    # Role-based filtering
    query = filter_by_user_scope(query, Complaint, current_user)
    
    complaints = query.all()
    
    # Aggregate by city
    city_stats = {}
    for c in complaints:
        city = c.withdrawal_city or c.victim_city
        if not city:
            continue
        if city not in city_stats:
            city_stats[city] = {"count": 0, "total_amount": 0.0, "channels": []}
        city_stats[city]["count"] += 1
        city_stats[city]["total_amount"] += c.amount
        if c.withdrawal_channel:
            city_stats[city]["channels"].append(c.withdrawal_channel.value)
    
    cells = []
    for city, stats in city_stats.items():
        if city not in CITY_COORDS:
            continue
        lat, lon = CITY_COORDS[city]
        
        # Simple risk score based on complaint count and amount
        risk_score = min(1.0, (stats["count"] * 0.1) + (stats["total_amount"] / 1e7))
        
        # Top channel
        from collections import Counter
        top_channel = Counter(stats["channels"]).most_common(1)
        top_channel_val = top_channel[0][0] if top_channel else "ATM"
        
        cells.append(HeatmapCell(
            city=city,
            lat=lat,
            lon=lon,
            risk_score=risk_score,
            complaint_count=stats["count"],
            avg_amount=stats["total_amount"] / stats["count"] if stats["count"] > 0 else 0,
            top_channel=ChannelType(top_channel_val),
        ))
    
    return HeatmapResponse(
        cells=cells,
        generated_at=datetime.utcnow(),
        filters={"channel": channel, "fraud_type": fraud_type, "hours": hours},
    )


# =============================================================================
# Gang Endpoints
# =============================================================================

@app.get("/gangs", response_model=List[GangSummary])
async def get_gangs(
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """List all gangs with summary statistics."""
    gangs = list_all_gangs(db)
    return [GangSummary(**g) for g in gangs]


@app.get("/gangs/{gang_id}", response_model=GangDetail)
async def get_gang(
    gang_id: str,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get detailed gang information."""
    gang = get_gang_summary(db, gang_id)
    if not gang:
        raise HTTPException(status_code=404, detail="Gang not found")
    return GangDetail(**gang)


# =============================================================================
# Optimizer Endpoint
# =============================================================================

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(
    request: OptimizeRequest,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get intervention plan for a complaint."""
    result = optimize_intervention(
        complaint_id=request.complaint_id,
        amount=request.amount,
        predictions=[p.model_dump() for p in request.predictions],
        team_count=request.team_count,
    )
    
    return OptimizeResponse(
        complaint_id=result.complaint_id,
        plans=[InterventionPlan(**p.__dict__) for p in result.plans],
        total_expected_recovery=result.total_expected_recovery,
        teams_used=result.teams_used,
        unassigned_predictions=[PredictionItem(**p.__dict__) for p in result.unassigned_predictions],
    )


# =============================================================================
# Outcome/Displacement Endpoint
# =============================================================================

@app.post("/outcomes", response_model=OutcomeResponse)
async def record_outcome_endpoint(
    request: OutcomeRequest,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Record intervention outcome and trigger displacement/feedback."""
    result = record_outcome(
        db=db,
        complaint_id=request.complaint_id,
        intervention_id=request.intervention_id,
        intercepted=request.intercepted,
        amount_recovered=request.amount_recovered,
        notes=request.notes,
    )
    
    log_audit(db, current_user.username, "outcome_recorded", {
        "complaint_id": request.complaint_id,
        "intervention_id": request.intervention_id,
        "intercepted": request.intercepted,
        "amount_recovered": request.amount_recovered,
    })
    
    return OutcomeResponse(**result)


# =============================================================================
# Federated Learning Endpoints
# =============================================================================

@app.post("/federated/round", response_model=FederatedRoundResponse)
async def federated_round(
    banks: Optional[List[str]] = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Run a federated learning round."""
    result = run_federated_round(db, banks)
    
    return FederatedRoundResponse(
        round_number=result.round_number,
        participating_banks=result.participating_banks,
        status="completed",
        aggregated_at=datetime.fromisoformat(result.aggregated_at),
        raw_rows_shared=result.raw_rows_shared,
    )


@app.get("/federated/status", response_model=FederatedStatusResponse)
async def federated_status(
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get federated learning status."""
    status = get_federated_status(db)
    return FederatedStatusResponse(**status)


# =============================================================================
# Alerts Endpoints
# =============================================================================

@app.get("/alerts", response_model=List[AlertResponse])
async def get_alerts_endpoint(
    lang: str = Query("en", pattern="^(en|hi)$"),
    severity: Optional[AlertSeverity] = None,
    channel: Optional[ChannelType] = None,
    city: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get alerts with filtering and language support."""
    alerts = get_alerts(
        db=db,
        lang=lang,
        severity=severity.value if severity else None,
        channel=channel.value if channel else None,
        city=city,
        limit=limit,
        user_role=current_user.role.value,
        user_state=current_user.state,
        user_district=current_user.district,
        user_bank=current_user.bank,
    )
    return [AlertResponse(**a) for a in alerts]


@app.post("/alerts/dispatch")
async def dispatch_alerts_endpoint(
    alert_ids: Optional[List[int]] = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Dispatch alerts via SMS/email/webhook (simulated)."""
    result = dispatch_alerts(db, alert_ids)
    return {"dispatched": result}


# =============================================================================
# Audit Endpoints
# =============================================================================

@app.get("/audit/verify", response_model=AuditVerifyResponse)
async def verify_audit(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Verify audit chain integrity."""
    result = verify_audit_chain(db)
    return AuditVerifyResponse(**result)


@app.get("/audit/log", response_model=List[AuditEntry])
async def get_audit_log_endpoint(
    limit: int = 100,
    actor: Optional[str] = None,
    action: Optional[str] = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get audit log entries."""
    entries = get_audit_log(db, limit, actor, action)
    return [AuditEntry(**e) for e in entries]


# =============================================================================
# Metrics Endpoint
# =============================================================================

@app.get("/metrics", response_model=ModelMetricsResponse)
async def get_metrics(
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get model evaluation metrics."""
    metrics = db.query(ModelMetrics).order_by(ModelMetrics.created_at.desc()).first()
    if not metrics:
        # Try loading from JSON file
        import json
        from app.config import METRICS_FILE
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                data = json.load(f)
            return ModelMetricsResponse(**data)
        raise HTTPException(status_code=404, detail="No metrics available")
    
    return ModelMetricsResponse.model_validate(metrics)


# =============================================================================
# Development Endpoints
# =============================================================================

@app.post("/dev/seed")
async def dev_seed(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Regenerate synthetic data and retrain model."""
    global predictor_instance
    
    # Generate data
    stats = generate_all_synthetic_data()
    
    # Run gang detection
    summaries = run_gang_detection(db)
    save_detected_gangs(db, summaries)
    assign_complaints_to_detected_gangs(db)
    
    # Retrain model
    metrics = train_model()
    
    # Reload predictor
    predictor_instance = get_predictor()
    
    log_audit(db, current_user.username, "dev_seed", stats)
    
    return SuccessResponse(
        success=True,
        message="Synthetic data regenerated and model retrained",
        data=stats,
    )


# =============================================================================
# Complaint Import Endpoint (CFCFRMS-style)
# =============================================================================

@app.post("/import/complaints", response_model=ComplaintImportResponse)
async def import_complaints(
    request: ComplaintImportRequest,
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """
    Bulk import complaints from CFCFRMS-style CSV/JSON.
    Validates each row, rejects bad rows with clear errors.
    After import, optionally triggers model retrain via /model/retrain.
    """
    imported = 0
    failed = 0
    errors = []
    
    for idx, row in enumerate(request.complaints):
        try:
            # Check if complaint already exists
            existing = db.query(Complaint).filter(
                Complaint.complaint_id == row.transaction_id
            ).first()
            if existing:
                errors.append(ComplaintImportError(
                    row_index=idx,
                    transaction_id=row.transaction_id,
                    errors=[f"Complaint ID {row.transaction_id} already exists"]
                ))
                failed += 1
                continue
            
            # Validate account chain
            if not row.account_chain or len(row.account_chain) == 0:
                errors.append(ComplaintImportError(
                    row_index=idx,
                    transaction_id=row.transaction_id,
                    errors=["account_chain cannot be empty"]
                ))
                failed += 1
                continue
            
            # Validate hop positions are sequential
            hop_positions = [h.get("hop_position", 0) for h in row.account_chain]
            if sorted(hop_positions) != list(range(1, len(hop_positions) + 1)):
                errors.append(ComplaintImportError(
                    row_index=idx,
                    transaction_id=row.transaction_id,
                    errors=["hop_position must be sequential starting from 1"]
                ))
                failed += 1
                continue
            
            # Convert account chain
            mule_chain = []
            for hop in row.account_chain:
                mule_chain.append({
                    "account_id": hop.get("account_id", ""),
                    "bank": hop.get("bank", ""),
                    "city": hop.get("city", ""),
                    "hop_position": hop.get("hop_position", 0),
                    "amount_received": hop.get("amount_received", 0.0),
                    "amount_forwarded": hop.get("amount_forwarded", 0.0),
                    "phone_number": hop.get("phone_number"),
                    "device_id": hop.get("device_id"),
                })
            
            # Determine fraud_type
            fraud_type = row.fraud_type or FraudType.UPI_Fraud
            
            # Create complaint record
            db_complaint = Complaint(
                complaint_id=row.transaction_id,
                timestamp=row.timestamp,
                fraud_type=fraud_type,
                amount=row.amount,
                victim_bank=row.victim_bank,
                hop_count=len(row.account_chain),
                mule_account_chain=mule_chain,
                victim_city=row.victim_city,
                victim_state=row.victim_state,
                withdrawal_city=None,  # Will be filled if withdrawal_lat/lon provided
                withdrawal_channel=row.withdrawal_channel,
                is_synthetic=False,
            )
            
            # If withdrawal coordinates provided, find nearest city
            if row.withdrawal_lat is not None and row.withdrawal_lon is not None:
                from app.geo import get_nearest_cashout_cells
                nearest = get_nearest_cashout_cells(
                    row.withdrawal_lat, row.withdrawal_lon, k=1
                )
                if nearest:
                    cell = nearest[0][0]
                    from app.geo import get_city_from_h3
                    city = get_city_from_h3(cell)
                    if city:
                        db_complaint.withdrawal_city = city
            
            db.add(db_complaint)
            db.flush()
            
            # Create mule account links
            from app.db import MuleAccount, ComplaintMuleLink
            for hop in row.account_chain:
                account_id = hop.get("account_id", "")
                mule = db.query(MuleAccount).filter(
                    MuleAccount.account_id == account_id
                ).first()
                if not mule:
                    mule = MuleAccount(
                        account_id=account_id,
                        bank=hop.get("bank", ""),
                        city=hop.get("city", ""),
                        is_synthetic=False,
                    )
                    db.add(mule)
                    db.flush()
                
                link = ComplaintMuleLink(
                    complaint_id=db_complaint.id,
                    mule_account_id=mule.id,
                    hop_position=hop.get("hop_position", 0),
                    amount_received=hop.get("amount_received", 0.0),
                    amount_forwarded=hop.get("amount_forwarded", 0.0),
                    timestamp=row.timestamp,
                )
                db.add(link)
            
            db.commit()
            imported += 1
            
        except Exception as e:
            db.rollback()
            errors.append(ComplaintImportError(
                row_index=idx,
                transaction_id=row.transaction_id,
                errors=[str(e)]
            ))
            failed += 1
            continue
    
    log_audit(db, current_user.username, "complaint_import", {
        "total_rows": len(request.complaints),
        "imported": imported,
        "failed": failed,
    })
    
    return ComplaintImportResponse(
        success=failed == 0,
        total_rows=len(request.complaints),
        imported=imported,
        failed=failed,
        errors=errors,
        message=f"Imported {imported}, failed {failed} out of {len(request.complaints)} rows"
    )


# =============================================================================
# Model Retrain Endpoint
# =============================================================================

@app.post("/model/retrain", response_model=RetrainResponse)
async def retrain_model(
    request: RetrainRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Retrain the prediction model.
    If force_retrain=False and model exists, returns existing model info.
    """
    global predictor_instance
    
    if not request.force_retrain and predictor_instance is not None:
        return RetrainResponse(
            success=True,
            model_version=predictor_instance.model_version,
            metrics={},
            message=f"Model {predictor_instance.model_version} already loaded. Use force_retrain=true to retrain."
        )
    
    # Retrain
    try:
        metrics = train_model()
        predictor_instance = get_predictor()
        
        log_audit(db, current_user.username, "model_retrain", {
            "model_version": predictor_instance.model_version,
            "metrics": metrics,
        })
        
        return RetrainResponse(
            success=True,
            model_version=predictor_instance.model_version,
            metrics=metrics,
            message=f"Model {predictor_instance.model_version} trained successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# =============================================================================
# Real Graph Metrics Endpoint (Elliptic Bitcoin Dataset)
# =============================================================================

@app.get("/metrics/real-graph", response_model=RealGraphMetricsResponse)
async def get_real_graph_metrics(
    current_user: User = Depends(require_any_role),
    db: Session = Depends(get_db)
):
    """Get metrics for the Elliptic Bitcoin dataset analysis (REAL BLOCKCHAIN DATA)."""
    from app.config import BASE_DIR
    metrics_file = BASE_DIR / "metrics_real_graph.json"
    
    if not metrics_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Real graph metrics not found. Run the analysis first via app/real_graph.py"
        )
    
    import json
    with open(metrics_file) as f:
        data = json.load(f)
    
    return RealGraphMetricsResponse(**data)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
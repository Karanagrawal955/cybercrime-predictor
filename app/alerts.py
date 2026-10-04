"""
Alerts system with Hindi/English templates and tamper-evident audit log.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import (
    ALERT_TEMPLATES, ALERT_SEVERITY_THRESHOLDS, CITY_NAMES
)
from app.db import (
    Alert, Complaint, AuditLog, AlertSeverity, ChannelType, Intervention
)


def determine_severity(probability: float) -> AlertSeverity:
    """Determine alert severity from probability."""
    if probability >= ALERT_SEVERITY_THRESHOLDS["HIGH"]:
        return AlertSeverity.HIGH
    elif probability >= ALERT_SEVERITY_THRESHOLDS["MEDIUM"]:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


def format_alert_message(
    template: str,
    city: str,
    channel: str,
    eta_min: int,
    amount: float,
    gang_id: str,
) -> str:
    """Format alert template with values."""
    return template.format(
        city=city,
        channel=channel,
        eta_min=eta_min,
        amount=amount,
        gang_id=gang_id,
    )


def create_alert(
    db: Session,
    complaint_id: int,
    city: str,
    channel: str,
    probability: float,
    eta_min: int,
    amount: float,
    gang_id: str = None,
) -> Alert:
    """Create alert records in both languages."""
    severity = determine_severity(probability)
    
    message_en = format_alert_message(
        ALERT_TEMPLATES["en"][severity.value],
        city, channel, eta_min, amount, gang_id or "unknown"
    )
    message_hi = format_alert_message(
        ALERT_TEMPLATES["hi"][severity.value],
        city, channel, eta_min, amount, gang_id or "unknown"
    )
    
    alert = Alert(
        complaint_id=complaint_id,
        severity=severity,
        channel=ChannelType(channel),
        city=city,
        message_en=message_en,
        message_hi=message_hi,
        probability=probability,
        eta_min=eta_min,
        amount=amount,
        gang_id=gang_id,
        dispatched_sms=False,
        dispatched_email=False,
        dispatched_webhook=False,
    )
    
    db.add(alert)
    db.commit()
    db.refresh(alert)
    
    # Log to audit
    log_audit(
        db,
        actor="system",
        action="alert_created",
        payload={
            "alert_id": alert.id,
            "complaint_id": complaint_id,
            "severity": severity.value,
            "city": city,
            "channel": channel,
            "probability": probability,
        }
    )
    
    return alert


def dispatch_alerts(db: Session, alert_ids: List[int] = None) -> Dict:
    """Simulate dispatching alerts (SMS, email, webhook)."""
    query = db.query(Alert)
    if alert_ids:
        query = query.filter(Alert.id.in_(alert_ids))
    else:
        # Dispatch undispatched high/medium alerts
        query = query.filter(
            Alert.dispatched_sms == False,
            Alert.severity.in_([AlertSeverity.HIGH, AlertSeverity.MEDIUM])
        )
    
    alerts = query.all()
    dispatched = {"sms": 0, "email": 0, "webhook": 0}
    
    for alert in alerts:
        # Simulate SMS dispatch
        alert.dispatched_sms = True
        dispatched["sms"] += 1
        
        # Simulate email dispatch
        alert.dispatched_email = True
        dispatched["email"] += 1
        
        # Simulate webhook dispatch
        alert.dispatched_webhook = True
        dispatched["webhook"] += 1
        
        log_audit(
            db,
            actor="system",
            action="alert_dispatched",
            payload={
                "alert_id": alert.id,
                "channels": ["sms", "email", "webhook"],
            }
        )
    
    db.commit()
    return dispatched


def get_alerts(
    db: Session,
    lang: str = "en",
    severity: str = None,
    channel: str = None,
    city: str = None,
    limit: int = 100,
    user_role: str = None,
    user_state: str = None,
    user_district: str = None,
    user_bank: str = None,
) -> List[Dict]:
    """Get alerts with filtering and role-based scoping."""
    query = db.query(Alert).join(Complaint)
    
    # Role-based filtering
    if user_role == "state_officer" and user_state:
        query = query.filter(Complaint.victim_state == user_state)
    elif user_role == "district_officer" and user_district:
        query = query.filter(Complaint.victim_city == user_district)
    elif user_role == "bank_nodal" and user_bank:
        query = query.filter(Complaint.victim_bank == user_bank)
    # i4c_admin sees all
    
    if severity:
        query = query.filter(Alert.severity == AlertSeverity(severity))
    if channel:
        query = query.filter(Alert.channel == ChannelType(channel))
    if city:
        query = query.filter(Alert.city == city)
    
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    
    result = []
    for alert in alerts:
        message = alert.message_hi if lang == "hi" else alert.message_en
        result.append({
            "id": alert.id,
            "complaint_id": alert.complaint_id,
            "severity": alert.severity.value,
            "channel": alert.channel.value,
            "city": alert.city,
            "message": message,
            "probability": alert.probability,
            "eta_min": alert.eta_min,
            "amount": alert.amount,
            "gang_id": alert.gang_id,
            "dispatched": {
                "sms": alert.dispatched_sms,
                "email": alert.dispatched_email,
                "webhook": alert.dispatched_webhook,
            },
            "created_at": alert.created_at.isoformat(),
        })
    
    return result


# =============================================================================
# Tamper-Evident Audit Log
# =============================================================================

def compute_hash(prev_hash: str, timestamp: datetime, payload: Dict) -> str:
    """Compute SHA-256 hash for audit chain."""
    data = f"{prev_hash}{timestamp.isoformat()}{json.dumps(payload, sort_keys=True)}"
    return hashlib.sha256(data.encode()).hexdigest()


def log_audit(
    db: Session,
    actor: str,
    action: str,
    payload: Dict = None,
) -> AuditLog:
    """Write tamper-evident audit log entry."""
    # Get previous hash
    last_entry = db.query(AuditLog).order_by(AuditLog.id.desc()).first()
    prev_hash = last_entry.current_hash if last_entry else "0" * 64
    
    timestamp = datetime.utcnow()
    current_hash = compute_hash(prev_hash, timestamp, payload or {})
    
    entry = AuditLog(
        timestamp=timestamp,
        actor=actor,
        action=action,
        payload=payload,
        prev_hash=prev_hash,
        current_hash=current_hash,
    )
    
    db.add(entry)
    db.commit()
    return entry


def verify_audit_chain(db: Session) -> Dict:
    """Verify the entire audit chain integrity."""
    entries = db.query(AuditLog).order_by(AuditLog.id).all()
    
    if not entries:
        return {
            "valid": True,
            "total_entries": 0,
            "message": "Audit log is empty",
        }
    
    errors = []
    prev_hash = "0" * 64
    
    for i, entry in enumerate(entries):
        # Verify hash chain
        expected_hash = compute_hash(prev_hash, entry.timestamp, entry.payload or {})
        if entry.current_hash != expected_hash:
            errors.append({
                "entry_id": entry.id,
                "error": "Hash mismatch",
                "expected": expected_hash,
                "actual": entry.current_hash,
            })
        
        # Verify prev_hash linkage
        if entry.prev_hash != prev_hash:
            errors.append({
                "entry_id": entry.id,
                "error": "Prev hash mismatch",
                "expected": prev_hash,
                "actual": entry.prev_hash,
            })
        
        prev_hash = entry.current_hash
    
    return {
        "valid": len(errors) == 0,
        "total_entries": len(entries),
        "errors": errors,
        "last_hash": prev_hash,
    }


def get_audit_log(
    db: Session,
    limit: int = 100,
    actor: str = None,
    action: str = None,
) -> List[Dict]:
    """Get audit log entries."""
    query = db.query(AuditLog)
    
    if actor:
        query = query.filter(AuditLog.actor == actor)
    if action:
        query = query.filter(AuditLog.action == action)
    
    entries = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "actor": e.actor,
            "action": e.action,
            "payload": e.payload,
            "prev_hash": e.prev_hash,
            "current_hash": e.current_hash,
        }
        for e in entries
    ]


# =============================================================================
# High-level alert creation from complaint processing
# =============================================================================

def create_alerts_from_predictions(
    db: Session,
    complaint_id: int,
    predictions: List[Dict],
    gang_id: str = None,
) -> List[Alert]:
    """Create alerts for top predictions from a complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        return []
    
    alerts = []
    for pred in predictions[:3]:  # Top 3
        alert = create_alert(
            db=db,
            complaint_id=complaint_id,
            city=pred["city"],
            channel=pred["channel"],
            probability=pred["probability"],
            eta_min=pred["eta_min"],
            amount=complaint.amount,
            gang_id=gang_id,
        )
        alerts.append(alert)
    
    return alerts


if __name__ == "__main__":
    from app.db import SessionLocal
    db = SessionLocal()
    try:
        # Test alert creation
        alert = create_alert(
            db, complaint_id=1, city="Delhi", channel="ATM",
            probability=0.85, eta_min=30, amount=500000, gang_id="gang_001"
        )
        print(f"Created alert: {alert.id} - {alert.message_en}")
        
        # Test audit
        log_audit(db, "test_user", "test_action", {"test": "data"})
        
        # Verify chain
        result = verify_audit_chain(db)
        print(f"Audit verification: {result}")
    finally:
        db.close()
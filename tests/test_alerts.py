"""
Tests for alerts and audit log.
"""

import pytest
import hashlib
import json
from datetime import datetime
from datetime import datetime
from sqlalchemy.orm import Session

from app.alerts import (
    determine_severity, create_alert, get_alerts, dispatch_alerts,
    log_audit, verify_audit_chain, compute_hash, create_alerts_from_predictions
)
from app.db import SessionLocal, Base, engine, Alert, Complaint, AuditLog, AlertSeverity, ChannelType
from app.synth import generate_all_synthetic_data


@pytest.fixture(scope="module")
def db():
    """Create a test database with synthetic data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    generate_all_synthetic_data()
    yield db
    db.close()


def test_severity_thresholds():
    """Test alert severity determination."""
    assert determine_severity(0.8) == AlertSeverity.HIGH
    assert determine_severity(0.5) == AlertSeverity.MEDIUM
    assert determine_severity(0.2) == AlertSeverity.LOW


def test_create_alert(db: Session):
    """Test creating alert with both languages."""
    # Create a complaint first
    complaint = Complaint(
        complaint_id="TEST-ALERT-001",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=100000,
        victim_bank="HDFC",
        hop_count=2,
        mule_account_chain=[{"account_id": "TEST123", "city": "Delhi"}],
        is_synthetic=False,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    
    alert = create_alert(
        db=db,
        complaint_id=complaint.id,
        city="Delhi",
        channel="ATM",
        probability=0.85,
        eta_min=30,
        amount=100000,
        gang_id="gang_001",
    )
    
    assert alert.id is not None
    assert alert.severity == AlertSeverity.HIGH
    assert alert.channel == ChannelType.ATM
    assert alert.city == "Delhi"
    assert alert.message_en is not None
    assert alert.message_hi is not None
    assert "Delhi" in alert.message_en
    assert "ATM" in alert.message_en
    assert "उच्च" in alert.message_hi  # Hindi for HIGH


def test_get_alerts_filtering(db: Session):
    """Test getting alerts with filters."""
    # Create test alerts
    complaint = Complaint(
        complaint_id="TEST-ALERT-002",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=100000,
        victim_bank="HDFC",
        hop_count=2,
        mule_account_chain=[{"account_id": "TEST456", "city": "Mumbai"}],
        is_synthetic=False,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    
    create_alert(db, complaint.id, "Mumbai", "ATM", 0.9, 15, 100000, "gang_002")
    create_alert(db, complaint.id, "Delhi", "Crypto", 0.3, 10, 50000, "gang_001")
    
    # Filter by severity
    high_alerts = get_alerts(db, severity="HIGH")
    assert len(high_alerts) >= 1
    for a in high_alerts:
        assert a["severity"] == "HIGH"
    
    # Filter by channel
    crypto_alerts = get_alerts(db, channel="Crypto")
    assert len(crypto_alerts) >= 1
    for a in crypto_alerts:
        assert a["channel"] == "Crypto"
    
    # Filter by city
    mumbai_alerts = get_alerts(db, city="Mumbai")
    assert len(mumbai_alerts) >= 1
    for a in mumbai_alerts:
        assert a["city"] == "Mumbai"


def test_alert_language_templates(db: Session):
    """Test that alerts have both English and Hindi messages."""
    complaint = Complaint(
        complaint_id="TEST-ALERT-003",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=100000,
        victim_bank="HDFC",
        hop_count=2,
        mule_account_chain=[{"account_id": "TEST789", "city": "Bengaluru"}],
        is_synthetic=False,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    
    create_alert(db, complaint.id, "Bengaluru", "AePS", 0.7, 60, 100000, "gang_005")
    
    en_alerts = get_alerts(db, lang="en", city="Bengaluru")
    hi_alerts = get_alerts(db, lang="hi", city="Bengaluru")
    
    assert len(en_alerts) == 1
    assert len(hi_alerts) == 1
    
    assert "Bengaluru" in en_alerts[0]["message"]
    assert "AePS" in en_alerts[0]["message"]
    assert "बेंगलुरु" in hi_alerts[0]["message"] or "Bengaluru" in hi_alerts[0]["message"]


def test_audit_log_chain(db: Session):
    """Test that audit log creates proper hash chain."""
    # Clear existing audit log for clean test
    db.query(AuditLog).delete()
    db.commit()
    
    # Log first entry
    log_audit(db, "user1", "action1", {"key": "value1"})
    
    # Log second entry
    log_audit(db, "user2", "action2", {"key": "value2"})
    
    # Verify chain
    result = verify_audit_chain(db)
    
    assert result["valid"] == True
    assert result["total_entries"] == 2
    assert len(result["errors"]) == 0


def test_audit_chain_detects_tampering(db: Session):
    """Test that audit chain detects tampering."""
    # Add an entry
    log_audit(db, "user1", "action1", {"data": "original"})
    
    # Get the entry and tamper with it
    entry = db.query(AuditLog).order_by(AuditLog.id.desc()).first()
    original_hash = entry.current_hash
    entry.payload = {"data": "tampered"}
    db.commit()
    
    # Verify should fail
    result = verify_audit_chain(db)
    
    assert result["valid"] == False
    assert len(result["errors"]) > 0


def test_compute_hash_deterministic():
    """Test that hash computation is deterministic."""
    prev_hash = "a" * 64
    timestamp = datetime(2026, 9, 15, 10, 0, 0)
    payload = {"key": "value", "num": 42}
    
    hash1 = compute_hash(prev_hash, timestamp, payload)
    hash2 = compute_hash(prev_hash, timestamp, payload)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256


def test_create_alerts_from_predictions(db: Session):
    """Test creating multiple alerts from predictions."""
    complaint = Complaint(
        complaint_id="TEST-ALERT-004",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=200000,
        victim_bank="HDFC",
        hop_count=3,
        mule_account_chain=[{"account_id": "TEST999", "city": "Chennai"}],
        is_synthetic=False,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    
    predictions = [
        {"city": "Chennai", "channel": "ATM", "probability": 0.6, "eta_min": 40},
        {"city": "Hyderabad", "channel": "AePS", "probability": 0.25, "eta_min": 100},
        {"city": "Bengaluru", "channel": "Crypto", "probability": 0.15, "eta_min": 10},
    ]
    
    alerts = create_alerts_from_predictions(db, complaint.id, predictions, "gang_005")
    
    assert len(alerts) == 3
    for i, alert in enumerate(alerts):
        assert alert.city == predictions[i]["city"]
        assert alert.channel.value == predictions[i]["channel"]
        assert alert.probability == predictions[i]["probability"]


def test_dispatch_alerts(db: Session):
    """Test dispatching alerts."""
    # Create undispatched alert
    complaint = Complaint(
        complaint_id="TEST-ALERT-005",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=100000,
        victim_bank="HDFC",
        hop_count=2,
        mule_account_chain=[{"account_id": "TEST111", "city": "Pune"}],
        is_synthetic=False,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    
    alert = create_alert(db, complaint.id, "Pune", "ATM", 0.8, 30, 100000)
    
    assert alert.dispatched_sms == False
    
    # Dispatch
    result = dispatch_alerts(db, [alert.id])
    
    assert result["sms"] == 1
    assert result["email"] == 1
    assert result["webhook"] == 1
    
    # Verify updated
    db.refresh(alert)
    assert alert.dispatched_sms == True
    assert alert.dispatched_email == True
    assert alert.dispatched_webhook == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
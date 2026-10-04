"""
Tests for displacement and feedback loop.
"""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from app.displacement import (
    update_policing_pressure, apply_displacement, update_gang_corridor_on_miss,
    record_outcome, get_pressure_map, find_nearest_unpoliced_city
)
from app.db import SessionLocal, Base, engine, DisplacementPressure, Gang, Complaint, Intervention, InterventionStatus, ChannelType
from app.synth import generate_all_synthetic_data


@pytest.fixture(scope="module")
def db():
    """Create a test database with synthetic data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    generate_all_synthetic_data()
    yield db
    db.close()


def test_update_policing_pressure(db: Session):
    """Test that policing pressure increases at a location."""
    update_policing_pressure("Delhi", "ATM", "gang_001", db, pressure_increase=0.3)
    
    pressure = db.query(DisplacementPressure).filter(
        DisplacementPressure.city == "Delhi",
        DisplacementPressure.channel == ChannelType.ATM,
        DisplacementPressure.gang_id == "gang_001",
    ).first()
    
    assert pressure is not None
    assert pressure.pressure_score == 0.3


def test_pressure_caps_at_max(db: Session):
    """Test that pressure caps at max_pressure (1.0)."""
    for _ in range(5):
        update_policing_pressure("Delhi", "ATM", "gang_001", db, pressure_increase=0.3, max_pressure=1.0)
    
    pressure = db.query(DisplacementPressure).filter(
        DisplacementPressure.city == "Delhi",
        DisplacementPressure.channel == ChannelType.ATM,
        DisplacementPressure.gang_id == "gang_001",
    ).first()
    
    assert pressure.pressure_score <= 1.0


def test_apply_displacement_increases_pressure(db: Session):
    """Test that displacement increases pressure at intercepted location."""
    result = apply_displacement("Delhi", "ATM", "gang_001", db)
    
    assert result["intercepted_city"] == "Delhi"
    assert result["pressure_increased"] == True
    
    # Check pressure increased
    pressure = db.query(DisplacementPressure).filter(
        DisplacementPressure.city == "Delhi",
        DisplacementPressure.channel == ChannelType.ATM,
        DisplacementPressure.gang_id == "gang_001",
    ).first()
    assert pressure.pressure_score > 0


def test_displacement_shifts_to_nearest_unpoliced(db: Session):
    """Test that displacement finds nearest unpoliced city."""
    # Set high pressure on Delhi
    update_policing_pressure("Delhi", "ATM", "gang_001", db, pressure_increase=1.0)
    
    # Set low pressure on nearby city
    update_policing_pressure("Mewat", "ATM", "gang_001", db, pressure_increase=0.0)
    
    nearest = find_nearest_unpoliced_city("Delhi", "ATM", "gang_001", db)
    
    # Should find Mewat (nearby and low pressure)
    assert nearest in ["Mewat", "Gurugram"]  # Gurugram might not be in our city list


def test_update_gang_corridor_on_miss(db: Session):
    """Test that gang corridor weights update on missed intervention."""
    # Ensure gang exists
    gang = db.query(Gang).filter(Gang.gang_id == "gang_001").first()
    assert gang is not None
    
    original_weights = gang.corridor_weights.copy()
    
    update_gang_corridor_on_miss(
        "gang_001", "Delhi", "ATM", "Mewat", "AePS", db
    )
    
    # Reload gang
    db.refresh(gang)
    
    # Weight for Mewat should increase, Delhi should decrease
    corridor = gang.corridor
    weights = gang.corridor_weights
    
    if "Mewat" in corridor and "Delhi" in corridor:
        mewat_idx = corridor.index("Mewat")
        delhi_idx = corridor.index("Delhi")
        
        assert weights[mewat_idx] >= original_weights[mewat_idx]
        assert weights[delhi_idx] <= original_weights[delhi_idx]


def test_record_outcome_intercepted(db: Session):
    """Test recording an intercepted outcome."""
    # Create a test complaint and intervention
    complaint = Complaint(
        complaint_id="TEST-DISP-001",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=100000,
        victim_bank="HDFC",
        hop_count=2,
        mule_account_chain=[{"account_id": "TEST123", "city": "Delhi"}],
        is_synthetic=False,
    )
    db.add(complaint)
    db.flush()
    
    intervention = Intervention(
        complaint_id=complaint.id,
        team_id="team_001",
        team_name="Delhi Cyber Cell",
        target_city="Delhi",
        target_channel=ChannelType.ATM,
        predicted_eta_min=30,
        travel_time_min=10,
        status=InterventionStatus.DEPLOYED,
    )
    db.add(intervention)
    db.commit()
    db.refresh(intervention)
    
    # Record interception
    result = record_outcome(
        db=db,
        complaint_id=complaint.id,
        intervention_id=intervention.id,
        intercepted=True,
        amount_recovered=80000,
    )
    
    assert result["intercepted"] == True
    assert result["amount_recovered"] == 80000
    assert "displacement" in result
    
    # Check intervention status updated
    db.refresh(intervention)
    assert intervention.status == InterventionStatus.INTERCEPTED
    assert intervention.amount_recovered == 80000


def test_record_outcome_missed(db: Session):
    """Test recording a missed outcome."""
    # Create test data
    complaint = Complaint(
        complaint_id="TEST-DISP-002",
        timestamp=datetime(2026, 9, 15, 10, 0, 0),
        fraud_type="UPI Fraud",
        amount=100000,
        victim_bank="HDFC",
        hop_count=2,
        mule_account_chain=[{"account_id": "TEST456", "city": "Mewat"}],
        withdrawal_city="Mewat",
        withdrawal_channel=ChannelType.AePS,
        is_synthetic=False,
    )
    db.add(complaint)
    db.flush()
    
    intervention = Intervention(
        complaint_id=complaint.id,
        team_id="team_001",
        team_name="Delhi Cyber Cell",
        target_city="Delhi",
        target_channel=ChannelType.ATM,
        predicted_eta_min=30,
        travel_time_min=60,
        status=InterventionStatus.DEPLOYED,
    )
    db.add(intervention)
    db.commit()
    db.refresh(intervention)
    
    # Record miss
    result = record_outcome(
        db=db,
        complaint_id=complaint.id,
        intervention_id=intervention.id,
        intercepted=False,
    )
    
    assert result["intercepted"] == False
    assert "corridor_updated" in result
    
    db.refresh(intervention)
    assert intervention.status == InterventionStatus.MISSED


def test_get_pressure_map(db: Session):
    """Test getting pressure map for visualization."""
    # Reset pressure for clean test
    from app.db import DisplacementPressure
    pressures = db.query(DisplacementPressure).filter(DisplacementPressure.gang_id == "gang_001").all()
    for p in pressures:
        p.pressure_score = 0.0
    db.commit()
    
    update_policing_pressure("Delhi", "ATM", "gang_001", db, 0.5)
    update_policing_pressure("Mewat", "AePS", "gang_001", db, 0.3)
    
    pressure_map = get_pressure_map(db, gang_id="gang_001")
    
    assert "Delhi" in pressure_map
    assert "ATM" in pressure_map["Delhi"]
    assert pressure_map["Delhi"]["ATM"] >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
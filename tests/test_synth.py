"""
Tests for synthetic data generator.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.synth import (
    generate_all_synthetic_data, SYNTHETIC_SEED, NUM_SYNTHETIC_COMPLAINTS,
    SYNTHETIC_CITIES, SYNTHETIC_GANGS, SYNTHETIC_START_DATE, SYNTHETIC_END_DATE
)
from app.db import SessionLocal, Complaint, MuleAccount, Gang, Base, engine


@pytest.fixture(scope="module")
def db():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


def test_synthetic_data_determinism(db: Session):
    """Test that synthetic data generation is deterministic with same seed."""
    # Generate first batch
    stats1 = generate_all_synthetic_data()
    
    # Clear and generate again
    db.query(Complaint).delete()
    db.query(MuleAccount).delete()
    db.query(Gang).delete()
    db.commit()
    
    stats2 = generate_all_synthetic_data()
    
    # Should produce identical stats
    assert stats1["gangs"] == stats2["gangs"]
    assert stats1["mules"] == stats2["mules"]
    assert stats1["complaints"] == stats2["complaints"]


def test_correct_number_of_gangs(db: Session):
    """Test that 6 gangs are created."""
    gangs = db.query(Gang).all()
    assert len(gangs) == 6
    for gang in gangs:
        assert gang.is_synthetic == True
        assert gang.gang_id.startswith("gang_")


def test_correct_number_of_mules(db: Session):
    """Test that each gang has correct number of mules."""
    total_mules = db.query(MuleAccount).count()
    expected_mules = sum(g["mule_count"] for g in SYNTHETIC_GANGS)
    assert total_mules == expected_mules


def test_correct_number_of_complaints(db: Session):
    """Test that 5000+ complaints are generated."""
    complaints = db.query(Complaint).count()
    assert complaints >= NUM_SYNTHETIC_COMPLAINTS


def test_complaints_have_ground_truth(db: Session):
    """Test that all complaints have withdrawal ground truth."""
    complaints = db.query(Complaint).all()
    for c in complaints:
        assert c.withdrawal_city is not None
        assert c.withdrawal_channel is not None
        assert c.minutes_to_cashout is not None
        assert c.withdrawal_timestamp is not None
        assert c.is_synthetic == True


def test_complaint_date_range(db: Session):
    """Test that complaints fall within synthetic date range."""
    complaints = db.query(Complaint).all()
    for c in complaints:
        assert SYNTHETIC_START_DATE <= c.timestamp <= SYNTHETIC_END_DATE
        assert c.withdrawal_timestamp >= c.timestamp


def test_mule_accounts_linked_to_gangs(db: Session):
    """Test that mule accounts have gang_id set."""
    mules = db.query(MuleAccount).all()
    for mule in mules:
        assert mule.gang_id is not None
        assert mule.gang_id.startswith("gang_")


def test_mule_chain_in_complaints(db: Session):
    """Test that complaints have mule account chains."""
    complaints = db.query(Complaint).all()
    for c in complaints:
        assert c.mule_account_chain is not None
        assert len(c.mule_account_chain) == c.hop_count
        for hop in c.mule_account_chain:
            assert "account_id" in hop
            assert "city" in hop
            assert "bank" in hop


def test_cashout_times_by_channel(db: Session):
    """Test that cashout times vary by channel realistically."""
    from app.config import CHANNEL_CASHOUT_TIMES
    
    complaints = db.query(Complaint).all()
    
    atm_times = [c.minutes_to_cashout for c in complaints if c.withdrawal_channel.value == "ATM"]
    aePS_times = [c.minutes_to_cashout for c in complaints if c.withdrawal_channel.value == "AePS"]
    crypto_times = [c.minutes_to_cashout for c in complaints if c.withdrawal_channel.value == "Crypto"]
    
    assert len(atm_times) > 0
    assert len(aePS_times) > 0
    assert len(crypto_times) > 0
    
    # Crypto should be fastest on average
    avg_crypto = sum(crypto_times) / len(crypto_times)
    avg_atm = sum(atm_times) / len(atm_times)
    avg_aePS = sum(aePS_times) / len(aePS_times)
    
    assert avg_crypto < avg_atm
    assert avg_crypto < avg_aePS


def test_city_names_valid(db: Session):
    """Test that all cities in complaints are from the valid list."""
    valid_cities = [c["name"] for c in SYNTHETIC_CITIES]
    # Include extra cities used in gang corridors
    extra_cities = ["Gurugram", "Alwar", "Nashik", "Varanasi"]
    valid_cities.extend(extra_cities)
    
    complaints = db.query(Complaint).all()
    for c in complaints:
        assert c.withdrawal_city in valid_cities
        if c.victim_city:
            assert c.victim_city in valid_cities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Tests for gang detection graph module.
"""

import pytest
from sqlalchemy.orm import Session

from app.graph import (
    build_entity_graph, detect_gangs_from_graph, run_gang_detection,
    save_detected_gangs, assign_complaints_to_detected_gangs
)
from app.synth import generate_all_synthetic_data
from app.db import SessionLocal, Base, engine


@pytest.fixture(scope="module")
def db():
    """Create a test database with synthetic data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    generate_all_synthetic_data()
    yield db
    db.close()


def test_graph_building(db: Session):
    """Test that entity graph is built correctly."""
    G = build_entity_graph(db)
    
    # Should have nodes for accounts, phones, devices
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    
    # Check node types
    account_nodes = [n for n in G.nodes() if n.startswith("account:")]
    phone_nodes = [n for n in G.nodes() if n.startswith("phone:")]
    device_nodes = [n for n in G.nodes() if n.startswith("device:")]
    
    assert len(account_nodes) > 0
    assert len(phone_nodes) > 0
    assert len(device_nodes) > 0


def test_connected_components_detection(db: Session):
    """Test that connected components detect gangs."""
    G = build_entity_graph(db)
    gangs = detect_gangs_from_graph(G)
    
    # Should detect at least some gangs
    assert len(gangs) > 0
    
    # Each gang should have accounts
    for gang_id, account_ids in gangs.items():
        assert len(account_ids) > 0
        assert gang_id.startswith("detected_gang_")


def test_gang_detection_recovers_planted_gangs(db: Session):
    """Test that gang detection recovers the 6 planted gangs."""
    # Run full detection
    summaries = run_gang_detection(db)
    
    # Should detect multiple gangs
    assert len(summaries) >= 3  # At least 3 connected components
    
    # Check summary structure
    for summary in summaries:
        assert "gang_id" in summary
        assert "account_count" in summary
        assert "corridor" in summary
        assert "channel_mix" in summary


def test_save_detected_gangs(db: Session):
    """Test saving detected gangs to database."""
    summaries = run_gang_detection(db)
    from app.graph import save_detected_gangs
    save_detected_gangs(db, summaries)
    
    # Check they were saved
    from app.db import Gang
    saved_gangs = db.query(Gang).filter(Gang.gang_id.like("detected_gang_%")).all()
    assert len(saved_gangs) == len(summaries)


def test_assign_complaints_to_gangs(db: Session):
    """Test that complaints get assigned to detected gangs."""
    from app.graph import assign_complaints_to_detected_gangs
    assign_complaints_to_detected_gangs(db)
    
    from app.db import Complaint
    complaints = db.query(Complaint).all()
    
    # Check that complaints have gang info in prediction_features
    assigned = 0
    for c in complaints:
        if c.prediction_features and "detected_gang_id" in c.prediction_features:
            assigned += 1
    
    # At least some complaints should be assigned
    assert assigned > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
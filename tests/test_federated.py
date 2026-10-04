"""
Tests for federated learning simulation.
"""

import pytest
from sqlalchemy.orm import Session

from app.federated import FederatedSimulator, run_federated_round, get_federated_status
from app.db import SessionLocal, Base, engine, FederatedRound
from app.synth import generate_all_synthetic_data


@pytest.fixture(scope="module")
def db():
    """Create a test database with synthetic data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    generate_all_synthetic_data()
    yield db
    db.close()


def test_federated_round_runs(db: Session):
    """Test that a federated round completes successfully."""
    result = run_federated_round(db, ["HDFC", "SBI", "ICICI"])
    
    assert result.round_number >= 1
    assert len(result.participating_banks) == 3
    assert result.raw_rows_shared == 0  # Critical: no raw data shared


def test_zero_raw_rows_shared(db: Session):
    """Test that 0 raw rows are ever shared."""
    # Run multiple rounds
    for i in range(3):
        result = run_federated_round(db, ["HDFC", "SBI"])
        assert result.raw_rows_shared == 0
    
    # Check database records
    rounds = db.query(FederatedRound).all()
    for r in rounds:
        assert r.raw_rows_shared == 0


def test_federated_status(db: Session):
    """Test getting federated learning status."""
    status = get_federated_status(db)
    
    assert "current_round" in status
    assert "total_rounds_completed" in status
    assert "recent_rounds" in status
    assert "global_prior_available" in status
    assert "privacy_guarantee" in status
    assert "0 raw rows ever shared" in status["privacy_guarantee"]


def test_global_prior_saved(db: Session):
    """Test that global prior is saved after round."""
    run_federated_round(db, ["HDFC", "SBI"])
    
    from app.config import MODEL_DIR
    prior_path = MODEL_DIR / "federated_prior_latest.json"
    assert prior_path.exists()


def test_simulator_trains_local_models(db: Session):
    """Test that simulator trains local models per bank."""
    simulator = FederatedSimulator(db)
    
    update = simulator.train_local_model("HDFC")
    
    assert update.bank == "HDFC"
    assert update.raw_rows_shared == 0
    assert "accuracy" in update.local_metrics
    assert update.num_samples >= 0


def test_aggregation_weights_by_samples(db: Session):
    """Test that FedAvg weights by number of samples."""
    simulator = FederatedSimulator(db)
    
    # Create updates with different sample counts
    from app.federated import ClientUpdate
    updates = [
        ClientUpdate(
            bank="HDFC", num_samples=1000,
            classifier_weights=[], regressor_weights=[],
            feature_importances={"feat1": 0.8, "feat2": 0.2},
            local_metrics={}
        ),
        ClientUpdate(
            bank="SBI", num_samples=100,
            classifier_weights=[], regressor_weights=[],
            feature_importances={"feat1": 0.2, "feat2": 0.8},
            local_metrics={}
        ),
    ]
    
    global_model = simulator.aggregate_updates(updates)
    
    # HDFC has 10x more samples, should dominate
    assert global_model.avg_feature_importances["feat1"] > 0.5
    assert global_model.avg_feature_importances["feat2"] < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
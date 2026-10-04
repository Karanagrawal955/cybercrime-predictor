"""
Tests for intervention optimizer.
"""

import pytest
from app.optimizer import (
    optimize_intervention, optimize_for_multiple_complaints,
    estimate_travel_time_km, calculate_expected_recovery,
    Team, Prediction, ComplaintContext
)
from app.config import TEAM_BASE_LOCATIONS, CITY_COORDS


def test_travel_time_estimation():
    """Test travel time estimation between cities."""
    # Same city
    time = estimate_travel_time_km("Delhi", "Delhi")
    assert time == 5  # Local deployment
    
    # Known cities
    time = estimate_travel_time_km("Delhi", "Mumbai")
    assert time > 60  # Should be several hours
    
    # Unknown city defaults to 2 hours
    time = estimate_travel_time_km("Unknown", "Delhi")
    assert time == 120


def test_expected_recovery_calculation():
    """Test expected recovery calculation."""
    # High probability, arrive before cashout
    recovery = calculate_expected_recovery(100000, 0.8, True, "ATM")
    assert recovery == 100000 * 0.8 * 0.85  # 68000
    
    # Low probability, arrive after cashout
    recovery = calculate_expected_recovery(100000, 0.3, False, "ATM")
    assert recovery == 100000 * 0.3 * 0.2  # 6000
    
    # Crypto always high recovery if exchange freeze
    recovery = calculate_expected_recovery(100000, 0.5, True, "Crypto")
    assert recovery == 100000 * 0.5 * 0.9  # 45000


def test_optimizer_never_assigns_team_twice():
    """Test that optimizer never assigns one team to two predictions."""
    predictions = [
        {"city": "Delhi", "channel": "ATM", "probability": 0.6, "eta_min": 45, "rank": 1},
        {"city": "Mewat", "channel": "AePS", "probability": 0.3, "eta_min": 120, "rank": 2},
        {"city": "Mumbai", "channel": "Crypto", "probability": 0.1, "eta_min": 15, "rank": 3},
    ]
    
    result = optimize_intervention(
        complaint_id="TEST-001",
        amount=500000,
        predictions=predictions,
        team_count=2,  # Only 2 teams
    )
    
    # Check no duplicate team IDs (excluding None for crypto)
    team_ids = [p.team_id for p in result.plans if p.team_id is not None]
    assert len(team_ids) == len(set(team_ids)), "Team assigned twice!"


def test_crypto_requires_no_team():
    """Test that crypto interventions don't require a team."""
    predictions = [
        {"city": "Delhi", "channel": "Crypto", "probability": 0.8, "eta_min": 15, "rank": 1},
    ]
    
    result = optimize_intervention(
        complaint_id="TEST-001",
        amount=100000,
        predictions=predictions,
        team_count=0,  # No teams available
    )
    
    assert len(result.plans) == 1
    assert result.plans[0].action_type == "exchange_freeze"
    assert result.plans[0].team_id is None
    assert result.plans[0].travel_time_min == 0


def test_optimizer_prioritizes_high_expected_recovery():
    """Test that optimizer picks highest expected recovery first."""
    predictions = [
        {"city": "Delhi", "channel": "ATM", "probability": 0.9, "eta_min": 10, "rank": 1},  # High prob, close
        {"city": "Mumbai", "channel": "ATM", "probability": 0.5, "eta_min": 300, "rank": 2},  # Far, low prob
    ]
    
    result = optimize_intervention(
        complaint_id="TEST-001",
        amount=1000000,
        predictions=predictions,
        team_count=1,
    )
    
    # Should pick Delhi (high prob, can arrive in time)
    assert result.plans[0].target_city == "Delhi"


def test_multiple_complaints_team_sharing():
    """Test that teams are shared across multiple complaints."""
    complaints = [
        {
            "complaint_id": "C1",
            "amount": 500000,
            "predictions": [
                {"city": "Delhi", "channel": "ATM", "probability": 0.8, "eta_min": 30, "rank": 1},
            ]
        },
        {
            "complaint_id": "C2",
            "amount": 100000,
            "predictions": [
                {"city": "Mumbai", "channel": "ATM", "probability": 0.6, "eta_min": 30, "rank": 1},
            ]
        },
    ]
    
    results = optimize_for_multiple_complaints(complaints, team_count=1)
    
    # Only 1 team, should be assigned to highest value complaint (C1)
    team_assignments = []
    for r in results:
        for p in r.plans:
            if p.team_id:
                team_assignments.append(p.team_id)
    
    # Team should only be used once
    assert len(team_assignments) <= 1


def test_optimizer_output_structure():
    """Test that optimizer returns correct structure."""
    predictions = [
        {"city": "Delhi", "channel": "ATM", "probability": 0.6, "eta_min": 45, "rank": 1},
    ]
    
    result = optimize_intervention(
        complaint_id="TEST-001",
        amount=500000,
        predictions=predictions,
        team_count=5,
    )
    
    assert result.complaint_id == "TEST-001"
    assert isinstance(result.plans, list)
    assert isinstance(result.total_expected_recovery, float)
    assert isinstance(result.teams_used, int)
    assert isinstance(result.unassigned_predictions, list)
    
    for plan in result.plans:
        assert plan.complaint_id == "TEST-001"
        assert plan.target_city in ["Delhi"]
        assert plan.action_type in ["physical", "exchange_freeze"]
        assert plan.expected_recovery >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
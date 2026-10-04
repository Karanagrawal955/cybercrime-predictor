"""
Intervention optimizer for deploying teams to predicted cash-out locations.
Greedy assignment maximizing expected recovered rupees.
Race clock: travel time vs predicted ETA.
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.config import (
    TEAM_BASE_LOCATIONS, CITY_COORDS, CHANNELS,
    settings
)
from app.db import ChannelType, InterventionStatus


@dataclass
class Team:
    """Police team available for deployment."""
    id: str
    name: str
    base_city: str
    base_lat: float
    base_lon: float
    is_available: bool = True
    current_assignment: Optional[str] = None  # complaint_id


@dataclass
class Prediction:
    """Single prediction with probability and ETA."""
    city: str
    channel: str
    probability: float
    eta_min: int
    rank: int


@dataclass
class ComplaintContext:
    """Context for optimization."""
    complaint_id: str
    amount: float
    predictions: List[Prediction]
    timestamp: str


@dataclass
class InterventionPlan:
    """Single intervention action."""
    complaint_id: str
    team_id: Optional[str]
    team_name: Optional[str]
    target_city: str
    target_channel: str
    predicted_eta_min: int
    travel_time_min: int
    arrival_before_cashout: bool
    probability: float
    expected_recovery: float
    reason: str
    action_type: str  # "physical" or "exchange_freeze"


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    complaint_id: str
    plans: List[InterventionPlan]
    total_expected_recovery: float
    teams_used: int
    unassigned_predictions: List[Prediction]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def estimate_travel_time_km(origin_city: str, dest_city: str, speed_kmph: float = 40.0) -> int:
    """Estimate travel time in minutes between cities."""
    if origin_city == dest_city:
        return 5  # Local deployment
    
    if origin_city not in CITY_COORDS or dest_city not in CITY_COORDS:
        return 120  # Default 2 hours for unknown
    
    lat1, lon1 = CITY_COORDS[origin_city]
    lat2, lon2 = CITY_COORDS[dest_city]
    distance = haversine_km(lat1, lon1, lat2, lon2)
    time_hours = distance / speed_kmph
    return int(time_hours * 60)


def get_available_teams(team_count: int, excluded_complaint_ids: List[str] = None) -> List[Team]:
    """Get available teams for deployment."""
    teams = []
    for i, t in enumerate(TEAM_BASE_LOCATIONS[:team_count]):
        teams.append(Team(
            id=t["id"],
            name=t["name"],
            base_city=t["city"],
            base_lat=t["lat"],
            base_lon=t["lon"],
            is_available=True,
        ))
    return teams


def calculate_expected_recovery(
    amount: float, 
    probability: float, 
    arrival_before_cashout: bool,
    channel: str
) -> float:
    """Calculate expected recovery based on probability and timing."""
    base_recovery = amount * probability
    
    if not arrival_before_cashout:
        # Reduced recovery if arriving after cash-out
        if channel == "Crypto":
            return base_recovery * 0.1  # Exchange freeze might still work
        return base_recovery * 0.2  # Physical intervention after cash-out
    
    # Channel-specific recovery rates when arriving in time
    recovery_rates = {
        "ATM": 0.85,
        "AePS": 0.75,
        "Crypto": 0.90,  # Exchange freeze is very effective
    }
    return base_recovery * recovery_rates.get(channel, 0.5)


def greedy_optimize(
    complaint: ComplaintContext,
    teams: List[Team],
    max_interventions_per_complaint: int = 3,
) -> OptimizationResult:
    """
    Greedy optimization: assign teams to highest expected recovery predictions.
    Each team can only be assigned once.
    """
    # Sort predictions by expected value (probability * amount)
    scored_predictions = []
    for pred in complaint.predictions:
        travel_time = estimate_travel_time_km(
            teams[0].base_city if teams else pred.city, 
            pred.city
        ) if teams else 0
        
        arrival_before = travel_time <= pred.eta_min
        expected_recovery = calculate_expected_recovery(
            complaint.amount, pred.probability, arrival_before, pred.channel
        )
        
        scored_predictions.append({
            "prediction": pred,
            "travel_time": travel_time,
            "arrival_before": arrival_before,
            "expected_recovery": expected_recovery,
        })
    
    # Sort by expected recovery descending
    scored_predictions.sort(key=lambda x: x["expected_recovery"], reverse=True)
    
    # Assign teams greedily
    plans = []
    used_teams = set()
    unassigned = []
    
    for scored in scored_predictions:
        if len(plans) >= max_interventions_per_complaint:
            unassigned.append(scored["prediction"])
            continue
        
        pred = scored["prediction"]
        
        # Find best available team
        best_team = None
        best_travel_time = float('inf')
        
        for team in teams:
            if team.id in used_teams:
                continue
            travel_time = estimate_travel_time_km(team.base_city, pred.city)
            if travel_time < best_travel_time:
                best_travel_time = travel_time
                best_team = team
        
        if best_team:
            arrival_before = best_travel_time <= pred.eta_min
            expected_recovery = calculate_expected_recovery(
                complaint.amount, pred.probability, arrival_before, pred.channel
            )
            
            # Determine action type
            action_type = "exchange_freeze" if pred.channel == "Crypto" else "physical"
            
            # For crypto, no team needed
            if pred.channel == "Crypto":
                team_id = None
                team_name = "Exchange Freeze Request"
                travel_time = 0
                arrival_before = True
            else:
                team_id = best_team.id
                team_name = best_team.name
                travel_time = best_travel_time
                used_teams.add(best_team.id)
            
            reason = _generate_reason(pred, best_travel_time, pred.eta_min, arrival_before, pred.channel)
            
            plan = InterventionPlan(
                complaint_id=complaint.complaint_id,
                team_id=team_id,
                team_name=team_name,
                target_city=pred.city,
                target_channel=pred.channel,
                predicted_eta_min=pred.eta_min,
                travel_time_min=travel_time if pred.channel != "Crypto" else 0,
                arrival_before_cashout=arrival_before,
                probability=pred.probability,
                expected_recovery=expected_recovery,
                reason=reason,
                action_type=action_type,
            )
            plans.append(plan)
        else:
            # No team available for this prediction
            if pred.channel != "Crypto":
                unassigned.append(pred)
            else:
                # Crypto doesn't need a team
                arrival_before = True
                expected_recovery = calculate_expected_recovery(
                    complaint.amount, pred.probability, arrival_before, pred.channel
                )
                plan = InterventionPlan(
                    complaint_id=complaint.complaint_id,
                    team_id=None,
                    team_name="Exchange Freeze Request",
                    target_city=pred.city,
                    target_channel=pred.channel,
                    predicted_eta_min=pred.eta_min,
                    travel_time_min=0,
                    arrival_before_cashout=True,
                    probability=pred.probability,
                    expected_recovery=expected_recovery,
                    reason=f"Crypto off-ramp at {pred.city}: Request exchange freeze for USDT P2P",
                    action_type="exchange_freeze",
                )
                plans.append(plan)
    
    total_expected = sum(p.expected_recovery for p in plans)
    
    return OptimizationResult(
        complaint_id=complaint.complaint_id,
        plans=plans,
        total_expected_recovery=total_expected,
        teams_used=len(used_teams),
        unassigned_predictions=unassigned,
    )


def _generate_reason(
    pred: Prediction, 
    travel_time: int, 
    eta_min: int, 
    arrival_before: bool,
    channel: str
) -> str:
    """Generate human-readable reason for intervention."""
    if channel == "Crypto":
        return f"Crypto off-ramp at {pred.city}: Request exchange freeze (no team needed)"
    
    if arrival_before:
        return (
            f"Team can reach {pred.city} in {travel_time} min, "
            f"cash-out predicted in {eta_min} min. "
            f"Arrive {eta_min - travel_time} min early. Probability: {pred.probability:.1%}"
        )
    else:
        return (
            f"Team reaches {pred.city} in {travel_time} min, "
            f"but cash-out predicted in {eta_min} min. "
            f"Late by {travel_time - eta_min} min. Probability: {pred.probability:.1%}"
        )


def optimize_intervention(
    complaint_id: str,
    amount: float,
    predictions: List[Dict],
    team_count: int = 10,
    excluded_complaint_ids: List[str] = None,
) -> OptimizationResult:
    """
    Main optimization entry point.
    Input: complaint details and predictions from model.
    Output: Ranked intervention plan.
    """
    # Convert predictions
    pred_objects = [
        Prediction(
            city=p["city"],
            channel=p["channel"],
            probability=p["probability"],
            eta_min=p["eta_min"],
            rank=p.get("rank", i+1),
        )
        for i, p in enumerate(predictions)
    ]
    
    complaint_ctx = ComplaintContext(
        complaint_id=complaint_id,
        amount=amount,
        predictions=pred_objects,
        timestamp="",  # Not used in current logic
    )
    
    teams = get_available_teams(team_count, excluded_complaint_ids)
    result = greedy_optimize(complaint_ctx, teams)
    
    return result


def optimize_for_multiple_complaints(
    complaints: List[Dict],
    team_count: int = 10,
) -> List[OptimizationResult]:
    """
    Optimize interventions for multiple complaints simultaneously.
    Teams are shared across complaints.
    """
    teams = get_available_teams(team_count)
    results = []
    
    # Sort complaints by amount * max probability (highest value first)
    scored_complaints = []
    for c in complaints:
        max_prob = max(p["probability"] for p in c["predictions"])
        scored_complaints.append((c, max_prob * c["amount"]))
    
    scored_complaints.sort(key=lambda x: x[1], reverse=True)
    
    used_teams = set()
    
    for complaint_data, _ in scored_complaints:
        # Filter available teams
        available_teams = [t for t in teams if t.id not in used_teams]
        
        pred_objects = [
            Prediction(
                city=p["city"],
                channel=p["channel"],
                probability=p["probability"],
                eta_min=p["eta_min"],
                rank=p.get("rank", i+1),
            )
            for i, p in enumerate(complaint_data["predictions"])
        ]
        
        complaint_ctx = ComplaintContext(
            complaint_id=complaint_data["complaint_id"],
            amount=complaint_data["amount"],
            predictions=pred_objects,
            timestamp="",
        )
        
        # Custom greedy with pre-assigned teams
        result = _greedy_with_teams(complaint_ctx, available_teams, used_teams)
        results.append(result)
    
    return results


def _greedy_with_teams(
    complaint: ComplaintContext,
    teams: List[Team],
    used_teams: set,
    max_interventions: int = 3,
) -> OptimizationResult:
    """Greedy optimization with pre-filtered teams."""
    scored_predictions = []
    for pred in complaint.predictions:
        # Use closest team for travel estimate
        best_travel = min(
            (estimate_travel_time_km(t.base_city, pred.city) for t in teams),
            default=120
        ) if teams else 120
        
        arrival_before = best_travel <= pred.eta_min
        expected_recovery = calculate_expected_recovery(
            complaint.amount, pred.probability, arrival_before, pred.channel
        )
        
        scored_predictions.append({
            "prediction": pred,
            "travel_time": best_travel,
            "arrival_before": arrival_before,
            "expected_recovery": expected_recovery,
        })
    
    scored_predictions.sort(key=lambda x: x["expected_recovery"], reverse=True)
    
    plans = []
    unassigned = []
    
    for scored in scored_predictions:
        if len(plans) >= max_interventions:
            unassigned.append(scored["prediction"])
            continue
        
        pred = scored["prediction"]
        
        # Find best available team
        best_team = None
        best_travel = float('inf')
        
        for team in teams:
            if team.id in used_teams:
                continue
            travel = estimate_travel_time_km(team.base_city, pred.city)
            if travel < best_travel:
                best_travel = travel
                best_team = team
        
        if best_team or pred.channel == "Crypto":
            arrival_before = best_travel <= pred.eta_min if best_team else True
            expected_recovery = calculate_expected_recovery(
                complaint.amount, pred.probability, arrival_before, pred.channel
            )
            
            action_type = "exchange_freeze" if pred.channel == "Crypto" else "physical"
            
            if pred.channel == "Crypto":
                team_id = None
                team_name = "Exchange Freeze Request"
                travel_time = 0
            else:
                team_id = best_team.id
                team_name = best_team.name
                travel_time = best_travel
                used_teams.add(best_team.id)
            
            reason = _generate_reason(pred, travel_time, pred.eta_min, arrival_before, pred.channel)
            
            plan = InterventionPlan(
                complaint_id=complaint.complaint_id,
                team_id=team_id,
                team_name=team_name,
                target_city=pred.city,
                target_channel=pred.channel,
                predicted_eta_min=pred.eta_min,
                travel_time_min=travel_time,
                arrival_before_cashout=arrival_before,
                probability=pred.probability,
                expected_recovery=expected_recovery,
                reason=reason,
                action_type=action_type,
            )
            plans.append(plan)
        else:
            unassigned.append(pred)
    
    total_expected = sum(p.expected_recovery for p in plans)
    
    return OptimizationResult(
        complaint_id=complaint.complaint_id,
        plans=plans,
        total_expected_recovery=total_expected,
        teams_used=len([p for p in plans if p.team_id]),
        unassigned_predictions=unassigned,
    )


if __name__ == "__main__":
    # Quick test
    test_predictions = [
        {"city": "Delhi", "channel": "ATM", "probability": 0.65, "eta_min": 45, "rank": 1},
        {"city": "Mewat", "channel": "AePS", "probability": 0.25, "eta_min": 120, "rank": 2},
        {"city": "Mumbai", "channel": "Crypto", "probability": 0.10, "eta_min": 15, "rank": 3},
    ]
    
    result = optimize_intervention(
        complaint_id="TEST-001",
        amount=500000,
        predictions=test_predictions,
        team_count=5,
    )
    
    print(f"Complaint: {result.complaint_id}")
    print(f"Total Expected Recovery: ₹{result.total_expected_recovery:,.0f}")
    print(f"Teams Used: {result.teams_used}")
    for plan in result.plans:
        print(f"  {plan.team_name} -> {plan.target_city} ({plan.target_channel}) "
              f"ETA:{plan.predicted_eta_min}min Travel:{plan.travel_time_min}min "
              f"Recovery:₹{plan.expected_recovery:,.0f} [{plan.action_type}]")
        print(f"    Reason: {plan.reason}")
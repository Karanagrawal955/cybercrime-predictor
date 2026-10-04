"""
Displacement and feedback loop for intervention outcomes.
Records intervention results, updates policing pressure, shifts probability mass.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import CITY_NAMES, CHANNELS, CITY_COORDS
from app.db import (
    Complaint, Intervention, DisplacementPressure, Gang, Alert, AuditLog, InterventionStatus, ChannelType
)
from app.optimizer import haversine_km


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    import math
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def find_nearest_unpoliced_city(
    city: str, 
    channel: str, 
    gang_id: str, 
    db: Session,
    max_distance_km: float = 100.0
) -> Optional[str]:
    """Find nearest city with low policing pressure for displacement."""
    if city not in CITY_COORDS:
        return None
    
    lat1, lon1 = CITY_COORDS[city]
    
    # Get pressure scores for all cities for this channel/gang
    pressures = db.query(DisplacementPressure).filter(
        DisplacementPressure.channel == channel,
        DisplacementPressure.gang_id == gang_id,
    ).all()
    
    pressure_map = {p.city: p.pressure_score for p in pressures}
    
    # Find candidate cities with low pressure
    candidates = []
    for candidate_city in CITY_NAMES:
        if candidate_city == city:
            continue
        if candidate_city not in CITY_COORDS:
            continue
        
        lat2, lon2 = CITY_COORDS[candidate_city]
        distance = haversine_km(lat1, lon1, lat2, lon2)
        
        if distance <= max_distance_km:
            pressure = pressure_map.get(candidate_city, 0.0)
            candidates.append((candidate_city, distance, pressure))
    
    if not candidates:
        return None
    
    # Sort by pressure (ascending) then distance (ascending)
    candidates.sort(key=lambda x: (x[2], x[1]))
    return candidates[0][0]


def update_policing_pressure(
    city: str,
    channel: str,
    gang_id: str,
    db: Session,
    pressure_increase: float = 0.3,
    max_pressure: float = 1.0,
):
    """Increase policing pressure at a location."""
    pressure = db.query(DisplacementPressure).filter(
        DisplacementPressure.city == city,
        DisplacementPressure.channel == channel,
        DisplacementPressure.gang_id == gang_id,
    ).first()
    
    if pressure:
        pressure.pressure_score = min(max_pressure, pressure.pressure_score + pressure_increase)
        pressure.updated_at = datetime.utcnow()
    else:
        pressure = DisplacementPressure(
            city=city,
            channel=channel,
            gang_id=gang_id,
            pressure_score=pressure_increase,
        )
        db.add(pressure)
    
    db.commit()


def decay_pressures(db: Session, decay_rate: float = 0.05):
    """Decay all pressure scores over time (call periodically)."""
    pressures = db.query(DisplacementPressure).all()
    for p in pressures:
        p.pressure_score = max(0.0, p.pressure_score - decay_rate)
    db.commit()


def apply_displacement(
    intercepted_city: str,
    intercepted_channel: str,
    gang_id: str,
    db: Session,
    displacement_strength: float = 0.5,
) -> Dict:
    """
    Apply displacement effect when interception occurs.
    Reduces probability at intercepted location, shifts to nearest unpoliced hub.
    """
    # Increase pressure at intercepted location
    update_policing_pressure(
        intercepted_city, intercepted_channel, gang_id, db,
        pressure_increase=0.4
    )
    
    # Find displacement target
    displaced_to = find_nearest_unpoliced_city(
        intercepted_city, intercepted_channel, gang_id, db
    )
    
    result = {
        "intercepted_city": intercepted_city,
        "intercepted_channel": intercepted_channel,
        "pressure_increased": True,
        "displaced_to": displaced_to,
        "displacement_strength": displacement_strength,
    }
    
    if displaced_to:
        # Increase pressure slightly at displacement target too (anticipation)
        update_policing_pressure(
            displaced_to, intercepted_channel, gang_id, db,
            pressure_increase=0.1
        )
        result["displacement_anticipation"] = True
    
    return result


def update_gang_corridor_on_miss(
    gang_id: str,
    attempted_city: str,
    attempted_channel: str,
    actual_city: str,
    actual_channel: str,
    db: Session,
    learning_rate: float = 0.1,
):
    """Update gang corridor weights when intervention misses."""
    gang = db.query(Gang).filter(Gang.gang_id == gang_id).first()
    if not gang:
        return
    
    corridor = gang.corridor or []
    weights = gang.corridor_weights or []
    
    if not corridor or not weights or len(corridor) != len(weights):
        return
    
    # Increase weight for actual withdrawal city
    if actual_city in corridor:
        idx = corridor.index(actual_city)
        weights[idx] = min(1.0, weights[idx] + learning_rate)
    
    # Decrease weight for attempted (wrong) city if in corridor
    if attempted_city in corridor and attempted_city != actual_city:
        idx = corridor.index(attempted_city)
        weights[idx] = max(0.0, weights[idx] - learning_rate * 0.5)
    
    # Renormalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    
    gang.corridor_weights = weights
    gang.updated_at = datetime.utcnow()
    db.commit()


def record_outcome(
    db: Session,
    complaint_id: str,
    intervention_id: int,
    intercepted: bool,
    amount_recovered: float = 0.0,
    notes: str = "",
) -> Dict:
    """
    Record intervention outcome and trigger displacement/feedback.
    Returns summary of actions taken.
    """
    intervention = db.query(Intervention).filter(Intervention.id == intervention_id).first()
    if not intervention:
        return {"error": "Intervention not found"}
    
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        return {"error": "Complaint not found"}
    
    # Update intervention
    intervention.status = InterventionStatus.INTERCEPTED if intercepted else InterventionStatus.MISSED
    intervention.completed_at = datetime.utcnow()
    intervention.amount_recovered = amount_recovered
    intervention.notes = notes
    
    # Get gang ID from complaint
    gang_id = None
    if complaint.prediction_features and "detected_gang_id" in complaint.prediction_features:
        gang_id = complaint.prediction_features["detected_gang_id"]
    
    result = {
        "intervention_id": intervention_id,
        "intercepted": intercepted,
        "amount_recovered": amount_recovered,
        "gang_id": gang_id,
    }
    
    if intercepted:
        # Apply displacement
        displacement_result = apply_displacement(
            intervention.target_city,
            intervention.target_channel.value,
            gang_id or "unknown",
            db,
        )
        result["displacement"] = displacement_result
        
        # Update alert status
        alerts = db.query(Alert).filter(Alert.complaint_id == complaint_id).all()
        for alert in alerts:
            alert.dispatched_sms = True
            alert.dispatched_email = True
    
    else:
        # Update gang corridor on miss (if we have actual withdrawal data)
        if complaint.withdrawal_city and complaint.withdrawal_channel:
            update_gang_corridor_on_miss(
                gang_id or "unknown",
                intervention.target_city,
                intervention.target_channel.value,
                complaint.withdrawal_city,
                complaint.withdrawal_channel.value,
                db,
            )
            result["corridor_updated"] = True
    
    db.commit()
    return result


def get_pressure_map(db: Session, gang_id: str = None, channel: str = None) -> Dict:
    """Get current pressure map for visualization."""
    query = db.query(DisplacementPressure)
    
    if gang_id:
        query = query.filter(DisplacementPressure.gang_id == gang_id)
    if channel:
        query = query.filter(DisplacementPressure.channel == channel)
    
    pressures = query.all()
    
    pressure_map = defaultdict(dict)
    for p in pressures:
        pressure_map[p.city][p.channel.value] = p.pressure_score
    
    return dict(pressure_map)


def get_displacement_summary(db: Session) -> Dict:
    """Get summary of displacement effects."""
    pressures = db.query(DisplacementPressure).all()
    
    total_pressure = sum(p.pressure_score for p in pressures)
    high_pressure_cities = [
        {"city": p.city, "channel": p.channel.value, "pressure": p.pressure_score, "gang": p.gang_id}
        for p in pressures if p.pressure_score > 0.5
    ]
    
    return {
        "total_pressure_points": len(pressures),
        "total_pressure_score": total_pressure,
        "high_pressure_locations": high_pressure_cities,
    }


if __name__ == "__main__":
    from app.db import SessionLocal
    db = SessionLocal()
    try:
        # Test displacement
        print("Testing displacement...")
        result = apply_displacement("Delhi", "ATM", "gang_001", db)
        print(f"Displacement result: {json.dumps(result, indent=2)}")
        
        # Check pressure map
        pressure_map = get_pressure_map(db, gang_id="gang_001")
        print(f"Pressure map: {json.dumps(pressure_map, indent=2)}")
    finally:
        db.close()
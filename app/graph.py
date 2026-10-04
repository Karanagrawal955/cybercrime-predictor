"""
Gang detection using graph-based connected components.
Builds a graph linking accounts, phones and devices from complaints.
Uses community detection to assign gang IDs.
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db import (
    Complaint, MuleAccount, ComplaintMuleLink, Gang, Base, engine, SessionLocal
)
from app.config import SYNTHETIC_GANGS, CITY_NAMES


def build_entity_graph(db: Session) -> nx.Graph:
    """
    Build a bipartite graph linking:
    - Mule accounts (nodes)
    - Phone numbers (nodes)
    - Device IDs (nodes)
    
    Edges exist between account-phone and account-device.
    Connected components = gangs.
    """
    G = nx.Graph()
    
    # Get all mule accounts with their links
    mules = db.query(MuleAccount).all()
    
    for mule in mules:
        account_node = f"account:{mule.account_id}"
        G.add_node(account_node, type="account", account_id=mule.account_id, 
                   bank=mule.bank, city=mule.city, gang_id=mule.gang_id)
        
        # Phone node
        if mule.phone_number:
            phone_node = f"phone:{mule.phone_number}"
            G.add_node(phone_node, type="phone", phone_number=mule.phone_number)
            G.add_edge(account_node, phone_node, weight=1.0)
        
        # Device node
        if mule.device_id:
            device_node = f"device:{mule.device_id}"
            G.add_node(device_node, type="device", device_id=mule.device_id)
            G.add_edge(account_node, device_node, weight=1.0)
    
    return G


def detect_gangs_from_graph(G: nx.Graph) -> Dict[str, List[str]]:
    """
    Detect gangs as connected components in the graph.
    Returns mapping of component_id -> list of account_ids.
    """
    components = list(nx.connected_components(G))
    gangs = {}
    
    for i, component in enumerate(components):
        # Extract account nodes from this component
        accounts = [n for n in component if n.startswith("account:")]
        if accounts:
            gang_id = f"detected_gang_{i:03d}"
            account_ids = [n.replace("account:", "") for n in accounts]
            gangs[gang_id] = account_ids
    
    return gangs


def analyze_gang_component(G: nx.Graph, component_nodes: Set[str]) -> Dict:
    """Analyze a single gang component for summary statistics."""
    accounts = [n for n in component_nodes if n.startswith("account:")]
    phones = [n for n in component_nodes if n.startswith("phone:")]
    devices = [n for n in component_nodes if n.startswith("device:")]
    
    # Get account details
    account_details = []
    for acc_node in accounts:
        node_data = G.nodes[acc_node]
        account_details.append(node_data)
    
    # Corridor analysis
    cities = [d.get("city") for d in account_details if d.get("city")]
    city_counts = defaultdict(int)
    for city in cities:
        city_counts[city] += 1
    
    corridor = sorted(city_counts.keys(), key=lambda c: city_counts[c], reverse=True)
    total = sum(city_counts.values())
    corridor_weights = [city_counts[c] / total for c in corridor] if total > 0 else []
    
    # Channel mix (from complaints linked to these accounts)
    # This would need complaint data - simplified here
    channel_mix = {"ATM": 0.5, "AePS": 0.3, "Crypto": 0.2}  # Default
    
    # Shared phones/devices
    shared_phones = [p for p in phones if G.degree(p) > 1]
    shared_devices = [d for d in devices if G.degree(d) > 1]
    
    return {
        "account_count": len(accounts),
        "phone_count": len(phones),
        "device_count": len(devices),
        "shared_phones": len(shared_phones),
        "shared_devices": len(shared_devices),
        "corridor": corridor[:3],  # Top 3 cities
        "corridor_weights": corridor_weights[:3],
        "channel_mix": channel_mix,
        "base_city": corridor[0] if corridor else None,
    }


def run_gang_detection(db: Session) -> List[Dict]:
    """
    Run full gang detection pipeline.
    Returns list of detected gang summaries.
    """
    print("Building entity graph...")
    G = build_entity_graph(db)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("Detecting connected components...")
    gangs = detect_gangs_from_graph(G)
    print(f"Detected {len(gangs)} gangs")
    
    # Analyze each gang
    gang_summaries = []
    for gang_id, account_ids in gangs.items():
        # Get component nodes
        component_nodes = set()
        for acc_id in account_ids:
            acc_node = f"account:{acc_id}"
            if acc_node in G:
                component_nodes.update(nx.node_connected_component(G, acc_node))
        
        analysis = analyze_gang_component(G, component_nodes)
        analysis["gang_id"] = gang_id
        analysis["account_ids"] = account_ids
        gang_summaries.append(analysis)
    
    return gang_summaries


def save_detected_gangs(db: Session, gang_summaries: List[Dict]):
    """Save detected gangs to database."""
    for summary in gang_summaries:
        # Check if gang already exists
        existing = db.query(Gang).filter(Gang.gang_id == summary["gang_id"]).first()
        if existing:
            # Update
            existing.corridor = summary["corridor"]
            existing.corridor_weights = summary["corridor_weights"]
            existing.channel_mix = summary["channel_mix"]
            existing.mule_count = summary["account_count"]
            existing.base_city = summary["base_city"]
            existing.updated_at = func.now()
        else:
            # Create new
            gang = Gang(
                gang_id=summary["gang_id"],
                name=f"Detected Gang {summary['gang_id']}",
                corridor=summary["corridor"],
                corridor_weights=summary["corridor_weights"],
                channel_mix=summary["channel_mix"],
                mule_count=summary["account_count"],
                base_city=summary["base_city"],
                is_synthetic=True,
            )
            db.add(gang)
    
    db.commit()


def assign_complaints_to_detected_gangs(db: Session):
    """Assign complaints to detected gangs based on mule account links."""
    # Get all mule accounts with their gang_id
    mules = db.query(MuleAccount).all()
    account_to_gang = {m.account_id: m.gang_id for m in mules if m.gang_id}
    
    # For each complaint, find the gang from its mule chain
    complaints = db.query(Complaint).all()
    for complaint in complaints:
        gang_ids = set()
        for hop in complaint.mule_account_chain:
            acc_id = hop.get("account_id")
            if acc_id in account_to_gang:
                gang_ids.add(account_to_gang[acc_id])
        
        if len(gang_ids) == 1:
            complaint.predicted_city = None  # Will be filled by model
            # Store gang_id in prediction_features for reference
            if not complaint.prediction_features:
                complaint.prediction_features = {}
            complaint.prediction_features["detected_gang_id"] = list(gang_ids)[0]
    
    db.commit()


def get_gang_summary(db: Session, gang_id: str) -> Optional[Dict]:
    """Get detailed summary for a specific gang."""
    gang = db.query(Gang).filter(Gang.gang_id == gang_id).first()
    if not gang:
        return None
    
    # Get mules
    mules = db.query(MuleAccount).filter(MuleAccount.gang_id == gang_id).all()
    
    # Get complaints linked to this gang's mules
    from sqlalchemy import func
    mule_ids = [m.id for m in mules]
    
    if mule_ids:
        complaint_count = db.query(func.count(Complaint.id)).join(
            ComplaintMuleLink, Complaint.id == ComplaintMuleLink.complaint_id
        ).filter(ComplaintMuleLink.mule_account_id.in_(mule_ids)).scalar() or 0
        
        total_amount = db.query(func.sum(Complaint.amount)).join(
            ComplaintMuleLink, Complaint.id == ComplaintMuleLink.complaint_id
        ).filter(ComplaintMuleLink.mule_account_id.in_(mule_ids)).scalar() or 0.0
    else:
        complaint_count = 0
        total_amount = 0.0
    
    # Channel mix from complaints
    channel_counts = defaultdict(int)
    if mule_ids:
        channels = db.query(Complaint.withdrawal_channel).join(
            ComplaintMuleLink, Complaint.id == ComplaintMuleLink.complaint_id
        ).filter(ComplaintMuleLink.mule_account_id.in_(mule_ids)).all()
        for (ch,) in channels:
            if ch:
                channel_counts[ch.value] += 1
    
    total_ch = sum(channel_counts.values())
    channel_mix = {k: v/total_ch for k, v in channel_counts.items()} if total_ch > 0 else {}
    
    return {
        "gang_id": gang.gang_id,
        "name": gang.name,
        "corridor": gang.corridor,
        "corridor_weights": gang.corridor_weights,
        "channel_mix": channel_mix or gang.channel_mix,
        "mule_count": len(mules),
        "base_city": gang.base_city,
        "total_complaints": complaint_count,
        "total_amount": float(total_amount),
        "mules": [
            {
                "account_id": m.account_id,
                "phone": m.phone_number,
                "device": m.device_id,
                "bank": m.bank,
                "city": m.city,
            }
            for m in mules
        ],
    }


def list_all_gangs(db: Session) -> List[Dict]:
    """List all gangs with summary stats."""
    gangs = db.query(Gang).all()
    summaries = []
    for gang in gangs:
        summary = get_gang_summary(db, gang.gang_id)
        if summary:
            summaries.append(summary)
    return summaries


if __name__ == "__main__":
    db = SessionLocal()
    try:
        summaries = run_gang_detection(db)
        save_detected_gangs(db, summaries)
        assign_complaints_to_detected_gangs(db)
        print("Gang detection complete!")
        for s in summaries:
            print(f"  {s['gang_id']}: {s['account_count']} accounts, corridor={s['corridor']}")
    finally:
        db.close()
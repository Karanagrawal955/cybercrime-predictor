"""
Synthetic data generator for cybercrime withdrawal prediction.
ALL DATA GENERATED HERE IS SYNTHETIC AND FOR DEMONSTRATION PURPOSES ONLY.
No real personal data, account numbers, or bank APIs are used.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import (
    SYNTHETIC_CITIES, CITY_NAMES, CITY_COORDS, CITY_TO_STATE,
    SYNTHETIC_GANGS, FRAUD_TYPES, CHANNELS, CHANNEL_CASHOUT_TIMES,
    AMOUNT_BUCKETS, HOP_COUNT_RANGE, SYNTHETIC_SEED,
    NUM_SYNTHETIC_COMPLAINTS, SYNTHETIC_START_DATE, SYNTHETIC_END_DATE,
    FEDERATED_BANKS, TEAM_BASE_LOCATIONS
)
from app.db import (
    Base, engine, SessionLocal, Complaint, MuleAccount, ComplaintMuleLink,
    Gang, User, Alert, Intervention, DisplacementPressure, ChannelType, FraudType
)
from app.auth import hash_password


def set_seed(seed: int = SYNTHETIC_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def generate_phone_number() -> str:
    """Generate a synthetic Indian phone number."""
    return f"+91{random.randint(7000000000, 9999999999)}"


def generate_device_id() -> str:
    """Generate a synthetic device ID."""
    return f"DEV-{uuid.uuid4().hex[:12].upper()}"


def generate_account_id(bank: str) -> str:
    """Generate a synthetic account ID."""
    return f"{bank[:3].upper()}{random.randint(1000000000, 9999999999)}"


def sample_city_from_corridor(corridor: List[str], weights: List[float]) -> str:
    """Sample a city from a gang's corridor based on weights."""
    return np.random.choice(corridor, p=weights)


def sample_channel(channel_mix: Dict[str, float]) -> str:
    """Sample a withdrawal channel based on gang's channel mix."""
    channels = list(channel_mix.keys())
    probs = list(channel_mix.values())
    return np.random.choice(channels, p=probs)


def sample_cashout_time(channel: str) -> int:
    """Sample minutes to cashout based on channel distribution (SYNTHETIC)."""
    params = CHANNEL_CASHOUT_TIMES[channel]
    # Truncated normal
    val = np.random.normal(params["mean"], params["std"])
    return int(np.clip(val, params["min"], params["max"]))


def sample_amount() -> float:
    """Sample a fraud amount from realistic distribution (SYNTHETIC)."""
    # Log-normal distribution for amounts
    val = np.random.lognormal(mean=10.5, sigma=1.2)
    return round(np.clip(val, 1000, 50_00_000), 2)


def get_amount_bucket(amount: float) -> str:
    """Get amount bucket label."""
    for low, high, label in AMOUNT_BUCKETS:
        if low <= amount < high:
            return label
    return "xlarge"


def create_mule_accounts_for_gang(gang: Dict, db: Session) -> List[MuleAccount]:
    """Create mule accounts for a gang with shared phones/devices."""
    mules = []
    gang_id = gang["id"]
    
    # Generate shared phone numbers and devices for this gang
    num_shared_phones = max(2, gang["mule_count"] // 3)
    num_shared_devices = max(2, gang["mule_count"] // 3)
    
    shared_phones = [generate_phone_number() for _ in range(num_shared_phones)]
    shared_devices = [generate_device_id() for _ in range(num_shared_devices)]
    
    for i in range(gang["mule_count"]):
        # Most mules use gang-shared phones/devices, some unique
        if random.random() < 0.7:
            phone = random.choice(shared_phones)
        else:
            phone = generate_phone_number()
            
        if random.random() < 0.7:
            device = random.choice(shared_devices)
        else:
            device = generate_device_id()
        
        # Assign city from corridor
        city = sample_city_from_corridor(gang["corridor"], gang["corridor_weights"])
        state = CITY_TO_STATE.get(city, "Unknown")
        bank = random.choice(FEDERATED_BANKS)
        
        mule = MuleAccount(
            account_id=generate_account_id(bank),
            phone_number=phone,
            device_id=device,
            bank=bank,
            city=city,
            state=state,
            gang_id=gang_id,
            is_synthetic=True,
        )
        mules.append(mule)
        db.add(mule)
    
    db.flush()
    return mules


def create_gang_records(db: Session) -> List[Gang]:
    """Create gang records in database."""
    gangs = []
    for gang_data in SYNTHETIC_GANGS:
        gang = Gang(
            gang_id=gang_data["id"],
            name=gang_data["name"],
            corridor=gang_data["corridor"],
            corridor_weights=gang_data["corridor_weights"],
            channel_mix=gang_data["channel_mix"],
            mule_count=gang_data["mule_count"],
            base_city=gang_data["base_city"],
            is_synthetic=True,
        )
        gangs.append(gang)
        db.add(gang)
    db.flush()
    return gangs


def create_complaints_for_gang(
    gang: Dict, 
    mules: List[MuleAccount], 
    db: Session,
    start_idx: int,
    num_complaints: int
) -> List[Complaint]:
    """Create complaints for a specific gang."""
    complaints = []
    gang_id = gang["id"]
    
    for i in range(num_complaints):
        # Random timestamp within range
        time_span = SYNTHETIC_END_DATE - SYNTHETIC_START_DATE
        random_seconds = random.randint(0, int(time_span.total_seconds()))
        timestamp = SYNTHETIC_START_DATE + timedelta(seconds=random_seconds)
        
        # Complaint details
        fraud_type = random.choice(FRAUD_TYPES)
        amount = sample_amount()
        victim_bank = random.choice(FEDERATED_BANKS)
        hop_count = random.randint(*HOP_COUNT_RANGE)
        
        # Select mule chain for this complaint
        chain_length = min(hop_count, len(mules))
        selected_mules = random.sample(mules, chain_length)
        
        # Build mule account chain
        mule_chain = []
        remaining_amount = amount
        for pos, mule in enumerate(selected_mules):
            received = remaining_amount * random.uniform(0.85, 0.98)  # Some loss per hop
            forwarded = received * random.uniform(0.9, 0.99) if pos < chain_length - 1 else 0
            mule_chain.append({
                "account_id": mule.account_id,
                "bank": mule.bank,
                "city": mule.city,
                "hop_position": pos + 1,
                "amount_received": round(received, 2),
                "amount_forwarded": round(forwarded, 2) if pos < chain_length - 1 else 0,
                "phone_number": mule.phone_number,
                "device_id": mule.device_id,
            })
            remaining_amount = forwarded
        
        # Ground truth withdrawal
        withdrawal_city = sample_city_from_corridor(gang["corridor"], gang["corridor_weights"])
        withdrawal_channel = sample_channel(gang["channel_mix"])
        minutes_to_cashout = sample_cashout_time(withdrawal_channel)
        withdrawal_timestamp = timestamp + timedelta(minutes=minutes_to_cashout)
        
        # Victim city (sometimes different from withdrawal city)
        victim_city = random.choice(CITY_NAMES)
        victim_state = CITY_TO_STATE.get(victim_city, "Unknown")
        
        complaint = Complaint(
            complaint_id=f"SYNTH-{gang_id}-{start_idx + i:06d}",
            timestamp=timestamp,
            fraud_type=FraudType(fraud_type),
            amount=amount,
            victim_bank=victim_bank,
            hop_count=hop_count,
            mule_account_chain=mule_chain,
            victim_city=victim_city,
            victim_state=victim_state,
            withdrawal_city=withdrawal_city,
            withdrawal_channel=ChannelType(withdrawal_channel),
            withdrawal_timestamp=withdrawal_timestamp,
            minutes_to_cashout=minutes_to_cashout,
            is_synthetic=True,
        )
        complaints.append(complaint)
        db.add(complaint)
    
    db.flush()
    return complaints


def link_complaints_to_mules(complaints: List[Complaint], db: Session):
    """Create ComplaintMuleLink records for each complaint."""
    for complaint in complaints:
        chain = complaint.mule_account_chain
        for hop_data in chain:
            # Find mule account by account_id
            mule = db.query(MuleAccount).filter(
                MuleAccount.account_id == hop_data["account_id"]
            ).first()
            if mule:
                link = ComplaintMuleLink(
                    complaint_id=complaint.id,
                    mule_account_id=mule.id,
                    hop_position=hop_data["hop_position"],
                    amount_received=hop_data["amount_received"],
                    amount_forwarded=hop_data["amount_forwarded"],
                    timestamp=complaint.timestamp + timedelta(minutes=hop_data["hop_position"] * 5),
                )
                db.add(link)


def update_gang_stats(db: Session):
    """Update gang statistics from complaints."""
    gangs = db.query(Gang).all()
    for gang in gangs:
        # Query via mule links
        from sqlalchemy import func
        total_complaints = db.query(func.count(Complaint.id)).join(
            ComplaintMuleLink, Complaint.id == ComplaintMuleLink.complaint_id
        ).join(
            MuleAccount, ComplaintMuleLink.mule_account_id == MuleAccount.id
        ).filter(MuleAccount.gang_id == gang.gang_id).scalar() or 0
        
        total_amount = db.query(func.sum(Complaint.amount)).join(
            ComplaintMuleLink, Complaint.id == ComplaintMuleLink.complaint_id
        ).join(
            MuleAccount, ComplaintMuleLink.mule_account_id == MuleAccount.id
        ).filter(MuleAccount.gang_id == gang.gang_id).scalar() or 0.0
        
        gang.total_complaints = total_complaints
        gang.total_amount = float(total_amount)
    
    db.commit()


def create_demo_users(db: Session):
    """Create demo users with hashed passwords."""
    from app.config import DEMO_USERS
    for user_data in DEMO_USERS:
        existing = db.query(User).filter(User.username == user_data["username"]).first()
        if not existing:
            user = User(
                username=user_data["username"],
                hashed_password=hash_password(user_data["password"]),
                role=user_data["role"],
                state=user_data.get("state"),
                district=user_data.get("district"),
                bank=user_data.get("bank"),
                is_active=True,
            )
            db.add(user)
    db.commit()


def init_displacement_pressure(db: Session):
    """Initialize displacement pressure table with zeros."""
    for city in CITY_NAMES:
        for channel in CHANNELS:
            for gang in SYNTHETIC_GANGS:
                pressure = DisplacementPressure(
                    city=city,
                    channel=ChannelType(channel),
                    gang_id=gang["id"],
                    pressure_score=0.0,
                )
                db.add(pressure)
    db.commit()


def generate_all_synthetic_data() -> Dict:
    """Main function to generate all synthetic data."""
    print("=" * 60)
    print("GENERATING SYNTHETIC DATA (ALL DATA IS SYNTHETIC)")
    print("=" * 60)
    
    set_seed(SYNTHETIC_SEED)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    try:
        # Clear existing synthetic data
        print("Clearing existing synthetic data...")
        db.query(ComplaintMuleLink).delete()
        db.query(Complaint).delete()
        db.query(MuleAccount).delete()
        db.query(Gang).delete()
        db.query(Alert).delete()
        db.query(Intervention).delete()
        db.query(DisplacementPressure).delete()
        db.query(User).delete()
        db.commit()
        
        # Create gangs
        print("Creating gangs...")
        gangs = create_gang_records(db)
        
        # Create mules for each gang
        print("Creating mule accounts...")
        all_mules = {}
        for gang in SYNTHETIC_GANGS:
            mules = create_mule_accounts_for_gang(gang, db)
            all_mules[gang["id"]] = mules
        
        # Distribute complaints across gangs
        print(f"Creating {NUM_SYNTHETIC_COMPLAINTS} complaints...")
        complaints_per_gang = NUM_SYNTHETIC_COMPLAINTS // len(SYNTHETIC_GANGS)
        remainder = NUM_SYNTHETIC_COMPLAINTS % len(SYNTHETIC_GANGS)
        
        all_complaints = []
        start_idx = 0
        for i, gang in enumerate(SYNTHETIC_GANGS):
            n = complaints_per_gang + (1 if i < remainder else 0)
            complaints = create_complaints_for_gang(
                gang, all_mules[gang["id"]], db, start_idx, n
            )
            all_complaints.extend(complaints)
            start_idx += n
        
        # Link complaints to mules
        print("Linking complaints to mule accounts...")
        link_complaints_to_mules(all_complaints, db)
        
        # Update gang stats
        print("Updating gang statistics...")
        update_gang_stats(db)
        
        # Create demo users
        print("Creating demo users...")
        create_demo_users(db)
        
        # Initialize displacement pressure
        print("Initializing displacement pressure...")
        init_displacement_pressure(db)
        
        db.commit()
        
        # Print summary
        total_complaints = db.query(Complaint).count()
        total_mules = db.query(MuleAccount).count()
        total_gangs = db.query(Gang).count()
        
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA GENERATION COMPLETE")
        print("=" * 60)
        print(f"Gangs: {total_gangs}")
        print(f"Mule Accounts: {total_mules}")
        print(f"Complaints: {total_complaints}")
        print(f"Date Range: {SYNTHETIC_START_DATE.date()} to {SYNTHETIC_END_DATE.date()}")
        print("=" * 60)
        
        return {
            "gangs": total_gangs,
            "mules": total_mules,
            "complaints": total_complaints,
        }
        
    finally:
        db.close()


def get_complaints_dataframe(db: Session) -> pd.DataFrame:
    """Get complaints as DataFrame for ML training."""
    complaints = db.query(Complaint).filter(Complaint.is_synthetic == True).all()
    
    rows = []
    for c in complaints:
        # Extract features from mule chain
        first_mule_city = None
        first_mule_bank = None
        if c.mule_account_chain:
            first_mule_city = c.mule_account_chain[0].get("city")
            first_mule_bank = c.mule_account_chain[0].get("bank")
        
        rows.append({
            "complaint_id": c.complaint_id,
            "timestamp": c.timestamp,
            "fraud_type": c.fraud_type.value,
            "amount": c.amount,
            "amount_bucket": get_amount_bucket(c.amount),
            "victim_bank": c.victim_bank,
            "hop_count": c.hop_count,
            "victim_city": c.victim_city,
            "victim_state": c.victim_state,
            "first_mule_city": first_mule_city,
            "first_mule_bank": first_mule_bank,
            "withdrawal_city": c.withdrawal_city,
            "withdrawal_channel": c.withdrawal_channel.value if c.withdrawal_channel else None,
            "minutes_to_cashout": c.minutes_to_cashout,
            "hour_of_day": c.timestamp.hour,
            "day_of_week": c.timestamp.weekday(),
            "is_weekend": c.timestamp.weekday() >= 5,
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    generate_all_synthetic_data()
"""
Configuration and constants for the Cybercrime Withdrawal Predictor.

ALL DATA IN THIS PROJECT IS SYNTHETIC AND FOR DEMONSTRATION PURPOSES ONLY.
No real personal data, account numbers, or bank APIs are used.
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables with demo defaults."""
    
    # Database
    database_url: str = "sqlite:///./data/app.db"
    
    # JWT Auth
    jwt_secret: str = "demo-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    
    # Encryption
    embedding_encryption_key: str | None = None  # base64 32-byte key; if None, derived from JWT secret
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]
    
    # Model
    model_path: str = "./data/model.joblib"
    model_retrain_interval_hours: int = 24
    
    # Federated Learning
    federated_round_interval_minutes: int = 60
    min_clients_for_aggregation: int = 2
    
    # Optimizer
    avg_team_speed_kmph: float = 40.0
    atm_intervention_radius_km: float = 5.0
    aePS_intervention_radius_km: float = 3.0
    
    model_config = {"protected_namespaces": ("settings_",)}


settings = Settings()

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent

# =============================================================================
# LOAD STATE WEIGHTS FROM YAML CONFIG
# =============================================================================

def _load_state_weights() -> Dict:
    """Load state weights from config YAML."""
    config_path = BASE_DIR / "config" / "state_weights.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

STATE_WEIGHTS = _load_state_weights()

# State weights for synthetic complaint distribution
STATE_WEIGHTS_MAP = STATE_WEIGHTS.get("states", {})

# Channel mix by state (optional override)
CHANNEL_MIX_BY_STATE = STATE_WEIGHTS.get("channel_mix_by_state", {})

# Fraud type distribution by state (optional)
FRAUD_TYPE_BY_STATE = STATE_WEIGHTS.get("fraud_type_by_state", {})

# City weights within state
CITY_WEIGHTS_BY_STATE = STATE_WEIGHTS.get("city_weights", {})

# =============================================================================
# SYNTHETIC DATA CONSTANTS - ALL LABELED AS SYNTHETIC
# =============================================================================

# 14 Indian cities/hubs with lat/lon (SYNTHETIC DATA)
SYNTHETIC_CITIES: List[Dict] = [
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    {"name": "Mewat", "lat": 28.1024, "lon": 77.0143, "state": "Haryana"},
    {"name": "Jamtara", "lat": 23.9567, "lon": 86.8007, "state": "Jharkhand"},
    {"name": "Bharatpur", "lat": 27.2152, "lon": 77.4907, "state": "Rajasthan"},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
]

CITY_NAMES = [c["name"] for c in SYNTHETIC_CITIES]
CITY_COORDS = {c["name"]: (c["lat"], c["lon"]) for c in SYNTHETIC_CITIES}
CITY_TO_STATE = {c["name"]: c["state"] for c in SYNTHETIC_CITIES}

# 6 Gangs with corridors, channel mix, and mule counts (SYNTHETIC DATA)
SYNTHETIC_GANGS: List[Dict] = [
    {
        "id": "gang_001",
        "name": "Delhi-Mewat Corridor",
        "corridor": ["Delhi", "Mewat", "Gurugram"],
        "corridor_weights": [0.5, 0.3, 0.2],
        "channel_mix": {"ATM": 0.6, "AePS": 0.3, "Crypto": 0.1},
        "mule_count": 8,
        "base_city": "Delhi",
    },
    {
        "id": "gang_002",
        "name": "Mumbai-Pune Expressway",
        "corridor": ["Mumbai", "Pune", "Nashik"],
        "corridor_weights": [0.4, 0.4, 0.2],
        "channel_mix": {"ATM": 0.4, "AePS": 0.4, "Crypto": 0.2},
        "mule_count": 7,
        "base_city": "Mumbai",
    },
    {
        "id": "gang_003",
        "name": "Jamtara-Bengaluru Nexus",
        "corridor": ["Jamtara", "Kolkata", "Bengaluru"],
        "corridor_weights": [0.3, 0.3, 0.4],
        "channel_mix": {"ATM": 0.2, "AePS": 0.3, "Crypto": 0.5},
        "mule_count": 9,
        "base_city": "Jamtara",
    },
    {
        "id": "gang_004",
        "name": "Rajasthan Triangle",
        "corridor": ["Jaipur", "Bharatpur", "Alwar"],
        "corridor_weights": [0.5, 0.3, 0.2],
        "channel_mix": {"ATM": 0.7, "AePS": 0.2, "Crypto": 0.1},
        "mule_count": 6,
        "base_city": "Jaipur",
    },
    {
        "id": "gang_005",
        "name": "South India Circuit",
        "corridor": ["Chennai", "Hyderabad", "Bengaluru"],
        "corridor_weights": [0.3, 0.3, 0.4],
        "channel_mix": {"ATM": 0.3, "AePS": 0.4, "Crypto": 0.3},
        "mule_count": 8,
        "base_city": "Chennai",
    },
    {
        "id": "gang_006",
        "name": "Gangetic Belt",
        "corridor": ["Lucknow", "Patna", "Varanasi"],
        "corridor_weights": [0.4, 0.4, 0.2],
        "channel_mix": {"ATM": 0.5, "AePS": 0.4, "Crypto": 0.1},
        "mule_count": 7,
        "base_city": "Lucknow",
    },
]

# Fraud types (SYNTHETIC DATA)
FRAUD_TYPES = [
    "UPI Fraud",
    "Investment Scam",
    "Digital Arrest",
    "Loan App Harassment",
    "Phishing",
    "QR Code Scam",
]

# Channels
CHANNELS = ["ATM", "AePS", "Crypto"]

# Channel-specific cashout time distributions (minutes after complaint) - SYNTHETIC
CHANNEL_CASHOUT_TIMES = {
    "ATM": {"mean": 45, "std": 20, "min": 10, "max": 180},
    "AePS": {"mean": 120, "std": 60, "min": 30, "max": 480},
    "Crypto": {"mean": 15, "std": 10, "min": 2, "max": 60},
}

# Amount buckets (INR)
AMOUNT_BUCKETS = [
    (0, 10_000, "micro"),
    (10_000, 50_000, "small"),
    (50_000, 2_00_000, "medium"),
    (2_00_000, 10_00_000, "large"),
    (10_00_000, float("inf"), "xlarge"),
]

# Hop count range
HOP_COUNT_RANGE = (1, 6)

# Random seed for reproducibility
SYNTHETIC_SEED = 42

# Number of synthetic complaints to generate
NUM_SYNTHETIC_COMPLAINTS = 5000

# Date range for synthetic data (last 90 days)
from datetime import datetime, timedelta
SYNTHETIC_END_DATE = datetime(2026, 9, 30, 23, 59, 59)
SYNTHETIC_START_DATE = SYNTHETIC_END_DATE - timedelta(days=90)

# Demo users (passwords are hashed at runtime)
DEMO_USERS = [
    {"username": "i4c_admin", "password": "admin123", "role": "i4c_admin", "state": None, "district": None},
    {"username": "state_mh", "password": "state123", "role": "state_officer", "state": "Maharashtra", "district": None},
    {"username": "state_dl", "password": "state123", "role": "state_officer", "state": "Delhi", "district": None},
    {"username": "state_ka", "password": "state123", "role": "state_officer", "state": "Karnataka", "district": None},
    {"username": "district_delhi_central", "password": "dist123", "role": "district_officer", "state": "Delhi", "district": "Central Delhi"},
    {"username": "district_mumbai", "password": "dist123", "role": "district_officer", "state": "Maharashtra", "district": "Mumbai"},
    {"username": "bank_hdfc", "password": "bank123", "role": "bank_nodal", "state": None, "district": None, "bank": "HDFC"},
    {"username": "bank_sbi", "password": "bank123", "role": "bank_nodal", "state": None, "district": None, "bank": "SBI"},
    {"username": "bank_icici", "password": "bank123", "role": "bank_nodal", "state": None, "district": None, "bank": "ICICI"},
]

# Alert severity thresholds
ALERT_SEVERITY_THRESHOLDS = {
    "HIGH": 0.7,
    "MEDIUM": 0.4,
    "LOW": 0.0,
}

# Alert templates (EN/HI)
ALERT_TEMPLATES = {
    "en": {
        "HIGH": "🚨 HIGH PRIORITY: Predicted cash-out at {city} via {channel} in ~{eta_min} min. Amount: ₹{amount:,.0f}. Gang: {gang_id}. Deploy team immediately.",
        "MEDIUM": "⚠️ MEDIUM: Likely withdrawal at {city} via {channel} within {eta_min} min. Amount: ₹{amount:,.0f}. Monitor {gang_id} corridor.",
        "LOW": "ℹ️ LOW: Possible cash-out at {city} via {channel}. Amount: ₹{amount:,.0f}. Track for pattern.",
    },
    "hi": {
        "HIGH": "🚨 उच्च प्राथमिकता: {city} में {channel} के माध्यम से नकदी निकासी की भविष्यवाणी ~{eta_min} मिनट में। राशि: ₹{amount:,.0f}। गिरोह: {gang_id}। तुरंत टीम तैनात करें।",
        "MEDIUM": "⚠️ मध्यम: {city} में {channel} के माध्यम से {eta_min} मिनट के भीतर निकासी संभावित। राशि: ₹{amount:,.0f}। {gang_id} कॉरिडोर की निगरानी करें।",
        "LOW": "ℹ️ कम: {city} में {channel} के माध्यम से संभावित निकासी। राशि: ₹{amount:,.0f}। पैटर्न के लिए ट्रैक करें।",
    },
}

# Team base locations (SYNTHETIC)
TEAM_BASE_LOCATIONS = [
    {"id": "team_001", "name": "Delhi Cyber Cell", "city": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"id": "team_002", "name": "Mumbai Cyber Cell", "city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"id": "team_003", "name": "Bengaluru Cyber Cell", "city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    {"id": "team_004", "name": "Kolkata Cyber Cell", "city": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"id": "team_005", "name": "Chennai Cyber Cell", "city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"id": "team_006", "name": "Hyderabad Cyber Cell", "city": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"id": "team_007", "name": "Jaipur Cyber Cell", "city": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"id": "team_008", "name": "Lucknow Cyber Cell", "city": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"id": "team_009", "name": "Pune Cyber Cell", "city": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"id": "team_010", "name": "Ahmedabad Cyber Cell", "city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
]

# Simulated banks for federated learning
FEDERATED_BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "PNB"]

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
METRICS_FILE = BASE_DIR / "metrics.json"
AUDIT_LOG_FILE = DATA_DIR / "audit.log"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
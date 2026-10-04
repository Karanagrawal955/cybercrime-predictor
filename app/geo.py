"""
Geospatial utilities for mapping real ATM/branch locations to H3 cells or pincode grids.
Loads real ATM/branch coordinates from CSV and maps them to candidate cash-out locations.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import h3

from app.config import CITY_COORDS, CITY_NAMES


@dataclass
class ATMLocation:
    """Represents an ATM/branch location from CSV."""
    lat: float
    lon: float
    name: str
    type: str  # "atm" or "bank"
    bank: str
    city: str
    state: str
    postcode: str
    h3_cell: str  # H3 cell at resolution 6
    pincode: Optional[str] = None


# H3 resolution for cash-out location grid (res 6 ≈ 1.2 km edge length)
H3_RESOLUTION = 6

# Path to sample ATM data (bundled for offline testing)
SAMPLE_ATM_CSV = Path(__file__).parent.parent / "data" / "real" / "sample_atms.csv"

# In-memory caches
_atm_locations: Optional[List[ATMLocation]] = None
_h3_to_atms: Optional[Dict[str, List[ATMLocation]]] = None
_city_to_h3: Optional[Dict[str, List[str]]] = None


def load_atm_locations(csv_path: Optional[Path] = None) -> List[ATMLocation]:
    """Load ATM/branch locations from CSV file."""
    global _atm_locations
    
    if _atm_locations is not None:
        return _atm_locations
    
    path = csv_path or SAMPLE_ATM_CSV
    if not path.exists():
        raise FileNotFoundError(f"ATM CSV not found at {path}")
    
    locations = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                h3_cell = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
                
                loc = ATMLocation(
                    lat=lat,
                    lon=lon,
                    name=row.get("name", ""),
                    type=row.get("type", "atm"),
                    bank=row.get("bank", ""),
                    city=row.get("addr_city", ""),
                    state=row.get("addr_state", ""),
                    postcode=row.get("addr_postcode", ""),
                    h3_cell=h3_cell,
                    pincode=row.get("addr_postcode", "") or None
                )
                locations.append(loc)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row {row}: {e}")
                continue
    
    _atm_locations = locations
    return locations


def get_atm_locations() -> List[ATMLocation]:
    """Get cached ATM locations, loading if necessary."""
    if _atm_locations is None:
        load_atm_locations()
    return _atm_locations


def build_h3_index() -> Dict[str, List[ATMLocation]]:
    """Build mapping from H3 cell to list of ATMs in that cell."""
    global _h3_to_atms
    
    if _h3_to_atms is not None:
        return _h3_to_atms
    
    atms = get_atm_locations()
    index = {}
    for atm in atms:
        index.setdefault(atm.h3_cell, []).append(atm)
    _h3_to_atms = index
    return index


def build_city_to_h3() -> Dict[str, List[str]]:
    """Build mapping from city name to list of H3 cells covering that city."""
    global _city_to_h3
    
    if _city_to_h3 is not None:
        return _city_to_h3
    
    atms = get_atm_locations()
    index = {}
    for atm in atms:
        if atm.city:
            index.setdefault(atm.city, set()).add(atm.h3_cell)
    
    # Convert sets to lists
    _city_to_h3 = {city: list(cells) for city, cells in index.items()}
    return _city_to_h3


def get_atms_in_h3(h3_cell: str) -> List[ATMLocation]:
    """Get all ATMs in a specific H3 cell."""
    index = build_h3_index()
    return index.get(h3_cell, [])


def get_atms_in_city(city: str) -> List[ATMLocation]:
    """Get all ATMs in a city."""
    atms = get_atm_locations()
    return [atm for atm in atms if atm.city.lower() == city.lower()]


def get_h3_cells_for_city(city: str) -> List[str]:
    """Get H3 cells covering a city."""
    index = build_city_to_h3()
    return index.get(city, [])


def get_candidate_cashout_locations(
    city: Optional[str] = None,
    state: Optional[str] = None,
    h3_cell: Optional[str] = None,
    bank: Optional[str] = None
) -> List[Dict]:
    """
    Get candidate cash-out locations for prediction.
    Can filter by city, state, H3 cell, or bank.
    Returns list of dicts with location info suitable for prediction.
    """
    atms = get_atm_locations()
    
    filtered = atms
    if city:
        filtered = [a for a in filtered if a.city.lower() == city.lower()]
    if state:
        filtered = [a for a in filtered if a.state.lower() == state.lower()]
    if h3_cell:
        filtered = [a for a in filtered if a.h3_cell == h3_cell]
    if bank:
        filtered = [a for a in filtered if bank.lower() in a.bank.lower()]
    
    # Deduplicate by H3 cell (keep first ATM in each cell)
    seen_cells = set()
    unique_atms = []
    for atm in filtered:
        if atm.h3_cell not in seen_cells:
            seen_cells.add(atm.h3_cell)
            unique_atms.append(atm)
    
    # Convert to prediction-ready format
    candidates = []
    for atm in unique_atms:
        # Get center of H3 cell for prediction coordinates
        cell_center = h3.cell_to_latlng(atm.h3_cell)
        
        candidates.append({
            "city": atm.city or "Unknown",
            "state": atm.state or "Unknown",
            "lat": cell_center[0],
            "lon": cell_center[1],
            "h3_cell": atm.h3_cell,
            "type": atm.type,
            "bank": atm.bank,
            "name": atm.name,
            "postcode": atm.postcode,
        })
    
    return candidates


def get_all_cashout_cells() -> List[str]:
    """Get all H3 cells that have at least one ATM."""
    index = build_h3_index()
    return list(index.keys())


def get_city_from_h3(h3_cell: str) -> Optional[str]:
    """Get the most common city for an H3 cell."""
    atms = get_atms_in_h3(h3_cell)
    if not atms:
        return None
    # Return most common city
    from collections import Counter
    cities = [a.city for a in atms if a.city]
    if not cities:
        return None
    return Counter(cities).most_common(1)[0][0]


def get_state_from_h3(h3_cell: str) -> Optional[str]:
    """Get the most common state for an H3 cell."""
    atms = get_atms_in_h3(h3_cell)
    if not atms:
        return None
    from collections import Counter
    states = [a.state for a in atms if a.state]
    if not states:
        return None
    return Counter(states).most_common(1)[0][0]


def get_nearest_cashout_cells(
    lat: float,
    lon: float,
    k: int = 10,
    max_distance_km: float = 50.0
) -> List[Tuple[str, float]]:
    """
    Find k nearest H3 cells with ATMs to a given coordinate.
    Returns list of (h3_cell, distance_km) sorted by distance.
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    
    cells = get_all_cashout_cells()
    results = []
    for cell in cells:
        cell_lat, cell_lon = h3.cell_to_latlng(cell)
        dist = haversine(lat, lon, cell_lat, cell_lon)
        if dist <= max_distance_km:
            results.append((cell, dist))
    
    results.sort(key=lambda x: x[1])
    return results[:k]


# Import Counter for city/state counting
from collections import Counter


# ============================================================
# Pincode grid support (alternative to H3)
# ============================================================

# Simple pincode to city/state mapping (extendable)
PINCODE_MAP = {
    "400001": {"city": "Mumbai", "state": "Maharashtra"},
    "400005": {"city": "Mumbai", "state": "Maharashtra"},
    "400021": {"city": "Mumbai", "state": "Maharashtra"},
    "400050": {"city": "Mumbai", "state": "Maharashtra"},
    "400058": {"city": "Mumbai", "state": "Maharashtra"},
    "110001": {"city": "Delhi", "state": "Delhi"},
    "110005": {"city": "Delhi", "state": "Delhi"},
    "110085": {"city": "Delhi", "state": "Delhi"},
    "110075": {"city": "Delhi", "state": "Delhi"},
    "560001": {"city": "Bengaluru", "state": "Karnataka"},
    "560034": {"city": "Bengaluru", "state": "Karnataka"},
    "560066": {"city": "Bengaluru", "state": "Karnataka"},
    "560100": {"city": "Bengaluru", "state": "Karnataka"},
    "600002": {"city": "Chennai", "state": "Tamil Nadu"},
    "600017": {"city": "Chennai", "state": "Tamil Nadu"},
    "600042": {"city": "Chennai", "state": "Tamil Nadu"},
    "600045": {"city": "Chennai", "state": "Tamil Nadu"},
    "500001": {"city": "Hyderabad", "state": "Telangana"},
    "500034": {"city": "Hyderabad", "state": "Telangana"},
    "500032": {"city": "Hyderabad", "state": "Telangana"},
}


def get_city_state_from_pincode(pincode: str) -> Optional[Dict]:
    """Get city/state from pincode (first 3 digits for broader area)."""
    # Try exact match
    if pincode in PINCODE_MAP:
        return PINCODE_MAP[pincode]
    # Try 3-digit prefix
    prefix = pincode[:3]
    for pc, info in PINCODE_MAP.items():
        if pc.startswith(prefix):
            return info
    return None


def get_atms_by_pincode(pincode: str) -> List[ATMLocation]:
    """Get ATMs matching a pincode (exact or prefix)."""
    atms = get_atm_locations()
    pincode_prefix = pincode[:3] if len(pincode) >= 3 else pincode
    return [a for a in atms if a.pincode and a.pincode.startswith(pincode_prefix)]
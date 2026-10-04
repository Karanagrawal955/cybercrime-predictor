#!/usr/bin/env python3
"""
Fetch ATM/branch locations from OpenStreetMap via Overpass API.
Outputs CSV with columns: lat, lon, name, type, bank, addr_city, addr_state, addr_postcode

Usage:
    python scripts/fetch_osm_atms.py --state Maharashtra --output data/real/atms_maharashtra.csv
    python scripts/fetch_osm_atms.py --bbox 18.9,72.8,19.3,73.1 --output data/atms_mumbai.csv

For offline testing, a sample CSV is provided at data/real/sample_atms.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass query template for ATMs and bank branches
OVERPASS_QUERY_TEMPLATE = """
[out:json][timeout:180];
(
  node["amenity"="atm"]({bbox});
  node["amenity"="bank"]({bbox});
  way["amenity"="atm"]({bbox});
  way["amenity"="bank"]({bbox});
  relation["amenity"="atm"]({bbox});
  relation["amenity"="bank"]({bbox});
);
out center tags;
"""

# Indian state bounding boxes (approx)
STATE_BBOXES = {
    "Maharashtra": "15.6,72.6,22.0,80.9",
    "Delhi": "28.4,76.8,28.9,77.3",
    "Karnataka": "11.5,74.0,18.5,78.5",
    "Tamil Nadu": "8.0,76.2,13.5,80.3",
    "Telangana": "15.8,77.0,19.9,81.5",
    "Uttar Pradesh": "23.8,77.0,30.4,84.6",
    "West Bengal": "21.5,85.8,27.3,89.9",
    "Gujarat": "20.1,68.1,24.7,74.5",
    "Rajasthan": "23.0,69.5,30.2,78.3",
    "Haryana": "27.6,74.3,30.9,77.6",
    "Bihar": "24.3,83.3,27.5,88.3",
    "Jharkhand": "21.9,83.5,25.4,87.8",
}

# Major Indian banks to filter
MAJOR_BANKS = {
    "State Bank of India", "SBI", "HDFC Bank", "HDFC", "ICICI Bank", "ICICI",
    "Punjab National Bank", "PNB", "Bank of Baroda", "BOB", "Canara Bank",
    "Union Bank of India", "Axis Bank", "Axis", "Kotak Mahindra Bank", "Kotak",
    "IndusInd Bank", "Yes Bank", "IDFC First Bank", "Federal Bank"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch ATM/branch locations from Overpass")
    parser.add_argument("--state", help="Indian state name (e.g., Maharashtra)")
    parser.add_argument("--bbox", help="Bounding box: min_lat,min_lon,max_lat,max_lon")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Seconds between requests")
    return parser.parse_args()


def get_bbox(state: Optional[str], bbox: Optional[str]) -> str:
    if bbox:
        return bbox
    if state and state in STATE_BBOXES:
        return STATE_BBOXES[state]
    raise ValueError("Either --state or --bbox must be provided")


def fetch_overpass(bbox: str, rate_limit: float = 1.0) -> List[Dict]:
    query = OVERPASS_QUERY_TEMPLATE.format(bbox=bbox)
    print(f"Querying Overpass API for bbox: {bbox}")
    
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
    response.raise_for_status()
    data = response.json()
    
    # Rate limiting
    time.sleep(rate_limit)
    
    return data.get("elements", [])


def extract_location(elem: Dict) -> Optional[tuple]:
    """Extract (lat, lon) from element (node, way, or relation)."""
    if elem["type"] == "node":
        return elem.get("lat"), elem.get("lon")
    elif "center" in elem:
        return elem["center"].get("lat"), elem["center"].get("lon")
    return None


def extract_tags(elem: Dict) -> Dict:
    return elem.get("tags", {})


def normalize_bank_name(tags: Dict) -> Optional[str]:
    """Extract and normalize bank name from tags."""
    for key in ["operator", "brand", "name"]:
        val = tags.get(key, "").strip()
        if val:
            # Check against known banks
            for bank in MAJOR_BANKS:
                if bank.lower() in val.lower():
                    return bank
            return val
    return None


def process_elements(elements: List[Dict]) -> List[Dict]:
    """Convert Overpass elements to standardized records."""
    records = []
    for elem in elements:
        loc = extract_location(elem)
        if not loc:
            continue
        lat, lon = loc
        tags = extract_tags(elem)
        
        # Skip if not ATM or bank
        amenity = tags.get("amenity")
        if amenity not in ("atm", "bank"):
            continue
            
        record = {
            "lat": lat,
            "lon": lon,
            "name": tags.get("name", ""),
            "type": amenity,
            "bank": normalize_bank_name(tags) or "",
            "addr_city": tags.get("addr:city", ""),
            "addr_state": tags.get("addr:state", ""),
            "addr_postcode": tags.get("addr:postcode", ""),
            "operator": tags.get("operator", ""),
            "network": tags.get("network", ""),
            "currency": tags.get("currency", ""),
            "opening_hours": tags.get("opening_hours", ""),
            "wheelchair": tags.get("wheelchair", ""),
        }
        records.append(record)
    return records


def write_csv(records: List[Dict], output_path: Path):
    if not records:
        print("No records to write")
        return
    
    fieldnames = [
        "lat", "lon", "name", "type", "bank",
        "addr_city", "addr_state", "addr_postcode",
        "operator", "network", "currency", "opening_hours", "wheelchair"
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {output_path}")


def main():
    args = parse_args()
    try:
        bbox = get_bbox(args.state, args.bbox)
        elements = fetch_overpass(bbox, args.rate_limit)
        records = process_elements(elements)
        write_csv(records, Path(args.output))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
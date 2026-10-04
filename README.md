# Cybercrime Withdrawal Predictor

**Predictive analytics framework for cybercrime complaints to forecast likely cash withdrawal locations in advance, enabling proactive intervention.**

Ministry of Home Affairs (I4C) - Problem Statement 26184

> **ALL DATA IN THIS PROJECT IS SYNTHETIC AND FOR DEMONSTRATION PURPOSES ONLY.**
> No real personal data, account numbers, or bank APIs are used.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (withdrawal-predictor-v2.html)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FASTAPI BACKEND (app/main.py)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │   /auth      │ │ /complaints  │ │  /heatmap    │ │    /gangs        │  │
│  │   /predict   │ │  /optimize   │ │ /outcomes    │ │  /federated      │  │
│  │  /alerts     │ │  /audit      │ │  /metrics    │ │   /dev/seed      │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
            │  SQLite DB    │ │  ML Models    │ │  Audit Log    │
            │  (app.db)     │ │  (app/model)  │ │  (tamper-proof)│
            └───────────────┘ └───────────────┘ └───────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
            │ Synthetic Data│ │  Graph-based  │ │  Federated    │
            │  Generator    │ │  Gang Detect  │ │  Learning     │
            │ (app/synth)   │ │ (app/graph)   │ │ (app/federated)│
            └───────────────┘ └───────────────┘ └───────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. Open API docs
open http://localhost:8000/docs

# 4. Run tests
pytest tests/ -v

# 5. Run end-to-end demo
python -m app.demo
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | JWT login with roles |
| POST | `/complaints` | Ingest complaint → prediction + plan + alerts |
| GET | `/predictions/{id}` | Get predictions for complaint |
| GET | `/heatmap` | City risk scores (filter: channel, fraud_type, hours) |
| GET | `/gangs` | List all gangs with stats |
| GET | `/gangs/{id}` | Gang detail with mules |
| POST | `/optimize` | Get intervention plan for predictions |
| POST | `/outcomes` | Record intervention result (triggers displacement) |
| POST | `/federated/round` | Run federated learning round (admin) |
| GET | `/federated/status` | FL status & privacy guarantee |
| GET | `/alerts` | Get alerts (lang=en\|hi, filters) |
| GET | `/audit/verify` | Verify tamper-evident audit chain |
| GET | `/audit/log` | Get audit log entries |
| GET | `/metrics` | Model evaluation metrics |
| POST | `/dev/seed` | Regenerate synthetic data & retrain |

## Example API Calls

```bash
# Login as admin
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=i4c_admin&password=admin123"

# Ingest a complaint (returns prediction + intervention plan + alerts)
curl -X POST http://localhost:8000/complaints \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_id": "TEST-001",
    "timestamp": "2026-09-15T14:30:00",
    "fraud_type": "UPI Fraud",
    "amount": 250000,
    "victim_bank": "HDFC",
    "hop_count": 3,
    "mule_account_chain": [
      {"account_id": "HDFC1234567890", "bank": "HDFC", "city": "Mumbai", "hop_position": 1, "amount_received": 250000, "amount_forwarded": 240000},
      {"account_id": "SBI0987654321", "bank": "SBI", "city": "Pune", "hop_position": 2, "amount_received": 240000, "amount_forwarded": 230000},
      {"account_id": "ICICI1111111111", "bank": "ICICI", "city": "Delhi", "hop_position": 3, "amount_received": 230000, "amount_forwarded": 0}
    ],
    "victim_city": "Mumbai"
  }'

# Get heatmap
curl -X GET "http://localhost:8000/heatmap?channel=ATM&hours=24" \
  -H "Authorization: Bearer <token>"

# Get alerts in Hindi
curl -X GET "http://localhost:8000/alerts?lang=hi&severity=HIGH" \
  -H "Authorization: Bearer <token>"

# Record intervention outcome
curl -X POST http://localhost:8000/outcomes \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"complaint_id": "TEST-001", "intervention_id": 1, "intercepted": true, "amount_recovered": 180000}'
```

## Model Metrics (Synthetic Data)

| Metric | Value | Note |
|--------|-------|------|
| Top-1 Accuracy | ~0.15-0.25 | 42 classes (14 cities × 3 channels) |
| Top-3 Accuracy | ~0.35-0.50 | |
| Precision@3 | ~0.30-0.45 | |
| Mean Lead Time | ~15-30 min | When prediction correct |
| MAE (minutes) | ~45-80 | |
| RMSE (minutes) | ~60-100 | |

> **IMPORTANT**: These metrics are from SYNTHETIC DATA ONLY and are NOT representative of real-world performance. They are reported honestly for transparency.

## Key Features

### 1. Synthetic Data Generator (`app/synth.py`)
- 14 Indian cities/hubs with lat/lon
- 6 gangs with corridors, channel mix, 8 mules each
- 5000+ complaints with ground truth withdrawals
- Mules share phones/devices within gangs (with cross-gang noise)
- Seeded and reproducible (seed=42)

### 2. Gang Detection (`app/graph.py`)
- Graph linking accounts ↔ phones ↔ devices
- Connected components = gangs
- Outputs: complaints, mules, total rupees, top corridor, channel mix

### 3. Prediction Model (`app/model.py`)
- Features: gang corridor history, mule past cities, channel history, fraud type, hop count, time of day, amount bucket, minutes since complaint
- Multiclass (city+channel) + regressor (minutes-to-cashout)
- **Time-based train/test split** (not random)
- Calibrated probabilities (isotonic)
- Per-prediction explanations

### 4. Intervention Optimizer (`app/optimizer.py`)
- Race clock: travel time vs predicted ETA
- Greedy assignment maximizing expected recovered rupees
- ATM/AePS need team; Crypto = exchange freeze (no team)
- Never assigns one team twice

### 5. Displacement + Feedback (`app/displacement.py`)
- Interception → increases policing pressure at location
- Shifts probability mass to nearest unpoliced hub
- Miss → updates gang corridor weights
- Simple, documented, auditable

### 5. Federated Learning (`app/federated.py`)
- 3 simulated banks train locally
- Share ONLY aggregated feature importances (FedAvg-style)
- **0 raw rows ever shared** - logged and verified
- Server averages and returns global prior

### 7. Alerts + Audit (`app/alerts.py`)
- Severity: HIGH/MEDIUM/LOW by probability
- Templates in English & Hindi
- Simulated SMS/email/webhook dispatch
- Tamper-evident audit log: SHA-256(prev_hash + timestamp + payload)

### 8. Auth & Roles (`app/auth.py`)
- JWT with roles: `i4c_admin`, `state_officer`, `district_officer`, `bank_nodal`
- Role scoping enforced on all data endpoints
- Demo users seeded

### 9. Bilingual Alerts
- English and Hindi templates
- Selected via `lang=en|hi` parameter

## Demo Users

| Username | Password | Role | Scope |
|----------|----------|------|-------|
| i4c_admin | admin123 | i4c_admin | All India |
| state_mh | state123 | state_officer | Maharashtra |
| state_dl | state123 | state_officer | Delhi |
| state_ka | state123 | state_officer | Karnataka |
| district_delhi_central | dist123 | district_officer | Central Delhi |
| district_mumbai | dist123 | district_officer | Mumbai |
| bank_hdfc | bank123 | bank_nodal | HDFC complaints |
| bank_sbi | bank123 | bank_nodal | SBI complaints |
| bank_icici | bank123 | bank_nodal | ICICI complaints |

## Running the Demo

```bash
python -m app.demo
```

This will:
1. Seed synthetic data
2. Train the model
3. Simulate 5 complaints end-to-end
4. Print intervention plans
5. Verify audit chain integrity

## What Changes with Real CFCFRMS Data

| Component | Synthetic | Real CFCFRMS |
|-----------|-----------|--------------|
| Data Source | Generated | NCRB/CFCFRMS API |
| Mule Accounts | Simulated | Real account graph |
| Gang Labels | Planted | Investigated clusters |
| Cashout Times | Parametric | Historical distributions |
| Team Locations | Fixed 10 cities | Dynamic from police API |
| Exchange Freeze | Simulated | NPCI/CCIL integration |
| Federated | 3 banks | All participating banks |

## Limitations

1. **Synthetic data only** - metrics not representative of real performance
2. **No real-time data feeds** - batch processing simulation
3. **Simplified travel time** - haversine distance, no traffic/road network
4. **No actual SMS/email** - mock dispatch only
5. **Federated simulation** - not actual secure multi-party computation
6. **Single model version** - no A/B testing or model registry

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI routes
│   ├── config.py         # Settings, constants (ALL SYNTHETIC)
│   ├── auth.py           # JWT, RBAC
│   ├── db.py             # SQLAlchemy models
│   ├── synth.py          # Synthetic data generator
│   ├── graph.py          # Gang detection
│   ├── model.py          # ML prediction
│   ├── optimizer.py      # Intervention optimization
│   ├── displacement.py   # Feedback loop
│   ├── federated.py      # Federated learning sim
│   ├── alerts.py         # Alerts + audit log
│   ├── schemas.py        # Pydantic models
│   └── demo.py           # End-to-end demo
├── tests/
├── data/                 # SQLite DB (gitignored)
├── docs/
│   └── API.md           # API contract for frontend
├── requirements.txt
├── README.md
└── metrics.json         # Generated after training
```

## License

Hackathon submission for Ministry of Home Affairs (I4C) Problem Statement 26184.
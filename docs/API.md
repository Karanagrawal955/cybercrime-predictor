# API Contract for Frontend Integration

**Base URL:** `http://localhost:8000`  
**Auth:** Bearer token (JWT)  
**CORS:** Enabled for all origins  

---

## Authentication

### POST `/auth/login`
Login with username/password.

**Request:**
```json
{
  "username": "i4c_admin",
  "password": "admin123"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": 1,
    "username": "i4c_admin",
    "role": "i4c_admin",
    "state": null,
    "district": null,
    "bank": null,
    "is_active": true,
    "created_at": "2026-09-15T10:00:00"
  }
}
```

**Usage:** Include in subsequent requests:
```
Authorization: Bearer <access_token>
```

---

## Complaints

### POST `/complaints`
Ingest a complaint → returns prediction + intervention plan + alerts.

**Request:**
```json
{
  "complaint_id": "CFCFRMS-2026-001234",
  "timestamp": "2026-09-15T14:30:00",
  "fraud_type": "UPI Fraud",
  "amount": 250000,
  "victim_bank": "HDFC",
  "hop_count": 3,
  "mule_account_chain": [
    {
      "account_id": "HDFC1234567890",
      "bank": "HDFC",
      "city": "Mumbai",
      "hop_position": 1,
      "amount_received": 250000,
      "amount_forwarded": 240000,
      "phone_number": "+919876543210",
      "device_id": "DEV-ABC123"
    },
    {
      "account_id": "SBI0987654321",
      "bank": "SBI",
      "city": "Pune",
      "hop_position": 2,
      "amount_received": 240000,
      "amount_forwarded": 230000,
      "phone_number": "+919876543210",
      "device_id": "DEV-ABC123"
    },
    {
      "account_id": "ICICI1111111111",
      "bank": "ICICI",
      "city": "Delhi",
      "hop_position": 3,
      "amount_received": 230000,
      "amount_forwarded": 0,
      "phone_number": "+919999999999",
      "device_id": "DEV-XYZ789"
    }
  ],
  "victim_city": "Mumbai",
  "victim_state": "Maharashtra"
}
```

**Response (200):**
```json
{
  "complaint": {
    "id": 123,
    "complaint_id": "CFCFRMS-2026-001234",
    "timestamp": "2026-09-15T14:30:00",
    "fraud_type": "UPI Fraud",
    "amount": 250000,
    "victim_bank": "HDFC",
    "hop_count": 3,
    "mule_account_chain": [...],
    "victim_city": "Mumbai",
    "victim_state": "Maharashtra",
    "predicted_city": "Delhi",
    "predicted_channel": "ATM",
    "predicted_eta_min": 35,
    "prediction_probability": 0.62,
    "model_version": "v20260915_143022",
    "is_synthetic": false,
    "created_at": "2026-09-15T14:30:05"
  },
  "predictions": [
    {
      "rank": 1,
      "city": "Delhi",
      "channel": "ATM",
      "probability": 0.62,
      "eta_min": 35,
      "explanation": "Top prediction: Delhi via ATM (62% confidence). Based on gang_001's corridor history, UPI Fraud pattern, and amount bucket ₹250,000."
    },
    {
      "rank": 2,
      "city": "Mewat",
      "channel": "AePS",
      "probability": 0.23,
      "eta_min": 95,
      "explanation": "Alternative #2: Mewat via AePS (23%). Gang gang_001 has used this corridor before."
    },
    {
      "rank": 3,
      "city": "Mumbai",
      "channel": "Crypto",
      "probability": 0.15,
      "eta_min": 12,
      "explanation": "Alternative #3: Mumbai via Crypto (15%). Gang gang_001 has used this corridor before."
    }
  ],
  "intervention_plan": {
    "complaint_id": "CFCFRMS-2026-001234",
    "plans": [
      {
        "complaint_id": "CFCFRMS-2026-001234",
        "team_id": "team_001",
        "team_name": "Delhi Cyber Cell",
        "target_city": "Delhi",
        "target_channel": "ATM",
        "predicted_eta_min": 35,
        "travel_time_min": 15,
        "arrival_before_cashout": true,
        "probability": 0.62,
        "expected_recovery": 131750.0,
        "reason": "Team can reach Delhi in 15 min, cash-out predicted in 35 min. Arrive 20 min early. Probability: 62.0%",
        "action_type": "physical"
      },
      {
        "complaint_id": "CFCFRMS-2026-001234",
        "team_id": "team_009",
        "team_name": "Pune Cyber Cell",
        "target_city": "Mewat",
        "target_channel": "AePS",
        "predicted_eta_min": 95,
        "travel_time_min": 45,
        "arrival_before_cashout": true,
        "probability": 0.23,
        "expected_recovery": 32475.0,
        "reason": "Team can reach Mewat in 45 min, cash-out predicted in 95 min. Arrive 50 min early. Probability: 23.0%",
        "action_type": "physical"
      },
      {
        "complaint_id": "CFCFRMS-2026-001234",
        "team_id": null,
        "team_name": "Exchange Freeze Request",
        "target_city": "Mumbai",
        "target_channel": "Crypto",
        "predicted_eta_min": 12,
        "travel_time_min": 0,
        "arrival_before_cashout": true,
        "probability": 0.15,
        "expected_recovery": 33750.0,
        "reason": "Crypto off-ramp at Mumbai: Request exchange freeze for USDT P2P",
        "action_type": "exchange_freeze"
      }
    ],
    "total_expected_recovery": 197975.0,
    "teams_used": 2,
    "unassigned_predictions": []
  },
  "alerts": [
    {
      "id": 1,
      "complaint_id": 123,
      "severity": "HIGH",
      "channel": "ATM",
      "city": "Delhi",
      "message": "🚨 HIGH PRIORITY: Predicted cash-out at Delhi via ATM in ~35 min. Amount: ₹250,000. Gang: gang_001. Deploy team immediately.",
      "probability": 0.62,
      "eta_min": 35,
      "amount": 250000,
      "gang_id": "gang_001",
      "dispatched": {"sms": false, "email": false, "webhook": false},
      "created_at": "2026-09-15T14:30:05"
    },
    ...
  ]
}
```

### GET `/predictions/{complaint_id}`
Get predictions for a specific complaint.

**Response (200):**
```json
{
  "complaint_id": 123,
  "predictions": [...],
  "model_version": "v20260915_143022",
  "generated_at": "2026-09-15T14:35:00"
}
```

---

## Heatmap

### GET `/heatmap`
Get city risk scores for map visualization.

**Query Parameters:**
- `channel` (optional): ATM, AePS, Crypto
- `fraud_type` (optional): UPI Fraud, Investment Scam, Digital Arrest, Loan App Harassment, Phishing, QR Code Scam
- `hours` (optional, default: 24): Lookback window

**Response (200):**
```json
{
  "cells": [
    {
      "city": "Delhi",
      "lat": 28.6139,
      "lon": 77.2090,
      "risk_score": 0.72,
      "complaint_count": 45,
      "avg_amount": 185000.0,
      "top_channel": "ATM"
    },
    {
      "city": "Mumbai",
      "lat": 19.0760,
      "lon": 72.8777,
      "risk_score": 0.65,
      "complaint_count": 38,
      "avg_amount": 210000.0,
      "top_channel": "AePS"
    }
  ],
  "generated_at": "2026-09-15T14:35:00",
  "filters": {"channel": "ATM", "fraud_type": null, "hours": 24}
}
```

---

## Gangs

### GET `/gangs`
List all gangs with summary stats.

**Response (200):**
```json
[
  {
    "gang_id": "gang_001",
    "name": "Delhi-Mewat Corridor",
    "corridor": ["Delhi", "Mewat", "Gurugram"],
    "corridor_weights": [0.5, 0.3, 0.2],
    "channel_mix": {"ATM": 0.6, "AePS": 0.3, "Crypto": 0.1},
    "mule_count": 8,
    "base_city": "Delhi",
    "total_complaints": 847,
    "total_amount": 125000000.0
  }
]
```

### GET `/gangs/{gang_id}`
Get detailed gang information including mules.

**Response (200):**
```json
{
  "gang_id": "gang_001",
  "name": "Delhi-Mewat Corridor",
  "corridor": ["Delhi", "Mewat", "Gurugram"],
  "corridor_weights": [0.5, 0.3, 0.2],
  "channel_mix": {"ATM": 0.6, "AePS": 0.3, "Crypto": 0.1},
  "mule_count": 8,
  "base_city": "Delhi",
  "total_complaints": 847,
  "total_amount": 125000000.0,
  "mules": [
    {
      "account_id": "HDFC1234567890",
      "phone": "+919876543210",
      "device": "DEV-ABC123",
      "bank": "HDFC",
      "city": "Delhi"
    }
  ]
}
```

---

## Optimizer

### POST `/optimize`
Get intervention plan for given predictions.

**Request:**
```json
{
  "complaint_id": "CFCFRMS-2026-001234",
  "amount": 250000,
  "predictions": [
    {"rank": 1, "city": "Delhi", "channel": "ATM", "probability": 0.62, "eta_min": 35, "explanation": "..."},
    {"rank": 2, "city": "Mewat", "channel": "AePS", "probability": 0.23, "eta_min": 95, "explanation": "..."},
    {"rank": 3, "city": "Mumbai", "channel": "Crypto", "probability": 0.15, "eta_min": 12, "explanation": "..."}
  ],
  "team_count": 10
}
```

**Response (200):**
```json
{
  "complaint_id": "CFCFRMS-2026-001234",
  "plans": [...],
  "total_expected_recovery": 197975.0,
  "teams_used": 2,
  "unassigned_predictions": []
}
```

---

## Outcomes (Displacement + Feedback)

### POST `/outcomes`
Record intervention result → triggers displacement & feedback.

**Request:**
```json
{
  "complaint_id": "CFCFRMS-2026-001234",
  "intervention_id": 5,
  "intercepted": true,
  "amount_recovered": 180000,
  "notes": "Successfully intercepted at ATM kiosk in Connaught Place"
}
```

**Response (200):**
```json
{
  "intervention_id": 5,
  "intercepted": true,
  "amount_recovered": 180000,
  "displacement": {
    "intercepted_city": "Delhi",
    "intercepted_channel": "ATM",
    "pressure_increased": true,
    "displaced_to": "Mewat",
    "displacement_strength": 0.5,
    "displacement_anticipation": true
  },
  "corridor_updated": false
}
```

---

## Federated Learning

### POST `/federated/round`
Run a federated learning round (admin only).

**Request:**
```json
["HDFC", "SBI", "ICICI"]
```

**Response (200):**
```json
{
  "round_number": 3,
  "participating_banks": ["HDFC", "SBI", "ICICI"],
  "status": "completed",
  "aggregated_at": "2026-09-15T14:40:00",
  "raw_rows_shared": 0
}
```

### GET `/federated/status`
Get federated learning status.

**Response (200):**
```json
{
  "current_round": 4,
  "total_rounds_completed": 3,
  "recent_rounds": [
    {
      "round": 3,
      "banks": ["HDFC", "SBI", "ICICI"],
      "status": "completed",
      "completed_at": "2026-09-15T14:40:00",
      "raw_rows_shared": 0
    }
  ],
  "global_prior_available": true,
  "privacy_guarantee": "0 raw rows ever shared - only aggregated feature importances"
}
```

---

## Alerts

### GET `/alerts`
Get alerts with filtering and bilingual support.

**Query Parameters:**
- `lang`: en | hi (default: en)
- `severity`: HIGH | MEDIUM | LOW
- `channel`: ATM | AePS | Crypto
- `city`: city name
- `limit`: max results (default: 100)

**Response (200):**
```json
[
  {
    "id": 1,
    "complaint_id": 123,
    "severity": "HIGH",
    "channel": "ATM",
    "city": "Delhi",
    "message": "🚨 HIGH PRIORITY: Predicted cash-out at Delhi via ATM in ~35 min. Amount: ₹250,000. Gang: gang_001. Deploy team immediately.",
    "probability": 0.62,
    "eta_min": 35,
    "amount": 250000,
    "gang_id": "gang_001",
    "dispatched": {"sms": false, "email": false, "webhook": false},
    "created_at": "2026-09-15T14:30:05"
  }
]
```

### POST `/alerts/dispatch`
Dispatch alerts via SMS/email/webhook (simulated).

**Request:**
```json
[1, 2, 3]  // alert IDs, or null for all undispatched HIGH/MEDIUM
```

**Response (200):**
```json
{"dispatched": {"sms": 3, "email": 3, "webhook": 3}}
```

---

## Audit Log

### GET `/audit/verify`
Verify tamper-evident audit chain integrity.

**Response (200):**
```json
{
  "valid": true,
  "total_entries": 156,
  "errors": [],
  "last_hash": "a1b2c3d4e5f6..."
}
```

### GET `/audit/log`
Get audit log entries.

**Query Parameters:**
- `limit`: max results (default: 100)
- `actor`: filter by actor
- `action`: filter by action

**Response (200):**
```json
[
  {
    "id": 1,
    "timestamp": "2026-09-15T14:30:05",
    "actor": "system",
    "action": "complaint_created",
    "payload": {"complaint_id": "CFCFRMS-2026-001234", "amount": 250000, "predictions": 3},
    "prev_hash": "000000000000...",
    "current_hash": "a1b2c3d4e5f6..."
  }
]
```

---

## Metrics

### GET `/metrics`
Get model evaluation metrics.

**Response (200):**
```json
{
  "model_version": "v20260915_143022",
  "train_start_date": "2026-07-01T00:00:00",
  "train_end_date": "2026-09-01T00:00:00",
  "test_start_date": "2026-09-01T00:00:00",
  "test_end_date": "2026-09-30T23:59:59",
  "top1_accuracy": 0.187,
  "top3_accuracy": 0.423,
  "precision_at_3": 0.381,
  "mean_lead_time_min": 22.5,
  "mae_minutes": 58.3,
  "rmse_minutes": 76.8,
  "num_train_samples": 3847,
  "num_test_samples": 962,
  "notes": "METRICS FROM SYNTHETIC DATA ONLY - NOT REPRESENTATIVE OF REAL PERFORMANCE",
  "trained_at": "2026-09-15T14:30:22"
}
```

---

## Development

### POST `/dev/seed`
Regenerate synthetic data and retrain model (admin only).

**Response (200):**
```json
{
  "success": true,
  "message": "Synthetic data regenerated and model retrained",
  "data": {
    "gangs": 6,
    "mules": 45,
    "complaints": 5000
  }
}
```

---

## Error Responses

All endpoints may return:

**401 Unauthorized:**
```json
{"detail": "Could not validate credentials"}
```

**403 Forbidden:**
```json
{"detail": "Role state_officer not authorized for this action"}
```

**404 Not Found:**
```json
{"detail": "Complaint not found"}
```

**422 Validation Error:**
```json
{"detail": [{"loc": ["body", "amount"], "msg": "value is not a valid float", "type": "type_error.float"}]}
```

---

## Frontend Integration Notes

1. **Call `/complaints`** when user submits a complaint - returns everything needed for dashboard
2. **Poll `/heatmap`** every 30-60 seconds for live risk map
3. **Use `/alerts?lang=hi`** for Hindi dashboard
4. **Call `/outcomes`** when officer updates intervention status
5. **All timestamps** in ISO 8601 UTC format
6. **Amounts** in INR (paise not used)
7. **Coordinates** in WGS84 (lat/lon)
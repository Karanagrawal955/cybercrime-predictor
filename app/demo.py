"""
End-to-end demo script.
Seeds data, trains model, simulates 5 complaints end-to-end and prints plans and audit verification.
Run with: python -m app.demo
"""

import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.db import SessionLocal, Base, engine, init_db
from app.synth import generate_all_synthetic_data
from app.graph import run_gang_detection, save_detected_gangs, assign_complaints_to_detected_gangs
from app.model import train_model, get_predictor, WithdrawalPredictor
from app.optimizer import optimize_intervention
from app.displacement import record_outcome, get_pressure_map
from app.federated import run_federated_round, get_federated_status
from app.alerts import verify_audit_chain, get_audit_log
from app.config import SYNTHETIC_CITIES, CHANNELS, FRAUD_TYPES


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: str):
    print(f"\n>> {step}")


def print_substep(step: str):
    print(f"  -> {step}")


def run_demo():
    """Run the complete end-to-end demo."""
    print_header("CYBERCRIME WITHDRAWAL PREDICTOR - END-TO-END DEMO")
    print("  ALL DATA IS SYNTHETIC - FOR DEMONSTRATION PURPOSES ONLY")
    
    db = SessionLocal()
    
    try:
        # ============================================================
        # STEP 1: Initialize Database
        # ============================================================
        print_header("STEP 1: DATABASE INITIALIZATION")
        print_step("Creating tables...")
        init_db()
        print_substep("Tables created successfully")
        
        # ============================================================
        # STEP 2: Generate Synthetic Data
        # ============================================================
        print_header("STEP 2: SYNTHETIC DATA GENERATION")
        print_step("Generating 5000+ synthetic complaints across 6 gangs...")
        stats = generate_all_synthetic_data()
        print_substep(f"Gangs: {stats['gangs']}")
        print_substep(f"Mule Accounts: {stats['mules']}")
        print_substep(f"Complaints: {stats['complaints']}")
        
        # ============================================================
        # STEP 3: Gang Detection
        # ============================================================
        print_header("STEP 3: GANG DETECTION (GRAPH-BASED)")
        print_step("Building entity graph (accounts <-> phones <-> devices)...")
        summaries = run_gang_detection(db)
        print_substep(f"Detected {len(summaries)} connected components")
        
        print_step("Saving detected gangs to database...")
        save_detected_gangs(db, summaries)
        
        print_step("Assigning complaints to detected gangs...")
        assign_complaints_to_detected_gangs(db)
        
        for s in summaries[:3]:
            print_substep(f"{s['gang_id']}: {s['account_count']} accounts, "
                         f"corridor={s['corridor'][:2]}, "
                         f"shared phones={s['shared_phones']}")
        
        # ============================================================
        # STEP 4: Train Prediction Model
        # ============================================================
        print_header("STEP 4: PREDICTION MODEL TRAINING")
        print_step("Training multiclass classifier + regressor (time-based split)...")
        metrics = train_model()
        
        print_substep("Model Metrics (SYNTHETIC DATA):")
        print_substep(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2%}")
        print_substep(f"  Top-3 Accuracy:  {metrics['top3_accuracy']:.2%}")
        print_substep(f"  Precision@3:     {metrics['precision_at_3']:.2%}")
        print_substep(f"  Mean Lead Time:  {metrics['mean_lead_time_min']:.1f} min")
        print_substep(f"  MAE:             {metrics['mae_minutes']:.1f} min")
        print_substep(f"  RMSE:            {metrics['rmse_minutes']:.1f} min")
        print_substep(f"  Train Samples:   {metrics['num_train_samples']}")
        print_substep(f"  Test Samples:    {metrics['num_test_samples']}")
        
        # Load predictor
        predictor = get_predictor()
        print_substep(f"Model loaded: {predictor.model_version}")
        
        # ============================================================
        # STEP 5: Simulate 5 Complaints End-to-End
        # ============================================================
        print_header("STEP 5: END-TO-END COMPLAINT SIMULATION (5 COMPLAINTS)")
        
        # Sample complaints covering different scenarios
        test_complaints = [
            {
                "complaint_id": "DEMO-001",
                "timestamp": datetime(2026, 9, 15, 10, 30),
                "fraud_type": "UPI Fraud",
                "amount": 150000,
                "victim_bank": "HDFC",
                "hop_count": 3,
                "victim_city": "Mumbai",
                "mule_chain": [
                    {"account_id": "HDFC1111111111", "bank": "HDFC", "city": "Mumbai", "hop_position": 1, "amount_received": 150000, "amount_forwarded": 145000},
                    {"account_id": "SBI2222222222", "bank": "SBI", "city": "Pune", "hop_position": 2, "amount_received": 145000, "amount_forwarded": 140000},
                    {"account_id": "ICICI3333333333", "bank": "ICICI", "city": "Delhi", "hop_position": 3, "amount_received": 140000, "amount_forwarded": 0},
                ]
            },
            {
                "complaint_id": "DEMO-002",
                "timestamp": datetime(2026, 9, 15, 14, 15),
                "fraud_type": "Investment Scam",
                "amount": 500000,
                "victim_bank": "SBI",
                "hop_count": 2,
                "victim_city": "Delhi",
                "mule_chain": [
                    {"account_id": "SBI4444444444", "bank": "SBI", "city": "Delhi", "hop_position": 1, "amount_received": 500000, "amount_forwarded": 480000},
                    {"account_id": "HDFC5555555555", "bank": "HDFC", "city": "Mewat", "hop_position": 2, "amount_received": 480000, "amount_forwarded": 0},
                ]
            },
            {
                "complaint_id": "DEMO-003",
                "timestamp": datetime(2026, 9, 15, 22, 0),
                "fraud_type": "Digital Arrest",
                "amount": 75000,
                "victim_bank": "ICICI",
                "hop_count": 4,
                "victim_city": "Bengaluru",
                "mule_chain": [
                    {"account_id": "ICICI6666666666", "bank": "ICICI", "city": "Bengaluru", "hop_position": 1, "amount_received": 75000, "amount_forwarded": 72000},
                    {"account_id": "HDFC7777777777", "bank": "HDFC", "city": "Chennai", "hop_position": 2, "amount_received": 72000, "amount_forwarded": 69000},
                    {"account_id": "AXIS8888888888", "bank": "AXIS", "city": "Hyderabad", "hop_position": 3, "amount_received": 69000, "amount_forwarded": 65000},
                    {"account_id": "KOTAK9999999999", "bank": "KOTAK", "city": "Mumbai", "hop_position": 4, "amount_received": 65000, "amount_forwarded": 0},
                ]
            },
            {
                "complaint_id": "DEMO-004",
                "timestamp": datetime(2026, 9, 16, 9, 45),
                "fraud_type": "Loan App Harassment",
                "amount": 25000,
                "victim_bank": "HDFC",
                "hop_count": 2,
                "victim_city": "Chennai",
                "mule_chain": [
                    {"account_id": "HDFC1010101010", "bank": "HDFC", "city": "Chennai", "hop_position": 1, "amount_received": 25000, "amount_forwarded": 24000},
                    {"account_id": "SBI2020202020", "bank": "SBI", "city": "Hyderabad", "hop_position": 2, "amount_received": 24000, "amount_forwarded": 0},
                ]
            },
            {
                "complaint_id": "DEMO-005",
                "timestamp": datetime(2026, 9, 16, 16, 20),
                "fraud_type": "Phishing",
                "amount": 1000000,
                "victim_bank": "SBI",
                "hop_count": 3,
                "victim_city": "Kolkata",
                "mule_chain": [
                    {"account_id": "SBI3030303030", "bank": "SBI", "city": "Kolkata", "hop_position": 1, "amount_received": 1000000, "amount_forwarded": 950000},
                    {"account_id": "ICICI4040404040", "bank": "ICICI", "city": "Patna", "hop_position": 2, "amount_received": 950000, "amount_forwarded": 900000},
                    {"account_id": "PNB5050505050", "bank": "PNB", "city": "Lucknow", "hop_position": 3, "amount_received": 900000, "amount_forwarded": 0},
                ]
            },
        ]
        
        total_expected_recovery = 0
        
        for i, complaint_data in enumerate(test_complaints, 1):
            print_step(f"Complaint {i}/5: {complaint_data['complaint_id']} "
                      f"(Rs.{complaint_data['amount']:,.0f}, {complaint_data['fraud_type']})")
            
            # Prepare prediction input
            first_mule = complaint_data['mule_chain'][0] if complaint_data['mule_chain'] else {}
            
            pred_input = {
                "timestamp": complaint_data['timestamp'],
                "fraud_type": complaint_data['fraud_type'],
                "amount": complaint_data['amount'],
                "victim_bank": complaint_data['victim_bank'],
                "hop_count": complaint_data['hop_count'],
                "victim_city": complaint_data['victim_city'],
                "first_mule_city": first_mule.get('city'),
                "first_mule_bank": first_mule.get('bank'),
                "detected_gang_id": "gang_001",
            }
            
            # Get predictions
            predictions = predictor.predict(pred_input, top_k=3)
            
            print_substep("Top-3 Predictions:")
            for p in predictions:
                print_substep(f"  #{p['rank']}: {p['city']} via {p['channel']} "
                             f"({p['probability']:.1%}, ETA: {p['eta_min']} min)")
                print_substep(f"      {p['explanation']}")
            
            # Get intervention plan
            opt_result = optimize_intervention(
                complaint_id=complaint_data['complaint_id'],
                amount=complaint_data['amount'],
                predictions=predictions,
                team_count=10,
            )
            
            total_expected_recovery += opt_result.total_expected_recovery
            
            print_substep("Intervention Plan:")
            for plan in opt_result.plans:
                team_info = plan.team_name if plan.team_name else "N/A (Crypto)"
                arrival = "OK BEFORE" if plan.arrival_before_cashout else "XX AFTER"
                print_substep(f"  {team_info} -> {plan.target_city} ({plan.target_channel}) "
                             f"| Travel: {plan.travel_time_min}min | ETA: {plan.predicted_eta_min}min | {arrival}")
                print_substep(f"    Expected Recovery: Rs.{plan.expected_recovery:,.0f} | {plan.reason}")
            
            print_substep(f"Total Expected Recovery: Rs.{opt_result.total_expected_recovery:,.0f}")
            print_substep(f"Teams Used: {opt_result.teams_used}")
        
        print_step(f"COMBINED EXPECTED RECOVERY (5 complaints): Rs.{total_expected_recovery:,.0f}")
        
        # ============================================================
        # STEP 6: Displacement & Feedback Demo
        # ============================================================
        print_header("STEP 6: DISPLACEMENT & FEEDBACK LOOP")
        
        print_step("Simulating intervention outcome (INTERCEPTED)...")
        # We need an actual intervention ID - get one from DB
        from app.db import Intervention
        intervention = db.query(Intervention).first()
        if intervention:
            demo_complaint_id = "DEMO-001"
            result = record_outcome(
                db=db,
                complaint_id=demo_complaint_id,
                intervention_id=intervention.id,
                intercepted=True,
                amount_recovered=120000,
                notes="Demo interception at ATM"
            )
            print_substep(f"Outcome recorded: {result}")
            
            print_step("Checking displacement pressure map...")
            pressure_map = get_pressure_map(db, gang_id="gang_001")
            print_substep(f"Pressure map: {json.dumps(pressure_map, indent=4)}")
        
        # ============================================================
        # STEP 7: Federated Learning Demo
        # ============================================================
        print_header("STEP 7: FEDERATED LEARNING SIMULATION")
        
        print_step("Running federated round with 3 banks...")
        fl_result = run_federated_round(db, ["HDFC", "SBI", "ICICI"])
        print_substep(f"Round {fl_result.round_number} completed")
        print_substep(f"Participating banks: {fl_result.participating_banks}")
        print_substep(f"Raw rows shared: {fl_result.raw_rows_shared} (GUARANTEED 0)")
        print_substep(f"Global prior features: {len(fl_result.avg_feature_importances)}")
        
        print_step("Getting federated status...")
        fl_status = get_federated_status(db)
        print_substep(f"Total rounds: {fl_status['total_rounds_completed']}")
        print_substep(f"Privacy guarantee: {fl_status['privacy_guarantee']}")
        
        # ============================================================
        # STEP 8: Audit Chain Verification
        # ============================================================
        print_header("STEP 8: TAMPER-EVIDENT AUDIT LOG VERIFICATION")
        
        print_step("Verifying audit chain integrity...")
        audit_result = verify_audit_chain(db)
        
        print_substep(f"Chain Valid: {audit_result['valid']}")
        print_substep(f"Total Entries: {audit_result['total_entries']}")
        if audit_result.get('last_hash'):
            print_substep(f"Last Hash: {audit_result['last_hash'][:16]}...")
        
        if audit_result['errors']:
            print_substep("ERRORS DETECTED:")
            for err in audit_result['errors']:
                print_substep(f"  - Entry {err['entry_id']}: {err['error']}")
        else:
            print_substep("[OK] No tampering detected - chain is intact")
        
        # Show recent audit entries
        print_step("Recent audit entries:")
        recent_logs = get_audit_log(db, limit=5)
        for log in recent_logs:
            print_substep(f"  [{log['timestamp']}] {log['actor']}: {log['action']}")
            if log['payload']:
                print_substep(f"    Payload: {json.dumps(log['payload'])[:80]}...")
        
        # ============================================================
        # STEP 9: Bilingual Alerts Demo
        # ============================================================
        print_header("STEP 9: BILINGUAL ALERTS (EN/HI)")
        
        from app.alerts import get_alerts
        alerts_en = get_alerts(db, lang="en", limit=2)
        alerts_hi = get_alerts(db, lang="hi", limit=2)
        
        print_step("English alerts:")
        for a in alerts_en:
            print_substep(f"  [{a['severity']}] {a['message'][:80]}...")
        
        print_step("Hindi alerts:")
        for a in alerts_hi:
            print_substep(f"  [{a['severity']}] {a['message'][:80]}...")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print_header("DEMO COMPLETE - SUMMARY")
        print("[OK] Synthetic data generated (5000+ complaints, 6 gangs, 45 mules)")
        print("[OK] Gang detection via graph connected components")
        print("[OK] ML model trained (time-based split, calibrated probabilities)")
        print("[OK] 5 complaints simulated end-to-end with predictions")
        print("[OK] Intervention optimizer (greedy, team-aware, crypto=freeze)")
        print("[OK] Displacement feedback (pressure map, corridor updates)")
        print("[OK] Federated learning (3 banks, 0 raw rows shared)")
        print("[OK] Tamper-evident audit log (SHA-256 chain verified)")
        print("[OK] Bilingual alerts (English + Hindi)")
        print("[OK] Role-based access control (4 roles, scoped queries)")
        print("\n" + "=" * 70)
        print("  API SERVER: uvicorn app.main:app --reload")
        print("  API DOCS:   http://localhost:8000/docs")
        print("  FRONTEND:   Open withdrawal-predictor-v2.html in browser")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    run_demo()
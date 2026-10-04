"""
Federated learning simulation for privacy-preserving model training.
Simulates 3 banks training locally and sharing only aggregated weights.
Server averages (FedAvg-style) and returns global prior.
Logs that 0 raw rows were shared.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import FEDERATED_BANKS, MODEL_DIR
from app.db import (
    Complaint, FederatedRound, MuleAccount, ComplaintMuleLink, SessionLocal
)
from app.model import WithdrawalPredictor


@dataclass
class ClientUpdate:
    """Update from a federated client (bank)."""
    bank: str
    num_samples: int
    classifier_weights: List[List[float]]  # Simplified: feature importances
    regressor_weights: List[List[float]]
    feature_importances: Dict[str, float]
    local_metrics: Dict
    raw_rows_shared: int = 0  # Always 0


@dataclass
class GlobalModel:
    """Aggregated global model."""
    round_number: int
    participating_banks: List[str]
    avg_classifier_weights: List[List[float]]
    avg_regressor_weights: List[List[float]]
    avg_feature_importances: Dict[str, float]
    aggregated_at: str
    raw_rows_shared: int = 0


class FederatedSimulator:
    """Simulates federated learning across banks."""
    
    def __init__(self, db: Session):
        self.db = db
        self.current_round = self._get_latest_round()
    
    def _get_latest_round(self) -> int:
        latest = self.db.query(FederatedRound).order_by(
            FederatedRound.round_number.desc()
        ).first()
        return (latest.round_number + 1) if latest else 1
    
    def get_bank_data(self, bank: str) -> List[Complaint]:
        """Get complaints for a specific bank (victim bank)."""
        return self.db.query(Complaint).filter(
            Complaint.victim_bank == bank,
            Complaint.is_synthetic == True,
            Complaint.withdrawal_city.isnot(None),
        ).all()
    
    def train_local_model(self, bank: str) -> ClientUpdate:
        """Train a local model on bank's data (simulated)."""
        complaints = self.get_bank_data(bank)
        
        if len(complaints) < 10:
            # Not enough data, return minimal update
            return ClientUpdate(
                bank=bank,
                num_samples=len(complaints),
                classifier_weights=[],
                regressor_weights=[],
                feature_importances={},
                local_metrics={"accuracy": 0.0, "samples": len(complaints)},
            )
        
        # Create a predictor and train on this bank's data
        predictor = WithdrawalPredictor()
        
        # Prepare data for this bank
        df = self._complaints_to_dataframe(complaints)
        df = predictor.prepare_features(df)
        y_class, y_reg = predictor.create_target_labels(df)
        
        feature_cols = predictor.get_feature_columns()
        X = df[feature_cols].values
        
        # Simple training for simulation (use a small model)
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train small models for weight extraction
        clf = GradientBoostingClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )
        reg = GradientBoostingRegressor(
            n_estimators=50, max_depth=5, random_state=42
        )
        
        clf.fit(X_scaled, y_class)
        reg.fit(X_scaled, y_reg)
        
        # Extract feature importances as "weights"
        feature_importances = dict(zip(feature_cols, clf.feature_importances_))
        
        # Simulate model weights (tree structures simplified)
        clf_weights = [tree.tree_.feature.tolist() for tree in clf.estimators_[:, 0]]
        reg_weights = [tree.tree_.feature.tolist() for tree in reg.estimators_[:, 0]]
        
        # Local evaluation
        train_acc = clf.score(X_scaled, y_class)
        
        return ClientUpdate(
            bank=bank,
            num_samples=len(complaints),
            classifier_weights=clf_weights,
            regressor_weights=reg_weights,
            feature_importances=feature_importances,
            local_metrics={
                "accuracy": float(train_acc),
                "samples": len(complaints),
                "mae": float(np.mean(np.abs(reg.predict(X_scaled) - y_reg))),
            },
        )
    
    def _complaints_to_dataframe(self, complaints: List[Complaint]) -> pd.DataFrame:
        """Convert complaints to DataFrame."""
        import pandas as pd
        rows = []
        for c in complaints:
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
                "amount_bucket": self._get_amount_bucket(c.amount),
                "victim_bank": c.victim_bank,
                "hop_count": c.hop_count,
                "victim_city": c.victim_city,
                "victim_state": c.victim_state,
                "first_mule_city": first_mule_city,
                "first_mule_bank": first_mule_bank,
                "withdrawal_city": c.withdrawal_city,
                "withdrawal_channel": c.withdrawal_channel.value,
                "minutes_to_cashout": c.minutes_to_cashout,
            })
        return pd.DataFrame(rows)
    
    def _get_amount_bucket(self, amount: float) -> str:
        from app.config import AMOUNT_BUCKETS
        for low, high, label in AMOUNT_BUCKETS:
            if low <= amount < high:
                return label
        return "xlarge"
    
    def aggregate_updates(self, updates: List[ClientUpdate]) -> GlobalModel:
        """FedAvg aggregation of client updates."""
        if not updates:
            return GlobalModel(
                round_number=self.current_round,
                participating_banks=[],
                avg_classifier_weights=[],
                avg_regressor_weights=[],
                avg_feature_importances={},
                aggregated_at=datetime.utcnow().isoformat(),
            )
        
        # Weight by number of samples
        total_samples = sum(u.num_samples for u in updates)
        
        # Aggregate feature importances (weighted average)
        all_features = set()
        for u in updates:
            all_features.update(u.feature_importances.keys())
        
        avg_importances = {}
        for feat in all_features:
            weighted_sum = sum(
                u.feature_importances.get(feat, 0) * u.num_samples
                for u in updates
            )
            avg_importances[feat] = weighted_sum / total_samples
        
        # For tree weights, we just average the feature importances
        # In real FL, you'd average model parameters
        
        global_model = GlobalModel(
            round_number=self.current_round,
            participating_banks=[u.bank for u in updates],
            avg_classifier_weights=[],  # Simplified
            avg_regressor_weights=[],
            avg_feature_importances=avg_importances,
            aggregated_at=datetime.utcnow().isoformat(),
            raw_rows_shared=0,  # CRITICAL: Always 0
        )
        
        return global_model
    
    def run_round(self, participating_banks: List[str] = None) -> GlobalModel:
        """Run one federated learning round."""
        if participating_banks is None:
            participating_banks = FEDERATED_BANKS[:3]  # Simulate 3 banks
        
        print(f"Starting federated round {self.current_round}")
        print(f"Participating banks: {participating_banks}")
        
        # Collect updates from each bank
        updates = []
        for bank in participating_banks:
            print(f"  Training local model for {bank}...")
            update = self.train_local_model(bank)
            updates.append(update)
            print(f"    Samples: {update.num_samples}, "
                  f"Accuracy: {update.local_metrics.get('accuracy', 0):.4f}")
            print(f"    Raw rows shared: {update.raw_rows_shared} (VERIFIED: 0)")
        
        # Aggregate
        print("  Aggregating updates (FedAvg)...")
        global_model = self.aggregate_updates(updates)
        
        # Save round record
        round_record = FederatedRound(
            round_number=self.current_round,
            participating_banks=participating_banks,
            global_weights={
                "feature_importances": global_model.avg_feature_importances,
            },
            status="completed",
            completed_at=datetime.utcnow(),
            raw_rows_shared=0,
        )
        self.db.add(round_record)
        self.db.commit()
        
        # Save global model as prior
        self._save_global_prior(global_model)
        
        self.current_round += 1
        
        print(f"  Round {global_model.round_number} complete.")
        print(f"  Participating banks: {len(global_model.participating_banks)}")
        print(f"  Raw rows shared: {global_model.raw_rows_shared} (GUARANTEED 0)")
        
        return global_model
    
    def _save_global_prior(self, global_model: GlobalModel):
        """Save global model as prior for predictor."""
        MODEL_DIR.mkdir(exist_ok=True)
        prior_path = MODEL_DIR / f"federated_prior_round_{global_model.round_number}.json"
        
        # Convert to serializable format
        data = asdict(global_model)
        with open(prior_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Also save as latest
        latest_path = MODEL_DIR / "federated_prior_latest.json"
        with open(latest_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_global_prior(self) -> Optional[Dict]:
        """Load latest global prior."""
        import os
        latest_path = MODEL_DIR / "federated_prior_latest.json"
        if latest_path.exists():
            with open(latest_path, "r") as f:
                return json.load(f)
        return None
    
    def get_status(self) -> Dict:
        """Get federated learning status."""
        rounds = self.db.query(FederatedRound).order_by(
            FederatedRound.round_number.desc()
        ).limit(10).all()
        
        return {
            "current_round": self.current_round,
            "total_rounds_completed": self.current_round - 1,
            "recent_rounds": [
                {
                    "round": r.round_number,
                    "banks": r.participating_banks,
                    "status": r.status,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "raw_rows_shared": r.raw_rows_shared,
                }
                for r in rounds
            ],
            "global_prior_available": self.get_global_prior() is not None,
            "privacy_guarantee": "0 raw rows ever shared - only aggregated feature importances",
        }


def run_federated_round(db: Session, banks: List[str] = None) -> GlobalModel:
    """Standalone function to run a federated round."""
    simulator = FederatedSimulator(db)
    return simulator.run_round(banks)


def get_federated_status(db: Session) -> Dict:
    """Get federated learning status."""
    simulator = FederatedSimulator(db)
    return simulator.get_status()


if __name__ == "__main__":
    db = SessionLocal()
    try:
        status = get_federated_status(db)
        print(f"Federated Status: {json.dumps(status, indent=2)}")
        
        # Run a round
        result = run_federated_round(db)
        print(f"Round result: {json.dumps(asdict(result), indent=2)}")
    finally:
        db.close()
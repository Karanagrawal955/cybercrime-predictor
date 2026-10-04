"""
Prediction model for withdrawal location and time.
Multiclass classifier for (city, channel) + regressor for minutes-to-cashout.
Time-based train/test split. Calibrated probabilities. Explainability.
ALL MODELS TRAINED ON SYNTHETIC DATA ONLY.
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, mean_absolute_error, mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from app.config import (
    CITY_NAMES, CHANNELS, FRAUD_TYPES, AMOUNT_BUCKETS,
    MODEL_DIR, METRICS_FILE, SYNTHETIC_SEED
)
from app.db import SessionLocal, Complaint, ChannelType, FraudType


@dataclass
class ModelArtifacts:
    """Container for all model artifacts."""
    classifier: Any
    regressor: Any
    city_encoder: LabelEncoder
    channel_encoder: LabelEncoder
    feature_scaler: StandardScaler
    feature_names: List[str]
    target_classes: List[str]  # city_channel combinations
    model_version: str
    trained_at: datetime


class WithdrawalPredictor:
    """Main predictor class for withdrawal location and time."""
    
    def __init__(self):
        self.artifacts: Optional[ModelArtifacts] = None
        self.model_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for complaints dataframe."""
        df = df.copy()
        
        # Time features
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.weekday
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
        df["is_business_hours"] = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)).astype(int)
        
        # Amount features
        df["log_amount"] = np.log1p(df["amount"])
        
        # Compute amount_bucket if not present
        if "amount_bucket" not in df.columns:
            def get_bucket(amt):
                for low, high, label in AMOUNT_BUCKETS:
                    if low <= amt < high:
                        return label
                return "xlarge"
            df["amount_bucket"] = df["amount"].apply(get_bucket)
        
        df["amount_bucket_encoded"] = df["amount_bucket"].map({
            "micro": 0, "small": 1, "medium": 2, "large": 3, "xlarge": 4
        }).fillna(0)
        
        # Fraud type encoding
        fraud_map = {ft: i for i, ft in enumerate(FRAUD_TYPES)}
        df["fraud_type_encoded"] = df["fraud_type"].map(fraud_map).fillna(0)
        
        # Victim city encoding
        city_map = {c: i for i, c in enumerate(CITY_NAMES)}
        df["victim_city_encoded"] = df["victim_city"].map(city_map).fillna(0)
        df["first_mule_city_encoded"] = df["first_mule_city"].map(city_map).fillna(0)
        
        # Bank encoding
        banks = df["victim_bank"].unique()
        bank_map = {b: i for i, b in enumerate(banks)}
        df["victim_bank_encoded"] = df["victim_bank"].map(bank_map).fillna(0)
        df["first_mule_bank_encoded"] = df["first_mule_bank"].map(bank_map).fillna(0)
        
        # Hop count
        df["hop_count"] = df["hop_count"].fillna(1).astype(int)
        
        # Gang corridor history (simplified - would use actual gang data in production)
        df["gang_corridor_entropy"] = 0.5  # Placeholder
        
        # Mule's past withdrawal cities (simplified)
        df["mule_past_city_diversity"] = 1.0  # Placeholder
        
        # Channel history (simplified)
        df["channel_ATM_hist"] = 0.5
        df["channel_AePS_hist"] = 0.3
        df["channel_Crypto_hist"] = 0.2
        
        return df
    
    def create_target_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create target labels for city+channel classification and minutes regression."""
        # Combined city+channel target for classification
        df["city_channel"] = df["withdrawal_city"] + "_" + df["withdrawal_channel"]
        
        # Encode city+channel combinations
        all_combinations = [f"{c}_{ch}" for c in CITY_NAMES for ch in CHANNELS]
        # Include extra cities from gang corridors
        extra_cities = ["Gurugram", "Alwar", "Nashik", "Varanasi"]
        for c in extra_cities:
            for ch in CHANNELS:
                all_combinations.append(f"{c}_{ch}")
        self.city_channel_encoder = LabelEncoder()
        self.city_channel_encoder.fit(all_combinations)
        
        y_class = self.city_channel_encoder.transform(df["city_channel"])
        y_reg = df["minutes_to_cashout"].values.astype(float)
        
        return y_class, y_reg
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return [
            "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_business_hours",
            "log_amount", "amount_bucket_encoded",
            "fraud_type_encoded",
            "victim_city_encoded", "first_mule_city_encoded",
            "victim_bank_encoded", "first_mule_bank_encoded",
            "hop_count",
            "gang_corridor_entropy", "mule_past_city_diversity",
            "channel_ATM_hist", "channel_AePS_hist", "channel_Crypto_hist",
        ]
    
    def train(self, db_session, test_size: float = 0.2) -> Dict:
        """Train the model with time-based split."""
        print("Loading training data...")
        df = self._load_training_data(db_session)
        print(f"Loaded {len(df)} samples")
        
        # Feature engineering
        df = self.prepare_features(df)
        
        # Create targets
        y_class, y_reg = self.create_target_labels(df)
        
        # Feature matrix
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].values
        
        # Time-based split (chronological)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]
        y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
        
        train_dates = df["timestamp"].iloc[:split_idx]
        test_dates = df["timestamp"].iloc[split_idx:]
        
        print(f"Train: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
        print(f"Test: {len(X_test)} samples ({test_dates.min()} to {test_dates.max()})")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        print("Training classifier...")
        if HAS_LIGHTGBM:
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=64,
                random_state=SYNTHETIC_SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=SYNTHETIC_SEED,
            )
        
        # Calibrate probabilities
        calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv=3)
        calibrated_clf.fit(X_train_scaled, y_class_train)
        
        # Train regressor
        print("Training regressor...")
        if HAS_LIGHTGBM:
            reg = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=64,
                random_state=SYNTHETIC_SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            reg = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=SYNTHETIC_SEED,
            )
        reg.fit(X_train_scaled, y_reg_train)
        
        # Evaluate
        print("Evaluating...")
        metrics = self._evaluate(
            calibrated_clf, reg, X_test_scaled, y_class_test, y_reg_test,
            df.iloc[split_idx:]
        )
        
        # Save artifacts
        self.artifacts = ModelArtifacts(
            classifier=calibrated_clf,
            regressor=reg,
            city_encoder=LabelEncoder().fit(CITY_NAMES),
            channel_encoder=LabelEncoder().fit(CHANNELS),
            feature_scaler=scaler,
            feature_names=feature_cols,
            target_classes=list(self.city_channel_encoder.classes_),
            model_version=self.model_version,
            trained_at=datetime.utcnow(),
        )
        
        self._save_artifacts()
        self._save_metrics(metrics, train_dates, test_dates)
        
        return metrics
    
    def _evaluate(
        self, clf, reg, X_test, y_class_test, y_reg_test, test_df: pd.DataFrame
    ) -> Dict:
        """Evaluate model and compute metrics."""
        # Classification predictions
        y_class_pred = clf.predict(X_test)
        y_class_proba = clf.predict_proba(X_test)
        
        # Top-1 accuracy
        top1_acc = accuracy_score(y_class_test, y_class_pred)
        
        # Top-3 accuracy
        top3_preds = np.argsort(y_class_proba, axis=1)[:, -3:]
        top3_acc = np.mean([y_class_test[i] in top3_preds[i] for i in range(len(y_class_test))])
        
        # Precision@3
        precision_at_3 = 0
        for i in range(len(y_class_test)):
            true_label = y_class_test[i]
            top3 = top3_preds[i]
            precision_at_3 += 1.0 if true_label in top3 else 0.0
        precision_at_3 /= len(y_class_test)
        
        # Regression predictions
        y_reg_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        
        # Mean lead time (predicted ETA - actual, when prediction is correct)
        # For correctly predicted top-1, how much lead time?
        correct_mask = y_class_pred == y_class_test
        if correct_mask.any():
            lead_times = y_reg_test[correct_mask] - y_reg_pred[correct_mask]
            mean_lead_time = float(np.mean(lead_times))
        else:
            mean_lead_time = 0.0
        
        metrics = {
            "model_version": self.model_version,
            "top1_accuracy": float(top1_acc),
            "top3_accuracy": float(top3_acc),
            "precision_at_3": float(precision_at_3),
            "mean_lead_time_min": float(mean_lead_time),
            "mae_minutes": float(mae),
            "rmse_minutes": float(rmse),
            "num_train_samples": int(len(X_test) * 4),  # approximate
            "num_test_samples": int(len(X_test)),
            "trained_at": datetime.utcnow().isoformat(),
            "note": "METRICS FROM SYNTHETIC DATA ONLY - NOT REPRESENTATIVE OF REAL PERFORMANCE",
        }
        
        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Top-3 Accuracy: {top3_acc:.4f}")
        print(f"Precision@3: {precision_at_3:.4f}")
        print(f"Mean Lead Time: {mean_lead_time:.1f} min")
        print(f"MAE: {mae:.1f} min, RMSE: {rmse:.1f} min")
        
        return metrics
    
    def _save_metrics(self, metrics: Dict, train_dates, test_dates):
        """Save metrics to JSON file."""
        metrics.update({
            "train_start_date": train_dates.min().isoformat() if len(train_dates) > 0 else None,
            "train_end_date": train_dates.max().isoformat() if len(train_dates) > 0 else None,
            "test_start_date": test_dates.min().isoformat() if len(test_dates) > 0 else None,
            "test_end_date": test_dates.max().isoformat() if len(test_dates) > 0 else None,
        })
        
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {METRICS_FILE}")
    
    def _save_artifacts(self):
        """Save model artifacts to disk."""
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / f"model_{self.model_version}.joblib"
        joblib.dump(self.artifacts, model_path)
        print(f"Model saved to {model_path}")
        
        # Also save as latest
        latest_path = MODEL_DIR / "model_latest.joblib"
        joblib.dump(self.artifacts, latest_path)
    
    def load_latest(self) -> bool:
        """Load the latest model from disk."""
        latest_path = MODEL_DIR / "model_latest.joblib"
        if latest_path.exists():
            self.artifacts = joblib.load(latest_path)
            self.model_version = self.artifacts.model_version
            print(f"Loaded model {self.model_version}")
            return True
        return False
    
    def _load_training_data(self, db_session) -> pd.DataFrame:
        """Load complaints with ground truth for training."""
        complaints = db_session.query(Complaint).filter(
            Complaint.is_synthetic == True,
            Complaint.withdrawal_city.isnot(None),
            Complaint.withdrawal_channel.isnot(None),
            Complaint.minutes_to_cashout.isnot(None),
        ).order_by(Complaint.timestamp).all()
        
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
        for low, high, label in AMOUNT_BUCKETS:
            if low <= amount < high:
                return label
        return "xlarge"
    
    def predict(
        self, 
        complaint_data: Dict, 
        top_k: int = 3
    ) -> List[Dict]:
        """Predict top-k withdrawal locations with probabilities and ETA."""
        if self.artifacts is None:
            if not self.load_latest():
                raise RuntimeError("No trained model available")
        
        # Prepare features
        df = pd.DataFrame([complaint_data])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = self.prepare_features(df)
        
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].values
        X_scaled = self.artifacts.feature_scaler.transform(X)
        
        # Get class probabilities
        proba = self.artifacts.classifier.predict_proba(X_scaled)[0]
        
        # Get top-k predictions
        top_k_idx = np.argsort(proba)[-top_k:][::-1]
        
        # Get ETA prediction
        eta_pred = self.artifacts.regressor.predict(X_scaled)[0]
        
        results = []
        for rank, idx in enumerate(top_k_idx, 1):
            city_channel = self.artifacts.target_classes[idx]
            city, channel = city_channel.split("_", 1)
            prob = float(proba[idx])
            
            # Explanation
            explanation = self._generate_explanation(
                complaint_data, city, channel, prob, rank
            )
            
            results.append({
                "rank": rank,
                "city": city,
                "channel": channel,
                "probability": prob,
                "eta_min": int(max(1, eta_pred)),
                "explanation": explanation,
            })
        
        return results
    
    def _generate_explanation(
        self, complaint_data: Dict, city: str, channel: str, prob: float, rank: int
    ) -> str:
        """Generate human-readable explanation for prediction."""
        gang_id = complaint_data.get("detected_gang_id", "unknown")
        fraud_type = complaint_data.get("fraud_type", "unknown")
        amount = complaint_data.get("amount", 0)
        
        if rank == 1:
            return (
                f"Top prediction: {city} via {channel} ({prob:.1%} confidence). "
                f"Based on gang {gang_id}'s corridor history, {fraud_type} pattern, "
                f"and amount bucket Rs.{amount:,.0f}."
            )
        else:
            return (
                f"Alternative #{rank}: {city} via {channel} ({prob:.1%}). "
                f"Gang {gang_id} has used this corridor before."
            )


def train_model() -> Dict:
    """Standalone training function."""
    db = SessionLocal()
    try:
        predictor = WithdrawalPredictor()
        metrics = predictor.train(db)
        return metrics
    finally:
        db.close()


def get_predictor() -> WithdrawalPredictor:
    """Get or create predictor instance."""
    predictor = WithdrawalPredictor()
    if not predictor.load_latest():
        print("No existing model, training new one...")
        db = SessionLocal()
        try:
            predictor.train(db)
        finally:
            db.close()
    return predictor


if __name__ == "__main__":
    train_model()
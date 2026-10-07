"""
Layer 3: Intelligence Layer — Random Forest Risk Prediction Engine
Predicts continuous spoilage risk score, categorical risk tier, and days-to-threshold.
Grounds predictions in grain physiology thresholds and feature interactions.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.layer2_features.feature_pipeline import (
    extract_features,
    features_to_vector,
    FEATURE_NAMES
)
from agri_sentinel.config import GRAIN_PROFILES

MODELS_DIR = Path("data/models")


class GrainRiskPredictionEngine:
    """
    Dual Random Forest architecture:
      - Classifier: Categorizes state into SAFE, WARNING, CRITICAL
      - Regressor 1: Computes fine-grained Spoilage Risk Score (0–100%)
      - Regressor 2: Estimates Days-to-Risk-Threshold (Safe Shelf-Life)
    """

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            class_weight="balanced"
        )
        self.score_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            random_state=42
        )
        self.days_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = FEATURE_NAMES

    def train(self, dataset_path: str = "data/grain_dataset.csv") -> Dict[str, Any]:
        """Trains the Random Forest ensemble on the physics-informed dataset."""
        df = pd.read_csv(dataset_path)

        X = df[self.feature_names].values
        y_tier = df["target_risk_tier"].values
        y_score = df["target_risk_score"].values
        y_days = df["target_days_to_threshold"].values

        # Split train/test (80/20)
        X_train, X_test, y_tier_tr, y_tier_te, y_score_tr, y_score_te, y_days_tr, y_days_te = (
            train_test_split(
                X, y_tier, y_score, y_days,
                test_size=0.2, random_state=42, stratify=y_tier
            )
        )

        # 1. Train Classifier
        self.classifier.fit(X_train, y_tier_tr)
        tier_preds = self.classifier.predict(X_test)
        clf_metrics = classification_report(y_tier_te, tier_preds, output_dict=True)

        # 2. Train Score Regressor
        self.score_regressor.fit(X_train, y_score_tr)
        score_preds = self.score_regressor.predict(X_test)
        score_mae = mean_absolute_error(y_score_te, score_preds)
        score_r2 = r2_score(y_score_te, score_preds)

        # 3. Train Days-to-Threshold Regressor
        self.days_regressor.fit(X_train, y_days_tr)
        days_preds = self.days_regressor.predict(X_test)
        days_mae = mean_absolute_error(y_days_te, days_preds)
        days_r2 = r2_score(y_days_te, days_preds)

        self.is_trained = True

        # Calculate feature importances
        importances = self.classifier.feature_importances_
        feature_importance_dict = {
            name: round(float(imp), 4)
            for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        }

        # Save artifacts
        self.save()

        return {
            "classification_accuracy": clf_metrics["accuracy"],
            "macro_f1": clf_metrics["macro avg"]["f1-score"],
            "score_mae": round(score_mae, 3),
            "score_r2": round(score_r2, 3),
            "days_mae": round(days_mae, 3),
            "days_r2": round(days_r2, 3),
            "feature_importance": feature_importance_dict
        }

    def predict(
        self,
        reading: SensorReading,
        prev_reading: Optional[SensorReading] = None
    ) -> Dict[str, Any]:
        """
        Runs real-time inference on an incoming SensorReading.
        """
        if not self.is_trained:
            self.load()

        feat_dict = extract_features(reading, prev_reading=prev_reading)
        x_vec = features_to_vector(feat_dict).reshape(1, -1)

        # 1. Predict tier & class probability
        tier = self.classifier.predict(x_vec)[0]
        probs = self.classifier.predict_proba(x_vec)[0]
        tier_idx = list(self.classifier.classes_).index(tier)
        confidence = float(probs[tier_idx])

        # 2. Predict continuous risk score
        risk_score = float(self.score_regressor.predict(x_vec)[0])
        risk_score = max(min(risk_score, 100.0), 0.0)

        # 3. Predict days to threshold
        days_to_threshold = float(self.days_regressor.predict(x_vec)[0])
        days_to_threshold = max(days_to_threshold, 0.0)

        # 4. Determine domain physiological danger trigger
        profile = GRAIN_PROFILES.get(reading.grain_type, GRAIN_PROFILES["Rice"])
        triggers = []
        if reading.moisture_pct > profile["safe_moisture_pct"]:
            diff = reading.moisture_pct - profile["safe_moisture_pct"]
            triggers.append(f"Moisture exceeded safe limit by +{diff:.1f}%")
        if reading.temp_c > profile["danger_temp_c"]:
            diff = reading.temp_c - profile["danger_temp_c"]
            triggers.append(f"Temperature exceeded danger zone by +{diff:.1f}°C")
        if reading.gas_ppm > 800.0:
            triggers.append(f"Respiration gas surge ({reading.gas_ppm:.0f} ppm)")

        trigger_summary = "; ".join(triggers) if triggers else "Conditions within biological safe margins"

        return {
            "unit_id": reading.unit_id,
            "grain_type": reading.grain_type,
            "risk_score": round(risk_score, 1),
            "risk_tier": tier,
            "days_to_threshold": round(days_to_threshold, 1),
            "confidence": round(confidence, 3),
            "primary_risk": profile["primary_risk"],
            "risk_mechanism": profile["risk_mechanism"],
            "recommended_action": profile["recommended_action"],
            "triggers": triggers,
            "trigger_summary": trigger_summary,
            "features": feat_dict
        }

    def save(self, directory: Path = MODELS_DIR):
        """Serializes trained model estimators to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "score_regressor": self.score_regressor,
                "days_regressor": self.days_regressor,
                "feature_names": self.feature_names
            },
            directory / "risk_model.joblib"
        )
        print(f"Risk models saved to {directory / 'risk_model.joblib'}")

    def load(self, directory: Path = MODELS_DIR):
        """Loads serialized model estimators from disk."""
        path = directory / "risk_model.joblib"
        if not path.exists():
            print("Model checkpoint not found. Training new model...")
            self.train()
            return
        data = joblib.load(path)
        self.classifier = data["classifier"]
        self.score_regressor = data["score_regressor"]
        self.days_regressor = data["days_regressor"]
        self.feature_names = data["feature_names"]
        self.is_trained = True


if __name__ == "__main__":
    engine = GrainRiskPredictionEngine()
    metrics = engine.train()
    print("Training complete:")
    print(f"Accuracy: {metrics['classification_accuracy']:.4f}")
    print(f"F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Risk Score MAE: {metrics['score_mae']:.3f}%")
    print(f"Days-to-Threshold MAE: {metrics['days_mae']:.3f} days")
    print("\nTop 5 Influential Features:")
    for k, v in list(metrics["feature_importance"].items())[:5]:
        print(f"  - {k}: {v}")

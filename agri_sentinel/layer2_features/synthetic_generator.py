"""
Layer 2: Data and Feature Layer — Physics-Informed Synthetic Dataset Generator
Generates a comprehensive dataset grounded in published ICAR, FAO, and grain physiology literature.
Simulates biological spoilage trajectories across 5 grain types and 4 storage structures.
"""
import os
from pathlib import Path
from typing import Optional
import random
import numpy as np
import pandas as pd
from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.layer2_features.feature_pipeline import (
    extract_features,
    FEATURE_NAMES
)
from agri_sentinel.config import GRAIN_PROFILES, STORAGE_TYPES

DATA_DIR = Path("data")


def generate_grain_dataset(
    n_samples_per_grain: int = 1200,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a physics-informed synthetic dataset modeling grain storage physiology.
    
    Target labels generated:
        - 'risk_score': Continuous spoilage risk percentage (0.0 to 100.0)
        - 'risk_tier': Discrete classification ('SAFE', 'WARNING', 'CRITICAL')
        - 'days_to_threshold': Safe days remaining before irreversible spoilage (0.0 to 30.0)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    rows = []
    grain_types = list(GRAIN_PROFILES.keys())
    storage_types = list(STORAGE_TYPES.keys())

    for grain in grain_types:
        prof = GRAIN_PROFILES[grain]

        for _ in range(n_samples_per_grain):
            storage = random.choice(storage_types)
            s_mod = STORAGE_TYPES[storage]["risk_modifier"]

            # Decide regime: 45% safe, 35% warning (early danger), 20% critical (microbial bloom/pests)
            regime = random.choices(["safe", "warning", "critical"], weights=[0.45, 0.35, 0.20])[0]

            days_stored = random.randint(1, 90)

            if regime == "safe":
                temp = float(np.random.normal(prof["safe_temp_c"] - 2.0, 2.5))
                rh = float(np.random.normal(prof["safe_rh_pct"] - 5.0, 4.0))
                moist = float(np.random.normal(prof["safe_moisture_pct"] - 1.2, 0.6))
                gas = float(np.random.normal(440.0, 30.0))
                # High safe days remaining
                days_left = float(max(15.0 - (days_stored * 0.05) + np.random.normal(0, 2.0), 10.0))
                base_risk = random.uniform(5.0, 25.0)

            elif regime == "warning":
                temp = float(np.random.normal(prof["danger_temp_c"], 2.0))
                rh = float(np.random.normal(prof["danger_rh_pct"], 3.5))
                moist = float(np.random.normal(prof["danger_moisture_pct"], 0.7))
                # Microbial gas starts creeping up
                gas = float(np.random.normal(720.0, 80.0))
                days_left = float(max(prof["safe_days_at_danger"] + np.random.normal(0, 1.0), 1.5))
                base_risk = random.uniform(40.0, 68.0)

            else:  # critical
                temp = float(np.random.normal(prof["critical_temp_c"], 2.5))
                rh = float(np.random.normal(prof["danger_rh_pct"] + 6.0, 4.0))
                moist = float(np.random.normal(prof["critical_moisture_pct"], 1.0))
                # Intense respiration from mold/weevils
                gas = float(np.random.normal(1250.0, 200.0))
                days_left = float(max(np.random.exponential(scale=1.2), 0.1))
                base_risk = random.uniform(75.0, 98.0)

            # Clamp physically plausible ranges
            temp = max(min(temp, 48.0), 10.0)
            rh = max(min(rh, 99.0), 20.0)
            moist = max(min(moist, 32.0), 6.0)
            gas = max(gas, 380.0)
            days_left = min(days_left, 30.0)

            # Synthetic rate of change (previous day delta)
            if regime == "safe":
                d_temp = np.random.normal(0.0, 0.3)
                d_moist = np.random.normal(0.0, 0.1)
                d_gas = np.random.normal(0.0, 15.0)
            elif regime == "warning":
                d_temp = np.random.normal(0.4, 0.3)
                d_moist = np.random.normal(0.3, 0.15)
                d_gas = np.random.normal(35.0, 20.0)
            else:
                d_temp = np.random.normal(1.2, 0.5)
                d_moist = np.random.normal(0.7, 0.3)
                d_gas = np.random.normal(110.0, 45.0)

            # Build SensorReading
            reading = SensorReading(
                unit_id=f"UNIT-{random.randint(100, 999)}",
                grain_type=grain,
                storage_type=storage,
                temp_c=round(temp, 2),
                rh_pct=round(rh, 2),
                moisture_pct=round(moist, 2),
                gas_ppm=round(gas, 1),
                days_in_storage=days_stored
            )

            # Extract engineered features
            feat = extract_features(reading)
            # Override derivatives with realistic historical trajectory
            feat["d_temp_dt"] = float(d_temp)
            feat["d_moisture_dt"] = float(d_moist)
            feat["d_gas_dt"] = float(d_gas)

            # Physiological continuous risk score formula based on ICAR curves
            # Ground truth is driven by (moist_delta * temp_delta) interaction + gas surge
            interaction_risk = max(feat["moisture_delta_safe"], 0) * max(feat["temp_delta_safe"], 0) * 2.8
            gas_risk = max(feat["gas_surge_ratio"] - 1.2, 0) * 12.0
            computed_risk = base_risk * 0.5 + (interaction_risk + gas_risk) * 0.5 * s_mod
            computed_risk = float(max(min(computed_risk, 100.0), 2.0))

            # Assign ground truth tier
            if computed_risk < 35.0:
                tier = "SAFE"
            elif computed_risk < 70.0:
                tier = "WARNING"
            else:
                tier = "CRITICAL"

            feat["target_risk_score"] = round(computed_risk, 2)
            feat["target_risk_tier"] = tier
            feat["target_days_to_threshold"] = round(days_left, 2)
            feat["grain_type"] = grain
            feat["storage_type"] = storage

            rows.append(feat)

    df = pd.DataFrame(rows)

    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / "grain_dataset.csv"

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} physics-informed grain storage samples at {output_path}")
    return df


if __name__ == "__main__":
    generate_grain_dataset()

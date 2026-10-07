"""
Layer 2: Data and Feature Layer — Feature Engineering Pipeline
Computes domain-specific grain physiology features including the Temperature x Moisture
interaction term, Vapor Pressure Deficit (VPD), and physiological danger deltas.
"""
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.config import GRAIN_PROFILES, STORAGE_TYPES

GRAIN_TYPES_LIST = ["Maize", "Wheat", "Rice", "Pulses", "Oilseeds"]
STORAGE_TYPES_LIST = ["gunny_bag", "kothi", "bulk_room", "metallic_bin"]

FEATURE_NAMES = [
    "temp_c",
    "rh_pct",
    "moisture_pct",
    "gas_ppm",
    "days_in_storage",
    "temp_moisture_interaction",
    "moisture_delta_safe",
    "temp_delta_safe",
    "rh_delta_safe",
    "gas_surge_ratio",
    "vpd_kpa",
    "storage_risk_multiplier",
    # One-hot grain types
    "grain_is_Maize",
    "grain_is_Wheat",
    "grain_is_Rice",
    "grain_is_Pulses",
    "grain_is_Oilseeds",
    # One-hot storage types
    "storage_is_gunny_bag",
    "storage_is_kothi",
    "storage_is_bulk_room",
    "storage_is_metallic_bin",
    # Rate of change
    "d_temp_dt",
    "d_moisture_dt",
    "d_gas_dt"
]


def compute_vapor_pressure_deficit(temp_c: float, rh_pct: float) -> float:
    """
    Computes interstitial Vapor Pressure Deficit (VPD) in kPa.
    Low VPD with elevated temperature indicates high mold germination probability.
    """
    svp = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100.0)
    vpd = max(svp - avp, 0.0)
    return round(vpd, 4)


def extract_features(
    reading: SensorReading,
    prev_reading: Optional[SensorReading] = None
) -> Dict[str, float]:
    """
    Transforms a raw SensorReading into engineered features for the Risk Prediction Engine.
    """
    profile = GRAIN_PROFILES.get(reading.grain_type, GRAIN_PROFILES["Rice"])
    storage_info = STORAGE_TYPES.get(reading.storage_type, STORAGE_TYPES["gunny_bag"])

    # 1. Biological interaction term (Section 10.2 of proposal)
    t_m_interaction = reading.temp_c * reading.moisture_pct

    # 2. Physiological deltas from safe storage standards
    moist_delta = reading.moisture_pct - profile["safe_moisture_pct"]
    temp_delta = reading.temp_c - profile["safe_temp_c"]
    rh_delta = reading.rh_pct - profile["safe_rh_pct"]

    # 3. Respiration surge indicator (Baseline ambient is ~420 ppm)
    gas_surge_ratio = reading.gas_ppm / 420.0

    # 4. Vapor Pressure Deficit
    vpd = compute_vapor_pressure_deficit(reading.temp_c, reading.rh_pct)

    # 5. Temporal rates of change (derivatives)
    if prev_reading is not None:
        d_temp = reading.temp_c - prev_reading.temp_c
        d_moist = reading.moisture_pct - prev_reading.moisture_pct
        d_gas = reading.gas_ppm - prev_reading.gas_ppm
    else:
        d_temp = 0.0
        d_moist = 0.0
        d_gas = 0.0

    feats: Dict[str, float] = {
        "temp_c": reading.temp_c,
        "rh_pct": reading.rh_pct,
        "moisture_pct": reading.moisture_pct,
        "gas_ppm": reading.gas_ppm,
        "days_in_storage": float(reading.days_in_storage),
        "temp_moisture_interaction": t_m_interaction,
        "moisture_delta_safe": moist_delta,
        "temp_delta_safe": temp_delta,
        "rh_delta_safe": rh_delta,
        "gas_surge_ratio": gas_surge_ratio,
        "vpd_kpa": vpd,
        "storage_risk_multiplier": storage_info["risk_modifier"],
        "d_temp_dt": d_temp,
        "d_moisture_dt": d_moist,
        "d_gas_dt": d_gas
    }

    # One-hot encode grain type
    for g in GRAIN_TYPES_LIST:
        feats[f"grain_is_{g}"] = 1.0 if reading.grain_type == g else 0.0

    # One-hot encode storage type
    for s in STORAGE_TYPES_LIST:
        feats[f"storage_is_{s}"] = 1.0 if reading.storage_type == s else 0.0

    return feats


def features_to_vector(feat_dict: Dict[str, float]) -> np.ndarray:
    """Converts feature dictionary to a consistent numpy 1D vector."""
    return np.array([feat_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)


def dataframe_from_readings(readings: List[SensorReading]) -> pd.DataFrame:
    """Converts a sequence of SensorReadings into a feature DataFrame."""
    rows = []
    prev = None
    for r in readings:
        f = extract_features(r, prev_reading=prev)
        rows.append(f)
        prev = r
    df = pd.DataFrame(rows)
    # Ensure all feature columns exist in exact order
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_NAMES]

"""
Layer 1: Sensing Layer — Realistic IoT Sensor Stream Simulator
Simulates physical time-series telemetry from grain probes deployed in smallholder storage.
Supports diurnal cycles, moisture absorption, microbial respiration heating, and insect outbreaks.
"""
import math
import random
from typing import List, Generator
from datetime import datetime, timedelta
from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.config import GRAIN_PROFILES, STORAGE_TYPES


class IoTTelemetrySimulator:
    """
    Generates realistic, physically-grounded sensor streams for testing AgriSentinel.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_point(
        self,
        unit_id: str,
        grain_type: str,
        storage_type: str,
        scenario: str = "optimal_safe",
        day: int = 1,
        hour: int = 12
    ) -> SensorReading:
        """
        Generates a single telemetry reading based on grain physiology and scenario.
        
        Scenarios:
            - 'optimal_safe': Storage within safe equilibrium, low respiration
            - 'moisture_ingress': Damp floor / rain leak raising moisture & humidity
            - 'heating_hotspot': Fungal respiration driving temperature and CO2 spikes
            - 'insect_infestation': Weevil boring driving gas ppm and micro-heating
        """
        profile = GRAIN_PROFILES.get(grain_type, GRAIN_PROFILES["Rice"])
        storage_mod = STORAGE_TYPES.get(storage_type, STORAGE_TYPES["gunny_bag"])["risk_modifier"]

        # Diurnal thermal wave (interstitial grain buffers external diurnal swings by ~70%)
        diurnal_temp = 1.8 * math.sin((hour - 8) * math.pi / 12.0)
        diurnal_rh = -2.5 * math.sin((hour - 8) * math.pi / 12.0)

        if scenario == "optimal_safe":
            temp = profile["safe_temp_c"] - 1.5 + diurnal_temp + random.uniform(-0.3, 0.3)
            rh = profile["safe_rh_pct"] - 3.0 + diurnal_rh + random.uniform(-1.0, 1.0)
            moisture = profile["safe_moisture_pct"] - 0.8 + random.uniform(-0.15, 0.15)
            gas = 420.0 + random.uniform(10, 40)

        elif scenario == "moisture_ingress":
            # Progressive moisture creeping over days
            progression = min(day / 15.0, 1.0)
            target_moist = profile["danger_moisture_pct"] + 1.2 * progression * storage_mod
            target_rh = profile["danger_rh_pct"] + 4.0 * progression
            moisture = (profile["safe_moisture_pct"] * (1 - progression)) + (target_moist * progression)
            rh = (profile["safe_rh_pct"] * (1 - progression)) + (target_rh * progression) + diurnal_rh
            # Fungal respiration starts warming grain as moisture increases
            temp = profile["safe_temp_c"] + (3.5 * progression) + diurnal_temp
            gas = 450.0 + (500.0 * progression) + random.uniform(0, 50)

        elif scenario == "heating_hotspot":
            # Active fungal bloom: high temp, high gas, high moisture
            progression = min(day / 8.0, 1.0)
            moisture = profile["critical_moisture_pct"] - 0.2 + random.uniform(-0.2, 0.3)
            rh = profile["danger_rh_pct"] + 5.0 + diurnal_rh
            temp = profile["danger_temp_c"] + (5.0 * progression) + diurnal_temp
            gas = 900.0 + (750.0 * progression) + random.uniform(-30, 80)

        elif scenario == "insect_infestation":
            # Fast CO2 surge, moderate moisture, localized biological heat
            progression = min(day / 10.0, 1.0)
            moisture = profile["danger_moisture_pct"] - 0.2 + random.uniform(-0.2, 0.2)
            rh = profile["danger_rh_pct"] - 2.0 + diurnal_rh
            temp = profile["danger_temp_c"] + (2.5 * progression)
            gas = 850.0 + (1100.0 * progression) + random.uniform(0, 100)

        else:
            temp = 25.0 + diurnal_temp
            rh = 65.0 + diurnal_rh
            moisture = 13.0
            gas = 450.0

        # Battery gradual drain over days
        battery = max(100.0 - (day * 0.4), 10.0)

        return SensorReading(
            unit_id=unit_id,
            grain_type=grain_type,
            storage_type=storage_type,
            temp_c=round(temp, 2),
            rh_pct=round(max(min(rh, 98.0), 10.0), 2),
            moisture_pct=round(max(min(moisture, 35.0), 6.0), 2),
            gas_ppm=round(max(gas, 380.0), 1),
            days_in_storage=int(day),
            battery_pct=round(battery, 1)
        )

    def generate_timeseries(
        self,
        unit_id: str,
        grain_type: str,
        storage_type: str,
        scenario: str,
        total_days: int = 14,
        readings_per_day: int = 4
    ) -> List[SensorReading]:
        """Generates a complete multi-day time-series trajectory."""
        readings = []
        base_time = datetime.now() - timedelta(days=total_days)

        for day in range(1, total_days + 1):
            for step in range(readings_per_day):
                hour = int(step * (24 / readings_per_day))
                reading = self.generate_point(
                    unit_id=unit_id,
                    grain_type=grain_type,
                    storage_type=storage_type,
                    scenario=scenario,
                    day=day,
                    hour=hour
                )
                ts = base_time + timedelta(days=day - 1, hours=hour)
                reading.timestamp = ts.isoformat()
                readings.append(reading)

        return readings

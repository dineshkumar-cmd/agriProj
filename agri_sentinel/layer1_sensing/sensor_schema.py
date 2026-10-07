"""
Layer 1: Sensing Layer — Sensor Schema & Telemetry Validation
Defines data structures for incoming IoT readings from the smallholder probe.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class SensorReading:
    """
    Standard IoT sensor reading packet from an AgriSentinel storage probe.
    """
    unit_id: str
    grain_type: str                  # Maize, Wheat, Rice, Pulses, Oilseeds
    storage_type: str                # gunny_bag, kothi, bulk_room, metallic_bin
    temp_c: float                    # Interstitial temperature in °C
    rh_pct: float                    # Interstitial relative humidity %
    moisture_pct: float              # Direct grain moisture content %
    gas_ppm: float                   # MQ135 CO2/VOC reading in ppm
    days_in_storage: int             # Total days stored since harvest/bagging
    battery_pct: float = 95.0        # Probe battery status %
    timestamp: Optional[str] = None  # ISO timestamp

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        self.validate()

    def validate(self):
        """Ensures physical plausibility of telemetry readings."""
        # Temperature clamp/check
        if not (-5.0 <= self.temp_c <= 65.0):
            raise ValueError(f"Temperature reading {self.temp_c}°C out of physical range (-5 to 65°C)")
        # Relative humidity clamp
        if not (5.0 <= self.rh_pct <= 100.0):
            raise ValueError(f"Relative humidity {self.rh_pct}% out of physical range (5 to 100%)")
        # Grain moisture clamp
        if not (4.0 <= self.moisture_pct <= 40.0):
            raise ValueError(f"Grain moisture {self.moisture_pct}% out of physical range (4 to 40%)")
        # Gas clamp
        if not (200.0 <= self.gas_ppm <= 6000.0):
            raise ValueError(f"Gas level {self.gas_ppm} ppm out of physical range (200 to 6000 ppm)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sanitize_reading(raw: Dict[str, Any]) -> SensorReading:
    """Safely coerces raw dict/JSON payloads into a validated SensorReading."""
    return SensorReading(
        unit_id=str(raw.get("unit_id", "UNIT-001")),
        grain_type=str(raw.get("grain_type", "Rice")),
        storage_type=str(raw.get("storage_type", "gunny_bag")),
        temp_c=float(max(min(float(raw.get("temp_c", 25.0)), 65.0), -5.0)),
        rh_pct=float(max(min(float(raw.get("rh_pct", 65.0)), 100.0), 5.0)),
        moisture_pct=float(max(min(float(raw.get("moisture_pct", 12.5)), 40.0), 4.0)),
        gas_ppm=float(max(min(float(raw.get("gas_ppm", 450.0)), 6000.0), 200.0)),
        days_in_storage=int(max(int(raw.get("days_in_storage", 1)), 0)),
        battery_pct=float(max(min(float(raw.get("battery_pct", 100.0)), 100.0), 0.0)),
        timestamp=raw.get("timestamp")
    )

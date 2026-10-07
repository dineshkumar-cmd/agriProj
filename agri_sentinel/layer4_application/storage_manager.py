"""
Layer 4: Application Layer — Multi-Unit Storage Facility Manager
Tracks multiple storage units (Gunny bags, Kothi, Bins) for smallholder farmers and FPO cooperatives.
Maintains traffic-light status cards and historical telemetry.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.layer1_sensing.mock_iot_stream import IoTTelemetrySimulator
from agri_sentinel.layer3_intelligence.fusion_engine import AgriSentinelFusionEngine


@dataclass
class StorageUnit:
    """Represents a monitored smallholder grain storage unit."""
    unit_id: str
    farmer_name: str
    phone: str
    village: str
    grain_type: str                   # Maize, Wheat, Rice, Pulses, Oilseeds
    storage_type: str                 # gunny_bag, kothi, bulk_room, metallic_bin
    quantity_quintals: float
    days_in_storage: int
    latest_reading: Optional[SensorReading] = None
    latest_fusion: Optional[Dict[str, Any]] = None
    history: List[SensorReading] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.latest_fusion:
            return self.latest_fusion.get("severity_tier", "SAFE")
        return "SAFE"

    @property
    def status_color(self) -> str:
        st = self.status
        if st == "CRITICAL":
            return "#EF4444"  # Red
        elif st == "WARNING":
            return "#F59E0B"  # Amber/Orange
        return "#10B981"      # Emerald Green


class StorageFacilityManager:
    """
    Manages an inventory of monitored storage units across a village cooperative or FPO.
    """

    def __init__(self, fusion_engine: Optional[AgriSentinelFusionEngine] = None):
        self.fusion_engine = fusion_engine or AgriSentinelFusionEngine()
        self.units: Dict[str, StorageUnit] = {}
        self._init_sample_units()

    def _init_sample_units(self):
        """Initializes realistic baseline storage units across Indian smallholder realities."""
        sim = IoTTelemetrySimulator(seed=101)

        configs = [
            ("UNIT-101", "Ramesh Patel", "+91-9823412001", "Dharwad, Karnataka", "Maize", "gunny_bag", 35.0, 14, "moisture_ingress"),
            ("UNIT-102", "Sunita Devi", "+91-9711245002", "Varanasi, UP", "Wheat", "kothi", 25.0, 32, "optimal_safe"),
            ("UNIT-103", "Murugan K.", "+91-9443218003", "Thanjavur, TN", "Rice", "gunny_bag", 50.0, 18, "optimal_safe"),
            ("UNIT-104", "Balwan Singh", "+91-9876523004", "Hisar, Haryana", "Pulses", "metallic_bin", 20.0, 22, "insect_infestation"),
            ("UNIT-105", "Gowri Bai", "+91-9988776605", "Anantapur, AP", "Oilseeds", "bulk_room", 30.0, 9, "optimal_safe"),
        ]

        for uid, farmer, phone, village, grain, st_type, qty, days, scenario in configs:
            reading = sim.generate_point(
                unit_id=uid,
                grain_type=grain,
                storage_type=st_type,
                scenario=scenario,
                day=days,
                hour=14
            )
            # Run initial fusion
            fusion_res = self.fusion_engine.fuse(reading)
            unit = StorageUnit(
                unit_id=uid,
                farmer_name=farmer,
                phone=phone,
                village=village,
                grain_type=grain,
                storage_type=st_type,
                quantity_quintals=qty,
                days_in_storage=days,
                latest_reading=reading,
                latest_fusion=fusion_res,
                history=[reading]
            )
            self.units[uid] = unit

    def get_unit(self, unit_id: str) -> Optional[StorageUnit]:
        return self.units.get(unit_id)

    def list_units(self) -> List[StorageUnit]:
        return list(self.units.values())

    def update_unit_telemetry(
        self,
        unit_id: str,
        reading: SensorReading,
        image_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Ingests a new telemetry packet and updates storage state."""
        unit = self.units.get(unit_id)
        if not unit:
            raise ValueError(f"Storage unit {unit_id} not found")

        prev = unit.latest_reading
        fusion_res = self.fusion_engine.fuse(reading, image_input=image_input, prev_reading=prev)

        unit.latest_reading = reading
        unit.latest_fusion = fusion_res
        unit.history.append(reading)
        if len(unit.history) > 100:
            unit.history.pop(0)

        return fusion_res

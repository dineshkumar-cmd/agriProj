"""
Layer 1: Sensing Layer — Hardware Specifications & BOM
Details the smallholder IoT probe hardware architecture, component specifications,
and cost justification proving the ₹500–800 price point.
"""
from typing import Dict, List, Any
from agri_sentinel.config import HARDWARE_BOM, TOTAL_HARDWARE_COST_INR


def get_bom_summary() -> Dict[str, Any]:
    """Returns the Bill of Materials breakdown and comparative cost analysis."""
    return {
        "target_audience": "Indian Smallholder Farmers (Gunny bags, Kothi, Bins)",
        "total_cost_inr": TOTAL_HARDWARE_COST_INR,
        "target_cost_range_inr": "₹500 – ₹800",
        "is_within_budget": 500 <= TOTAL_HARDWARE_COST_INR <= 800,
        "comparison_with_silo_systems": {
            "commercial_silo_cable_node_inr": "₹15,000 – ₹45,000 per cable point",
            "agrisentinel_inr": f"₹{TOTAL_HARDWARE_COST_INR}",
            "cost_reduction_ratio": "18x to 50x lower cost",
        },
        "components": HARDWARE_BOM,
        "circuit_architecture": {
            "microcontroller": "ESP32 (NodeMCU / ESP-WROOM-32) with Deep Sleep support",
            "temp_humidity_bus": "GPIO 4 (Single-wire digital DHT22 / I2C SHT30)",
            "capacitive_moisture_pin": "ADC1 / GPIO 34 (Analog reading with calibration curve)",
            "gas_sensor_pin": "ADC1 / GPIO 35 (Analog reading after 24h burn-in calibration)",
            "power_draw": "Peak 160mA (WiFi TX for 2s), Deep Sleep 15µA (sleeps 58 minutes every hour)",
            "battery_endurance_days": "Approximately 45–60 days on a single 2200mAh 18650 cell"
        }
    }

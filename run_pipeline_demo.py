"""
AgriSentinel: End-to-End CLI Pipeline Demonstration
Executes all four processing layers sequentially:
  - Layer 1: Sensing Layer (Telemetry Ingestion & Hardware Verification)
  - Layer 2: Feature Layer (Grain Physiology & Interaction Engineering)
  - Layer 3: Intelligence Layer (Random Forest Prediction + MobileNet Vision + Multimodal Fusion)
  - Layer 4: Application Layer (Regional Multi-Lingual Alerts)
"""
import sys
from pathlib import Path
from pprint import pprint

# Ensure Windows PowerShell/cmd console handles Unicode and Indic scripts gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.layer1_sensing.mock_iot_stream import IoTTelemetrySimulator
from agri_sentinel.layer1_sensing.hardware_specs import get_bom_summary
from agri_sentinel.layer2_features.feature_pipeline import extract_features
from agri_sentinel.layer3_intelligence.risk_model import GrainRiskPredictionEngine
from agri_sentinel.layer3_intelligence.vision_model import GrainVisualClassifier, generate_benchmark_grain_images
from agri_sentinel.layer3_intelligence.fusion_engine import AgriSentinelFusionEngine
from agri_sentinel.layer4_application.alert_dispatcher import RegionalAlertDispatcher


def main():
    print("=" * 80)
    print("🌾 AGRISENTINEL: AI-POWERED POSTHARVEST GRAIN SPOILAGE EARLY-WARNING SYSTEM")
    print("   Academic Year 2026 — B.Tech AIML Final Year Project")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # LAYER 1: SENSING LAYER
    # -------------------------------------------------------------------------
    print("\n" + "—" * 80)
    print(">>> LAYER 1: SENSING LAYER (Smallholder IoT Sensing & Hardware BOM)")
    print("—" * 80)
    bom = get_bom_summary()
    print(f"Target Hardware Budget: {bom['target_cost_range_inr']}")
    print(f"Calculated BOM Total:  ₹{bom['total_cost_inr']} (Budget compliance: {bom['is_within_budget']})")
    print(f"Smallholder vs Silo Cost Advantage: {bom['comparison_with_silo_systems']['cost_reduction_ratio']}")

    sim = IoTTelemetrySimulator(seed=42)
    # Simulate a smallholder maize gunny bag experiencing monsoon moisture seepage
    reading = sim.generate_point(
        unit_id="BAG-MZ-44",
        grain_type="Maize",
        storage_type="gunny_bag",
        scenario="moisture_ingress",
        day=12,
        hour=15
    )
    print(f"\n[Layer 1 Live Ingest] Unit: {reading.unit_id} ({reading.grain_type} in {reading.storage_type})")
    print(f"  - Temperature:     {reading.temp_c} °C")
    print(f"  - Rel Humidity:    {reading.rh_pct} %")
    print(f"  - Grain Moisture:  {reading.moisture_pct} % (Danger threshold: >15.5%)")
    print(f"  - Respiration Gas: {reading.gas_ppm} ppm")
    print(f"  - Days in Storage: {reading.days_in_storage} days")
    print(f"  - Probe Battery:   {reading.battery_pct} %")

    # -------------------------------------------------------------------------
    # LAYER 2: FEATURE ENGINEERING LAYER
    # -------------------------------------------------------------------------
    print("\n" + "—" * 80)
    print(">>> LAYER 2: DATA & FEATURE ENGINEERING LAYER")
    print("—" * 80)
    features = extract_features(reading)
    print(f"Computed Physiological Interaction Features:")
    print(f"  - Temp x Moisture Interaction Term: {features['temp_moisture_interaction']:.2f}")
    print(f"  - Moisture Delta from Safe Limit:   +{features['moisture_delta_safe']:.2f}%")
    print(f"  - Temp Delta from Safe Limit:       +{features['temp_delta_safe']:.2f}°C")
    print(f"  - Gas Respiration Surge Ratio:      {features['gas_surge_ratio']:.2f}x baseline")
    print(f"  - Vapor Pressure Deficit (VPD):     {features['vpd_kpa']:.4f} kPa")

    # -------------------------------------------------------------------------
    # LAYER 3: INTELLIGENCE LAYER (Risk Model + Vision + Fusion)
    # -------------------------------------------------------------------------
    print("\n" + "—" * 80)
    print(">>> LAYER 3: INTELLIGENCE LAYER")
    print("—" * 80)

    # 1. Train or Load Risk Engine
    risk_engine = GrainRiskPredictionEngine()
    risk_engine.load()
    sensor_pred = risk_engine.predict(reading)
    print(f"[A. Random Forest Time-Series Risk Engine]:")
    print(f"  - Predicted Spoilage Risk Score: {sensor_pred['risk_score']}%")
    print(f"  - Categorical Risk Tier:        {sensor_pred['risk_tier']}")
    print(f"  - Estimated Days-to-Threshold:   {sensor_pred['days_to_threshold']} safe days remaining")
    print(f"  - Confidence:                   {sensor_pred['confidence'] * 100:.1f}%")
    print(f"  - Active Biological Hazard:     {sensor_pred['primary_risk']}")
    print(f"  - Danger Trigger:               {sensor_pred['trigger_summary']}")

    # 2. Visual Classifier
    generate_benchmark_grain_images()
    vision_clf = GrainVisualClassifier()
    sample_img_path = Path("data/sample_images/healthy_grain.jpg")
    vis_pred = vision_clf.classify(sample_img_path)
    print(f"\n[B. Lightweight MobileNetV2 Smartphone Vision Classifier]:")
    print(f"  - Inspected Image:        {sample_img_path.name}")
    print(f"  - Visual Appearance:      {vis_pred['condition']}")
    print(f"  - Visual Confidence:      {vis_pred['confidence'] * 100:.1f}%")
    print(f"  - Visual Defect Severity: {vis_pred['visual_risk_score']}%")

    # 3. Multimodal Fusion Engine (Early-Warning Window Demonstration)
    fusion_engine = AgriSentinelFusionEngine(risk_engine=risk_engine, vision_classifier=vision_clf)
    fusion_res = fusion_engine.fuse(reading, image_input=sample_img_path)
    print(f"\n[C. Multimodal Sensor-Vision Fusion Engine]:")
    print(f"  - Fused Risk Index:       {fusion_res['fused_risk_score']}%")
    print(f"  - Final Severity Tier:    {fusion_res['severity_tier']}")
    print(f"  - Multimodal State:       {fusion_res['fusion_state']}")
    print(f"  - Early Warning Trigger:  {'🚨 YES — PRE-VISUAL EARLY WARNING DETECTED!' if fusion_res['is_early_warning'] else 'No'}")
    print(f"  - Fusion Insight:         {fusion_res['fusion_insight']}")

    # -------------------------------------------------------------------------
    # LAYER 4: APPLICATION & REGIONAL ALERT LAYER
    # -------------------------------------------------------------------------
    print("\n" + "—" * 80)
    print(">>> LAYER 4: APPLICATION & REGIONAL ALERT LAYER")
    print("—" * 80)
    dispatcher = RegionalAlertDispatcher()
    all_alerts = dispatcher.format_all_languages(fusion_res)

    for lang_code in ["en", "hi", "ta", "te", "kn"]:
        alert = all_alerts[lang_code]
        print(f"\n[{alert['language_name'].upper()}] — Urgency: {alert['urgency']}")
        if alert['badge']:
            print(f"  🏷️ Tag: {alert['badge']}")
        print(f"  📌 Title: {alert['title']}")
        print(f"  💬 Body:  {alert['message_body']}")

    print("\n" + "=" * 80)
    print("✅ END-TO-END PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 80)


if __name__ == "__main__":
    main()

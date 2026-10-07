"""
Layer 3: Intelligence Layer — Multimodal Sensor-Vision Fusion Engine
Implements the core early-warning value proposition:
  - Detects pre-visual risk when sensor risk rises while grain appearance remains healthy
  - Escalates to critical when both IoT sensors and phone camera confirm spoilage
  - Suppresses false alerts when visual dust/chaff occurs in dry, safe storage
"""
from typing import Dict, Any, Optional
from agri_sentinel.layer1_sensing.sensor_schema import SensorReading
from agri_sentinel.layer3_intelligence.risk_model import GrainRiskPredictionEngine
from agri_sentinel.layer3_intelligence.vision_model import GrainVisualClassifier


class AgriSentinelFusionEngine:
    """
    Fuses continuous time-series IoT risk with episodic smartphone camera confirmation.
    """

    def __init__(
        self,
        risk_engine: Optional[GrainRiskPredictionEngine] = None,
        vision_classifier: Optional[GrainVisualClassifier] = None
    ):
        self.risk_engine = risk_engine or GrainRiskPredictionEngine()
        self.vision_classifier = vision_classifier or GrainVisualClassifier()

    def fuse(
        self,
        sensor_reading: SensorReading,
        image_input: Optional[Any] = None,
        prev_reading: Optional[SensorReading] = None
    ) -> Dict[str, Any]:
        """
        Executes multimodal fusion over IoT sensor reading and optional photograph.
        """
        # 1. IoT Sensor Risk Prediction
        sensor_result = self.risk_engine.predict(sensor_reading, prev_reading=prev_reading)
        sensor_score = sensor_result["risk_score"]  # 0 to 100
        sensor_tier = sensor_result["risk_tier"]

        # 2. Visual Classifier Assessment (if photo provided)
        if image_input is not None:
            vision_result = self.vision_classifier.classify(image_input)
            has_photo = True
        else:
            vision_result = {
                "condition": "No Photo Submitted",
                "confidence": 0.0,
                "visual_risk_score": None,
                "is_damaged": False
            }
            has_photo = False

        vis_score = vision_result.get("visual_risk_score")
        vis_condition = vision_result.get("condition")

        # 3. Multimodal Decision Matrix
        is_early_warning = False
        is_dual_confirmed = False
        is_surface_anomaly = False

        if has_photo:
            # Case A: Pre-visual Early-Warning Window (The Hero Feature)
            if sensor_score >= 45.0 and vis_condition == "Healthy Grain":
                is_early_warning = True
                fusion_state = "PRE_VISUAL_EARLY_WARNING"
                fused_risk_score = sensor_score * 0.85 + 10.0  # Retain high urgency based on sensors
                severity_tier = "WARNING" if fused_risk_score < 75.0 else "CRITICAL"
                fusion_insight = (
                    "CRITICAL OPPORTUNITY: Rising moisture/temperature detected by sensors before damage "
                    "is visible to the human eye! Aerate or dry now to prevent visible mold."
                )

            # Case B: Dual Confirmation (Severe outbreak)
            elif sensor_score >= 45.0 and vis_condition in ["Visible Mold", "Insect Infestation"]:
                is_dual_confirmed = True
                fusion_state = "DUAL_CONFIRMED_HIGH_RISK"
                fused_risk_score = max(sensor_score, vis_score) * 0.7 + min(sensor_score, vis_score) * 0.3
                severity_tier = "CRITICAL"
                fusion_insight = (
                    "DUAL CONFIRMATION: Both internal IoT sensors and phone camera confirm active spoilage. "
                    "Segregate this storage unit immediately."
                )

            # Case C: Surface Anomaly / False Alarm Suppression
            elif sensor_score < 30.0 and vis_condition in ["Early Discoloration", "Visible Mold"]:
                is_surface_anomaly = True
                fusion_state = "SURFACE_ANOMALY_OR_FALSE_ALARM"
                # Moderate the score so farmer inspects sample without panic
                fused_risk_score = 35.0
                severity_tier = "SAFE"
                fusion_insight = (
                    "MONITOR ONLY: Photo flagged discoloration, but storage probe confirms cold, dry, "
                    "low-gas conditions. Likely benign dust/chaff rather than active fungal colony."
                )

            # Case D: Both Agree Safe
            elif sensor_score < 35.0 and vis_condition == "Healthy Grain":
                fusion_state = "STABLE_OPTIMAL"
                fused_risk_score = sensor_score * 0.9
                severity_tier = "SAFE"
                fusion_insight = "All storage conditions optimal. Grain quality preserved."

            else:
                # Weighted blend
                fusion_state = "MODERATE_RISK"
                fused_risk_score = (sensor_score * 0.6) + (vis_score * 0.4)
                severity_tier = "WARNING" if fused_risk_score >= 40.0 else "SAFE"
                fusion_insight = "Moderate conditions detected. Continue periodic monitoring."
        else:
            # Sensor-only mode
            fusion_state = "SENSOR_TELEMETRY_ONLY"
            fused_risk_score = sensor_score
            severity_tier = sensor_tier
            fusion_insight = (
                f"Evaluation based on live sensor probe alone ({sensor_tier}). "
                "Submit a smartphone camera photo for multimodal confirmation."
            )

        fused_risk_score = round(float(max(min(fused_risk_score, 100.0), 0.0)), 1)

        return {
            "unit_id": sensor_reading.unit_id,
            "grain_type": sensor_reading.grain_type,
            "storage_type": sensor_reading.storage_type,
            "fused_risk_score": fused_risk_score,
            "severity_tier": severity_tier,
            "fusion_state": fusion_state,
            "fusion_insight": fusion_insight,
            "is_early_warning": is_early_warning,
            "is_dual_confirmed": is_dual_confirmed,
            "is_surface_anomaly": is_surface_anomaly,
            "sensor_result": sensor_result,
            "vision_result": vision_result,
            "estimated_days_safe": sensor_result["days_to_threshold"],
            "recommended_action": sensor_result["recommended_action"]
        }

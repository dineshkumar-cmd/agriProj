"""
Layer 4: Application Layer — Multi-Lingual Regional Alert Dispatcher
Translates raw machine learning risk outputs into actionable advice for smallholder farmers.
Supports: English, Hindi, Tamil, Telugu, and Kannada.
"""
from typing import Dict, Any, List
from datetime import datetime
from agri_sentinel.config import ALERT_TEMPLATES, GRAIN_PROFILES


class RegionalAlertDispatcher:
    """
    Generates actionable, regional-language alerts for SMS, WhatsApp, and Mobile App notifications.
    """

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "हिंदी (Hindi)",
        "ta": "தமிழ் (Tamil)",
        "te": "తెలుగు (Telugu)",
        "kn": "ಕನ್ನಡ (Kannada)"
    }

    def __init__(self, default_lang: str = "en"):
        self.default_lang = default_lang

    def format_alert(
        self,
        fusion_output: Dict[str, Any],
        lang: str = "en",
        farmer_phone: str = "+91-9876543210"
    ) -> Dict[str, Any]:
        """
        Constructs a localized alert packet from a fusion engine inference result.
        """
        lang = lang.lower()
        if lang not in ALERT_TEMPLATES:
            lang = self.default_lang

        tmpl = ALERT_TEMPLATES[lang]
        severity = fusion_output.get("severity_tier", "SAFE")
        grain_type = fusion_output.get("grain_type", "Rice")
        unit_id = fusion_output.get("unit_id", "UNIT-01")
        fused_score = fusion_output.get("fused_risk_score", 10.0)
        safe_days = fusion_output.get("estimated_days_safe", 14.0)

        sensor_res = fusion_output.get("sensor_result", {})
        feat = sensor_res.get("features", {})
        temp = feat.get("temp_c", 25.0)
        moist = feat.get("moisture_pct", 12.0)
        gas_ppm = feat.get("gas_ppm", 420.0)
        risk_type = sensor_res.get("primary_risk", "Fungal Spoilage")
        action = fusion_output.get("recommended_action", "Aerate and monitor daily.")

        # Determine alert template tier
        if severity == "CRITICAL":
            title = tmpl["critical_title"]
            body = tmpl["critical_body"].format(
                unit_id=unit_id,
                grain_type=grain_type,
                risk_score=fused_score,
                gas_ppm=f"{gas_ppm:.0f}",
                action=action
            )
            urgency = "HIGH"
        elif severity == "WARNING":
            title = tmpl["warning_title"]
            body = tmpl["warning_body"].format(
                unit_id=unit_id,
                grain_type=grain_type,
                risk_score=fused_score,
                risk_type=risk_type,
                action=action
            )
            urgency = "MEDIUM"
        else:
            title = tmpl["safe_title"]
            body = tmpl["safe_body"].format(
                unit_id=unit_id,
                grain_type=grain_type,
                moisture=f"{moist:.1f}",
                temp=f"{temp:.1f}",
                safe_days=f"{safe_days:.0f}"
            )
            urgency = "LOW"

        # Badge tag for multimodal context
        badge = None
        if fusion_output.get("is_early_warning"):
            badge = tmpl["early_window_badge"]
        elif fusion_output.get("is_dual_confirmed"):
            badge = tmpl["dual_confirm_badge"]

        return {
            "recipient": farmer_phone,
            "language": lang,
            "language_name": self.SUPPORTED_LANGUAGES.get(lang, "English"),
            "severity": severity,
            "urgency": urgency,
            "title": title,
            "badge": badge,
            "message_body": body,
            "recommended_action": action,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sms_payload": f"{title}\n{body}",
            "is_sent": True
        }

    def format_all_languages(self, fusion_output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generates the alert concurrently across all 5 regional languages for multi-lingual preview."""
        return {
            code: self.format_alert(fusion_output, lang=code)
            for code in self.SUPPORTED_LANGUAGES.keys()
        }

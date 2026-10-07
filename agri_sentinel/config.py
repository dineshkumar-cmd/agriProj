"""
AgriSentinel Configuration & Domain Science Specifications
Encodes grain-specific biological spoilage thresholds, multi-lingual alert templates,
hardware bill-of-materials (BOM), and storage unit definitions.
"""
from typing import Dict, Any

# ==============================================================================
# 1. GRAIN STORAGE PHYSIOLOGY THRESHOLDS (Table 10.1 from Proposal)
# Grounded in ICAR-CIPHET, FAO, and published post-harvest grain research.
# ==============================================================================
GRAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "Maize": {
        "display_name": "Maize / Corn",
        "safe_moisture_pct": 13.5,
        "danger_moisture_pct": 15.5,
        "critical_moisture_pct": 17.5,
        "safe_temp_c": 24.0,
        "danger_temp_c": 28.0,
        "critical_temp_c": 33.0,
        "safe_rh_pct": 65.0,
        "danger_rh_pct": 75.0,
        "primary_risk": "Aflatoxin (Aspergillus flavus) production",
        "risk_mechanism": "High moisture combined with warm air accelerates aflatoxin B1 accumulation",
        "safe_days_at_danger": 4,   # Days before irreversible fungal infestation
        "recommended_action": "Spread in thin layers and sun-dry to below 13.5% moisture immediately."
    },
    "Wheat": {
        "display_name": "Wheat",
        "safe_moisture_pct": 13.0,
        "danger_moisture_pct": 14.5,
        "critical_moisture_pct": 16.5,
        "safe_temp_c": 20.0,
        "danger_temp_c": 25.0,
        "critical_temp_c": 30.0,
        "safe_rh_pct": 65.0,
        "danger_rh_pct": 72.0,
        "primary_risk": "Aspergillus shift & storage mold",
        "risk_mechanism": "Above 25°C, microflora shifts toward Aspergillus dominance with rapid germination loss",
        "safe_days_at_danger": 5,
        "recommended_action": "Aerate storage bin immediately; move upper layers to cooler, ventilated area."
    },
    "Rice": {
        "display_name": "Paddy / Rice",
        "safe_moisture_pct": 13.0,
        "danger_moisture_pct": 14.5,
        "critical_moisture_pct": 16.0,
        "safe_temp_c": 25.0,
        "danger_temp_c": 30.0,
        "critical_temp_c": 35.0,
        "safe_rh_pct": 70.0,
        "danger_rh_pct": 78.0,
        "primary_risk": "Grain mildew & yellowing (time-progressive)",
        "risk_mechanism": "Progressive fungal respiration causing localized heating and yellowing of kernels",
        "safe_days_at_danger": 6,
        "recommended_action": "Rotate paddy stack, check gunny bag bases for floor dampness, dry immediately."
    },
    "Pulses": {
        "display_name": "Pulses (Chickpea / Lentil / Pigeonpea)",
        "safe_moisture_pct": 11.5,
        "danger_moisture_pct": 13.0,
        "critical_moisture_pct": 15.0,
        "safe_temp_c": 24.0,
        "danger_temp_c": 28.0,
        "critical_temp_c": 32.0,
        "safe_rh_pct": 60.0,
        "danger_rh_pct": 70.0,
        "primary_risk": "Pulse beetle (Callosobruchus) infestation + mold",
        "risk_mechanism": "Moisture above 12% triggers rapid beetle egg oviposition and larval boring",
        "safe_days_at_danger": 3,
        "recommended_action": "Sun-dry seed to <11% moisture; mix dried neem leaves or insect trap in bag."
    },
    "Oilseeds": {
        "display_name": "Oilseeds (Groundnut / Mustard / Soybean)",
        "safe_moisture_pct": 8.5,
        "danger_moisture_pct": 10.0,
        "critical_moisture_pct": 12.0,
        "safe_temp_c": 22.0,
        "danger_temp_c": 27.0,
        "critical_temp_c": 32.0,
        "safe_rh_pct": 55.0,
        "danger_rh_pct": 65.0,
        "primary_risk": "Lipid peroxidation, rancidity & Aspergillus mold",
        "risk_mechanism": "High oil content leads to enzymatic rancidity and rapid mold colonization at low moisture thresholds",
        "safe_days_at_danger": 3,
        "recommended_action": "Avoid stacking bags over 4 layers high; ventilate room and keep off bare ground."
    }
}

# ==============================================================================
# 2. MICROBIAL GAS & RESPIRATION THRESHOLDS (MQ135 CO2 / VOC)
# Normal air is ~420 ppm. Stored grain microbial activity produces CO2 and VOCs.
# ==============================================================================
GAS_THRESHOLDS = {
    "normal_max_ppm": 600.0,       # Healthy quiescent grain respiration
    "elevated_ppm": 900.0,         # Early fungal colony respiration / insect activity
    "critical_ppm": 1400.0         # Severe microbial bloom / intense infestation hotspot
}

# ==============================================================================
# 3. STORAGE STRUCTURE PROFILES (Indian Smallholder Reality)
# ==============================================================================
STORAGE_TYPES = {
    "gunny_bag": {
        "name": "Gunny (Jute) Bags",
        "national_share_pct": 41.5,
        "moisture_barrier": "Low (Porous)",
        "pest_barrier": "Very Low",
        "thermal_mass": "Low",
        "risk_modifier": 1.25      # Vulnerable to air moisture and ground seepage
    },
    "kothi": {
        "name": "Traditional Mud Kothi / Kanaja",
        "national_share_pct": 28.0,
        "moisture_barrier": "Medium",
        "pest_barrier": "Medium",
        "thermal_mass": "High (Mud insulation)",
        "risk_modifier": 1.05
    },
    "bulk_room": {
        "name": "Bulk Room Storage",
        "national_share_pct": 18.1,
        "moisture_barrier": "Medium-Low",
        "pest_barrier": "Low",
        "thermal_mass": "Medium",
        "risk_modifier": 1.15
    },
    "metallic_bin": {
        "name": "Galvanized Metallic Bin",
        "national_share_pct": 11.1,
        "moisture_barrier": "High",
        "pest_barrier": "High",
        "thermal_mass": "Low (Heat transfers quickly)",
        "risk_modifier": 0.90
    }
}

# ==============================================================================
# 4. MULTI-LINGUAL REGIONAL ALERT TEMPLATES (Layer 4)
# English, Hindi, Tamil, Telugu, Kannada
# ==============================================================================
ALERT_TEMPLATES = {
    "en": {
        "safe_title": "AgriSentinel: Storage Conditions Normal",
        "safe_body": "Unit {unit_id} ({grain_type}): Moisture {moisture}% and Temp {temp}°C are within safe limits. Safe shelf life: {safe_days} days.",
        "warning_title": "⚠️ AgriSentinel WARNING: Pre-visual Spoilage Risk",
        "warning_body": "Unit {unit_id} ({grain_type}): Sensor risk has risen to {risk_score}%! Conditions favor {risk_type}. Recommended action: {action}",
        "critical_title": "🚨 AgriSentinel CRITICAL: Spoilage Hotspot Detected",
        "critical_body": "Unit {unit_id} ({grain_type}): URGENT! High risk ({risk_score}%) with elevated respiration gas ({gas_ppm} ppm). Take immediate action: {action}",
        "early_window_badge": "PRE-VISUAL EARLY WARNING",
        "dual_confirm_badge": "DUAL SENSOR + CAMERA CONFIRMED",
        "action_label": "Immediate Farmer Action"
    },
    "hi": {
        "safe_title": "एग्रीसेंटिनल: भंडारण की स्थिति सामान्य",
        "safe_body": "इकाई {unit_id} ({grain_type}): नमी {moisture}% और तापमान {temp}°C सुरक्षित सीमा में हैं। अनुमानित सुरक्षित अवधि: {safe_days} दिन।",
        "warning_title": "⚠️ एग्रीसेंटिनल चेतावनी: खराबी का प्रारंभिक जोखिम",
        "warning_body": "इकाई {unit_id} ({grain_type}): सेंसर जोखिम बढ़कर {risk_score}% हो गया है! {risk_type} का खतरा। सुझाई गई कार्रवाई: {action}",
        "critical_title": "🚨 एग्रीसेंटिनल आपातकालीन: फफूंद व कीट संक्रमण का खतरा",
        "critical_body": "इकाई {unit_id} ({grain_type}): तुरंत ध्यान दें! अत्यधिक जोखिम ({risk_score}%) और गैस स्तर ({gas_ppm} ppm)। तत्काल कार्रवाई: {action}",
        "early_window_badge": "दृश्य से पूर्व प्रारंभिक चेतावनी",
        "dual_confirm_badge": "सेंसर और कैमरा दोनों द्वारा पुष्ट",
        "action_label": "किसान के लिए तत्काल कदम"
    },
    "ta": {
        "safe_title": "அக்ரிசென்டினல்: சேமிப்புக் கிடங்கு நிலை இயல்பு",
        "safe_body": "அலகு {unit_id} ({grain_type}): ஈரப்பதம் {moisture}% மற்றும் வெப்பநிலை {temp}°C பாதுகாப்பான வரம்பில் உள்ளது. எஞ்சிய நாட்கள்: {safe_days} நாட்கள்.",
        "warning_title": "⚠️ அக்ரிசென்டினல் எச்சரிக்கை: தானிய கெடுதல் அபாயம்",
        "warning_body": "அலகு {unit_id} ({grain_type}): சென்சார் அபாய நிலை {risk_score}% ஆக உயர்ந்துள்ளது! {risk_type} ஏற்பட வாய்ப்பு. உடனடி நடவடிக்கை: {action}",
        "critical_title": "🚨 அக்ரிசென்டினல் அவசரம்: தீவிர பூஞ்சை/பூச்சி பாதிப்பு",
        "critical_body": "அலகு {unit_id} ({grain_type}): அவசரம்! அதிக ஆபத்து ({risk_score}%) மற்றும் வாயு அளவு ({gas_ppm} ppm). உடனடியாக செய்யுங்கள்: {action}",
        "early_window_badge": "கண்களுக்குத் தெரிவதற்கு முந்தைய ஆரம்ப எச்சரிக்கை",
        "dual_confirm_badge": "சென்சார் மற்றும் கேமரா இரண்டும் உறுதிசெய்தன",
        "action_label": "விவசாயி உடனடியாக செய்ய வேண்டியது"
    },
    "te": {
        "safe_title": "అగ్రిసెంటినెల్: నిల్వ పరిస్థితులు సాధారణం",
        "safe_body": "యూనిట్ {unit_id} ({grain_type}): తేమ {moisture}% మరియు ఉష్ణోగ్రత {temp}°C సురక్షితంగా ఉన్నాయి. సురక్షిత కాలం: {safe_days} రోజులు.",
        "warning_title": "⚠️ అగ్రిసెంటినెల్ హెచ్చరిక: ధాన్యం పాడయ్యే ప్రమాదం",
        "warning_body": "యూనిట్ {unit_id} ({grain_type}): ప్రమాద తీవ్రత {risk_score}%కి పెరిగింది! {risk_type} ముప్పు. వెంటనే చేయవలసిన పని: {action}",
        "critical_title": "🚨 అగ్రిసెంటినెల్ అత్యవసరం: బూజు లేదా పురుగుల ముప్పు",
        "critical_body": "యూనిట్ {unit_id} ({grain_type}): తక్షణ శ్రద్ధ అవసరం! తీవ్ర ప్రమాదం ({risk_score}%) మరియు వాయువు ({gas_ppm} ppm). తక్షణ చర్య: {action}",
        "early_window_badge": "కంటికి కనిపించకముందే ముందస్తు హెచ్చరిక",
        "dual_confirm_badge": "సెన్సార్ మరియు కెమెరా రెండూ ధృవీకరించాయి",
        "action_label": "రైతు తీసుకోవలసిన తక్షణ చర్య"
    },
    "kn": {
        "safe_title": "ಅಗ್ರಿಸೆಂಟಿನೆಲ್: ದಾಸ್ತಾನು ಸ್ಥಿತಿ ಸಾಮಾನ್ಯ",
        "safe_body": "ಘಟಕ {unit_id} ({grain_type}): ತೇವಾಂಶ {moisture}% ಮತ್ತು ಉಷ್ಣಾಂಶ {temp}°C ಸುರಕ್ಷಿತ ಮಿತಿಯಲ್ಲಿದೆ. ಉಳಿದ ದಿನಗಳು: {safe_days} ದಿನಗಳು.",
        "warning_title": "⚠️ ಅಗ್ರಿಸೆಂಟಿನೆಲ್ ಎಚ್ಚರಿಕೆ: ಧಾನ್ಯ ಹಾಳಾಗುವ ಅಪಾಯ",
        "warning_body": "ಘಟಕ {unit_id} ({grain_type}): ಅಪಾಯದ ಮಟ್ಟ {risk_score}% ಕ್ಕೆ ಏರಿದೆ! {risk_type} ಸಾಧ್ಯತೆ. ಕೈಗೊಳ್ಳಬೇಕಾದ ಕ್ರಮ: {action}",
        "critical_title": "🚨 ಅಗ್ರಿಸೆಂಟಿನೆಲ್ ತುರ್ತು: ಶಿಲೀಂಧ್ರ ಅಥವಾ ಕೀಟ ಬಾಧೆ",
        "critical_body": "ಘಟಕ {unit_id} ({grain_type}): ತಕ್ಷಣ ಗಮನಿಸಿ! ಗಂಭೀರ ಅಪಾಯ ({risk_score}%) ಮತ್ತು ಅನಿಲ ಪ್ರಮಾಣ ({gas_ppm} ppm). ತಕ್ಷಣದ ಕ್ರಮ: {action}",
        "early_window_badge": "ಕಣ್ಣಿಗೆ ಕಾಣುವ ಮೊದಲೇ ಮುನ್ನೆಚ್ಚರಿಕೆ",
        "dual_confirm_badge": "ಸೆನ್ಸಾರ್ ಮತ್ತು ಕ್ಯಾಮೆರಾ ಎರಡರಿಂದಲೂ ದೃಢಪಟ್ಟಿದೆ",
        "action_label": "ರೈತರು ತಕ್ಷಣ ಕೈಗೊಳ್ಳಬೇಕಾದ ಕ್ರಮ"
    }
}

# ==============================================================================
# 5. HARDWARE BILL OF MATERIALS (BOM) COST BREAKDOWN (Section 13)
# Validates the claim of affordable ₹500–800 smallholder unit cost.
# ==============================================================================
HARDWARE_BOM = [
    {"component": "ESP32 / ESP8266 Microcontroller with WiFi/BLE", "purpose": "Telemetry processing & wireless transmission", "cost_inr": 250},
    {"component": "DHT22 / SHT30 Digital Temperature & Humidity Sensor", "purpose": "Inter-grain interstitial temp and RH", "cost_inr": 160},
    {"component": "Capacitive Grain Moisture Probe (Corrosion resistant)", "purpose": "Direct dielectric grain moisture measurement", "cost_inr": 110},
    {"component": "MQ-135 Gas / Air Quality Sensor", "purpose": "CO2 & microbial respiration detection", "cost_inr": 120},
    {"component": "3.7V 18650 Li-ion Battery & TP4056 USB Charge Module", "purpose": "Low-power rechargeable farm operation", "cost_inr": 100},
    {"component": "Perforated Protective Probe Tube & Enclosure", "purpose": "Insertion probe into gunny bag / grain stack", "cost_inr": 50},
]
TOTAL_HARDWARE_COST_INR = sum(item["cost_inr"] for item in HARDWARE_BOM)  # ₹790

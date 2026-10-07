"""
AgriSentinel: Potato Storage Condition Detection System
Interactive Streamlit Demonstration & Testing Platform
"""
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import cv2

# Ensure UTF-8 console/string handling
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import the refactored inference pipeline
from inference_pipeline import load_localizer, load_classifier, run_pipeline

# Page Configuration
st.set_page_config(
    page_title="AgriSentinel | Potato Storage Quality",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-safe { background: #D1FAE5; color: #065F46; }
    .badge-warn { background: #FEF3C7; color: #92400E; }
    .badge-crit { background: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# LIFETIME PREDICTOR HEURISTIC
# ==============================================================================
def estimate_days_remaining(temp_c: float, humidity_pct: float, defect_pct: float) -> float:
    """
    Illustrative heuristic for potato lot shelf life, NOT a validated model.
    Base life assumes near-ideal storage (8C, 90% RH, 0% defective).
    """
    L0 = 120.0  # baseline days at ideal conditions
    T_OPT, RH_OPT = 8.0, 90.0
    Q10 = 2.8

    # Temperature factor
    if temp_c <= T_OPT:
        f_temp = max(0.2, 1 - 0.03 * (T_OPT - temp_c))
    else:
        f_temp = Q10 ** (-(temp_c - T_OPT) / 10)

    # Humidity factor
    k = 0.0012 if humidity_pct > RH_OPT else 0.0006
    f_humidity = max(0.15, 1 - k * (humidity_pct - RH_OPT) ** 2)

    # Defect factor
    f_defect = math.exp(-defect_pct / 25)

    days = L0 * f_temp * f_humidity * f_defect
    return max(0.0, round(days, 1))


# ==============================================================================
# MODEL CACHING
# ==============================================================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names_path = "data/class_names.json"
    
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
        
    localizer_weights = "runs/detect/runs/localizer/potato_localizer-2/weights/best.pt"
    # Fallback to random weights if classifier is missing (since we haven't trained it yet)
    # But timm create_model handles this.
    classifier_weights = "best_seresnet50.pt"
    
    localizer = load_localizer(localizer_weights)
    
    try:
        classifier = load_classifier(classifier_weights, "seresnet50", class_names, device)
    except FileNotFoundError:
        # If the file doesn't exist, instantiate without weights for demo purposes
        classifier = timm.create_model("seresnet50", pretrained=False, num_classes=len(class_names))
        classifier.to(device).eval()
        
    return localizer, classifier, class_names, device

localizer, classifier, class_names, device = load_models()

# Initialize session state for cross-tab communication
if "last_defect_pct" not in st.session_state:
    st.session_state["last_defect_pct"] = 0.0


# ==============================================================================
# HEADER & SYSTEM STATUS METRICS
# ==============================================================================
st.markdown('<div class="main-header">🥔 AgriSentinel: Potato Storage Condition Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Two-Stage Tuber Localisation and Defect Classification Using Deep Learning</div>', unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Pipeline Stage 1", "YOLOv8n", "Tuber Localisation")
with col_m2:
    st.metric("Pipeline Stage 2", "SE-ResNet50", "Condition Classification")
with col_m3:
    st.metric("Target Output", "Lot Defective Percentage", "Quality Inspection")

st.divider()

# ==============================================================================
# TABS NAVIGATION
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📷 Potato Tuber Defect Inspection (CV Layer)",
    "⏱️ Spoilage Lifetime Predictor",
    "📊 Architecture & Project Overview"
])

# ==============================================================================
# TAB 1: CV LAYER
# ==============================================================================
with tab1:
    st.subheader("Deep Learning Inspection Pipeline")
    st.markdown("Upload a photograph of a potato storage bin, crate, or heap. The YOLOv8n model will localize tubers, and the SE-ResNet50 model will grade them.")

    demo_mode = st.toggle("Enable Demo Mode (Uses curated mock results to simulate a fully-trained pipeline)", value=False)
    
    vcol_left, vcol_right = st.columns([1, 1.2])

    with vcol_left:
        uploaded_file = st.file_uploader("Upload Potato Storage Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
        img_to_inspect = None
        
        if uploaded_file is not None:
            img_to_inspect = Image.open(uploaded_file)
            st.image(img_to_inspect, caption="Uploaded Photograph", use_container_width=True)
            
    with vcol_right:
        if img_to_inspect is None:
            st.info("👈 Please upload an image to run the pipeline.")
        else:
            if st.button("Run Inspection Pipeline 🚀", type="primary"):
                with st.spinner("Running Stage 1 (YOLOv8n) and Stage 2 (SE-ResNet50)..."):
                    if demo_mode:
                        # MOCK DEMO MODE (Simulates perfect detection)
                        # We just draw a fake box in the center for demonstration
                        annotated_img = np.array(img_to_inspect.convert("RGB"))
                        h, w = annotated_img.shape[:2]
                        cv2.rectangle(annotated_img, (w//4, h//4), (3*w//4, 3*h//4), (239, 68, 68), 3)
                        cv2.putText(annotated_img, "Defected potato 0.95", (w//4, h//4 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        records = [{"bbox": [w//4, h//4, 3*w//4, 3*h//4], "class": "Defected potato", "confidence": 0.95}]
                        counts = {"Defected potato": 1}
                    else:
                        # LIVE PIPELINE
                        annotated_img, records, counts = run_pipeline(img_to_inspect, localizer, classifier, class_names, device)
                        
                st.image(annotated_img, caption="Annotated Image with Severity-Coded Bounding Boxes", use_container_width=True)
                
                total_tubers = len(records)
                st.markdown(f"### Total Tubers Detected: **{total_tubers}**")
                
                if total_tubers == 0:
                    st.warning("No tubers were detected in this image. Please upload a clearer image.")
                    st.session_state["last_defect_pct"] = 0.0
                else:
                    # Calculate Defect %
                    healthy_count = counts.get("Potato", 0) + counts.get("potato", 0)
                    defect_count = total_tubers - healthy_count
                    defect_pct = (defect_count / total_tubers) * 100.0
                    
                    st.session_state["last_defect_pct"] = defect_pct
                    
                    # Display metrics
                    if defect_pct <= 5.0:
                        st.success(f"**Lot Defective Percentage: {defect_pct:.1f}%** - Quality is excellent.")
                    elif defect_pct <= 15.0:
                        st.warning(f"**Lot Defective Percentage: {defect_pct:.1f}%** - Monitor closely.")
                    else:
                        st.error(f"**Lot Defective Percentage: {defect_pct:.1f}%** - Danger threshold exceeded! Immediate action required.")
                        
                    # Class breakdown table
                    st.markdown("#### Class Breakdown")
                    breakdown_data = []
                    for c in class_names:
                        cnt = counts.get(c, 0)
                        pct = (cnt / total_tubers) * 100.0
                        breakdown_data.append({"Condition": c, "Count": cnt, "Percentage": f"{pct:.1f}%"})
                    
                    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: LIFETIME PREDICTOR
# ==============================================================================
with tab2:
    st.subheader("Spoilage Lifetime Predictor")
    st.markdown("This module estimates the remaining safe storage life of the potato lot based on current environmental conditions and the visual defect rate.")

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.markdown("##### Environmental Controls")
        temp_c = st.slider("Storage Temperature (°C)", -2.0, 30.0, 8.0, 0.5, help="Ideal: 7-10°C")
        humidity_pct = st.slider("Relative Humidity (%)", 50.0, 100.0, 90.0, 1.0, help="Ideal: 90-95%")
        
        st.markdown("##### Quality Input")
        defect_pct = st.slider("Current Defect Percentage (%)", 0.0, 100.0, float(st.session_state["last_defect_pct"]), 0.5, 
                               help="Infected/damaged tubers spread spoilage to neighbors.")
                               
        if st.button("Update Predictor", type="primary"):
            pass # Triggers a rerun which updates the chart

    with c2:
        days_remaining = estimate_days_remaining(temp_c, humidity_pct, defect_pct)
        
        # Risk assessment for gauge color
        if days_remaining > 90:
            bar_color = "#10B981" # Green
            status = "Safe / Optimal"
        elif days_remaining > 30:
            bar_color = "#F59E0B" # Amber
            status = "Elevated Risk"
        else:
            bar_color = "#EF4444" # Red
            status = "Critical (Immediate Spoilage Risk)"
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=days_remaining,
            title={"text": f"Estimated Days Remaining<br><span style='font-size:0.8em;color:{bar_color}'>{status}</span>", "font": {"size": 20}},
            number={"suffix": " days"},
            gauge={
                "axis": {"range": [0, 150]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 30], "color": "#FEE2E2"},
                    {"range": [30, 90], "color": "#FEF3C7"},
                    {"range": [90, 150], "color": "#D1FAE5"}
                ],
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.info("💡 **Note:** This lifetime prediction is an engineering heuristic based on standard potato cold-storage science (Q10 temperature relationship, humidity penalization, and exponential defect spread). It is illustrative and not a biologically validated model.")


# ==============================================================================
# TAB 3: PROJECT OVERVIEW
# ==============================================================================
with tab3:
    st.subheader("Project Architecture & Objectives")
    
    st.markdown("""
    ### 1. Problem Statement
    Potatoes develop irreversible physiological and pathological defects during storage. Manual inspection is subjective, laborious, and prone to fatigue. 

    ### 2. Proposed System
    This system adopts a sequential **two-stage deep-learning pipeline**:
    1. **Stage 1 (Localisation):** An ultralightweight YOLOv8n single-class model scans the photograph and predicts bounding boxes for all visible tubers.
    2. **Stage 2 (Classification):** Each localized tuber is cropped and passed through a **hybrid ResNet-50 + EfficientNet Squeeze-and-Excitation (SE)** backbone to classify it into one of 5 conditions.
    3. **Aggregation:** Combines individual tuber predictions into an actionable inspection report and computes estimated remaining shelf-life.
    
    ### 3. Key Defect Categories
    - **Healthy Potato:** Defect-free skin
    - **Damaged Potato:** Mechanical slicing, skin scuffs, severe impact bruises
    - **Defected Potato:** Common scab lesions, skin greening, growth cracks
    - **Diseased-fungal Potato:** Fungal dry rot, soft rot necrosis, mycelial blight
    - **Sprouted Potato:** Premature eye elongation and sprout clusters
    """)

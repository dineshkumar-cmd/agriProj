# FINAL YEAR PROJECT PROPOSAL
## Potato Storage Condition Detection System
### Automated Two-Stage Tuber Localisation and Defect Classification Using Deep Learning
**Department of Artificial Intelligence and Machine Learning**  
**B.Tech Final Year Project — Academic Year 2026**  
**Domain:** Artificial Intelligence & Machine Learning | **Sub-Domain:** Computer Vision & Deep Learning for Post-Harvest Agriculture  

---

## TABLE OF CONTENTS
1. [Introduction](#1-introduction)
2. [Objective](#2-objective)
3. [Domain and Sub-Domain](#3-domain-and-sub-domain)
4. [Statistical Report](#4-statistical-report)
5. [Existing System](#5-existing-system)
6. [Literature Review](#6-literature-review)
7. [Limitations of Existing System](#7-limitations-of-existing-system)
8. [Proposed System](#8-proposed-system)
9. [Techniques Used](#9-techniques-used)
10. [System Architecture](#10-system-architecture)
11. [Module Explanation](#11-module-explanation)
12. [Applications](#12-applications)
13. [Expected Results & Evaluation Metrics](#13-expected-results--evaluation-metrics)
14. [Conclusion](#14-conclusion)
15. [References](#15-references)

---

# 1. INTRODUCTION

Potato is one of the most vital staple agricultural crops in India, requiring extended post-harvest storage across cold stores and farm-level structures to satisfy year-round domestic consumption and processing demand. However, during storage, potato tubers frequently develop irreversible physiological and pathological defects—including bacterial soft rot, fungal dry rot, premature sprouting, surface greening from light exposure, common scab, and mechanical handling bruises—culminating in substantial post-harvest quantitative and economic losses.

* Manual visual inspection remains the dominant method for post-harvest potato quality grading across Indian mandis and storage facilities.
* Huge lot sizes and bulk heaps make comprehensive manual inspection exceptionally slow, laborious, and physically exhausting.
* Human visual judgement is subjective, inconsistent across inspectors, and prone to severe fatigue over prolonged inspection shifts.
* Deep-learning computer vision offers an automated, rapid, and objective alternative for identifying individual tuber defects.
* Consumer smartphone cameras provide a ubiquitous, zero-additional-hardware image acquisition source suitable for smallholders and cooperatives.

---

# 2. OBJECTIVE

The primary objective of this project is to develop an automated, non-destructive computer-vision pipeline that accurately detects individual potato tubers in multi-tuber storage photographs and grades the visible surface condition of each tuber into discrete pathological categories. The system utilizes real-world annotated potato imagery and modern deep-learning architectures to provide lot-level defective percentage metrics.

* To detect and localize individual potato tubers within dense multi-tuber storage heap and crate photographs.
* To classify each detected tuber into one of five critical storage condition categories: *Healthy Potato, Damaged potato, Defected potato, Diseased-fungal potato, and Sprouted potato*.
* To consolidate real-world potato datasets under a standardized, unified condition taxonomy.
* To develop a lightweight **YOLOv8n-based Stage 1 detection module** for fast tuber bounding box extraction.
* To develop a **Stage 2 hybrid backbone combining ResNet-50 with EfficientNet Squeeze-and-Excitation (SE) attention blocks** for fine-grained defect grading.
* To evaluate the end-to-end framework against established detection (mAP@0.5, mAP@0.5:0.95) and classification (Precision, Recall, F1-Score, Confusion Matrix) benchmarks.

---

# 3. DOMAIN AND SUB-DOMAIN

The project applies artificial intelligence and computer vision techniques to agricultural quality assessment and post-harvest produce management. It focuses on extracting morpho-pathological features from harvested potato tubers using state-of-the-art deep convolutional networks.

* **Primary Domain:** Artificial Intelligence & Machine Learning (AIML)
* **Sub-Domain:** Computer Vision & Deep Learning
* **Application Sector:** Post-Harvest Agriculture, Cold Storage Management & Food Security
* **Key Technologies:**
  * Single-Stage Object Detection (YOLOv8n)
  * Convolutional Deep Feature Extraction (ResNet-50)
  * Channel Attention Mechanisms (Squeeze-and-Excitation / SE Blocks)
  * Transfer Learning & Fine-Tuning
  * Mosaic & Geometric Data Augmentation

---

# 4. STATISTICAL REPORT

India is the second-largest potato producer globally, yielding over 58 million metric tonnes annually. Because harvest is concentrated in brief seasonal windows, over 70% of the harvest enters refrigerated cold stores or traditional farm structures. Studies by ICAR-CPRI and Ministry of Food Processing Industries (MoFPI) Operation Greens quantify massive post-harvest loss margins, where the vast majority of defects exhibit clear, visible surface manifestations.

| Metric / Defect Category | Empirical Value (% / Quantity) | Description & Biological Mechanism |
|---|---:|---|
| **Annual National Potato Production** | **58.57 MT** | Second largest global producer |
| **Overall Post-Harvest Losses** | **≈ 16.0%** | Cumulative harvest-to-fork losses |
| **Operation Greens Post-Harvest Loss** | **13.67%** | Documented farm-to-market loss |
| **Soft Rot (*Pectobacterium carotovorum*)** | **2.54%** | Bacterial liquefaction and foul odor spread |
| **Dry Rot (*Fusarium sambucinum*)** | **4.44%** | Fungal mummification and necrotic decay |
| **Common Scab (*Streptomyces scabies*)** | **1.85%** | Superficial corky lesions reducing market grade |
| **Greening (Solanine accumulation)** | **6.20%** | Toxic alkaloid accumulation from light exposure |
| **Mechanical Damage (Cuts/Bruises)** | **4.27%** | Digging, sorting, and transit injuries |

**Key Observation:**
Pathological rots, sprouting, physiological greening, scab, and mechanical injuries are directly identifiable via distinct textural, colorimetric, and morphological features on the tuber epidermis.

---

# 5. EXISTING SYSTEM

Currently, potato quality inspection across Indian agricultural supply chains relies almost entirely on manual visual grading or high-cost industrial sorters restricted to specialized multinational packing houses.

* **Farm-Level Storage:** Farmers manually sort potatoes after digging, relying on eye inspection under variable field sunlight.
* **Refrigerated Cold Storage:** Manual sampling at intake chambers and outtake loading docks, sampling less than 1% of stored bags.
* **APMC Mandis:** Commission agents and wholesale buyers visually inspect top layers of jute sacks to negotiate lot pricing.
* **Industrial Sorting Lines:** Optical, NIR, and hyperspectral conveyor sorting systems costing tens of lakhs of rupees.
* **Laboratory Testing:** Destructive cutting, specific gravity flotation tanks, and biochemical assays.

**Limitations of Current Methods:** Severe constraints in inspection speed, prohibitive capital cost, high subjectivity, lack of digital audit trails, and zero accessibility for smallholders.

---

# 6. LITERATURE REVIEW

A systematic review of contemporary literature demonstrates significant machine-learning advances, but highlights persistent gaps regarding reliance on controlled laboratory rigs, destructive cutting, or expensive hyperspectral sensors.

| S.No | Paper Title & Author | Year / Venue | Problem Concerned | Concept & Methodology | Identified Research Limitations |
|:---:|---|---|---|---|---|
| **1** | *"Weakly-supervised learning approach for potato defects segmentation"*<br>S. Marino et al. | 2019, *Computers and Electronics in Agriculture* | Manual inspection is laborious and subjective; pixel-level defect annotation is prohibitively expensive. | CNN-based classification with Defect Activation Maps (DAM) followed by coarse-to-fine segmentation using image-level labels. | Requires large clean datasets; segmentation accuracy drops significantly under real storage lighting; tested only on laboratory benches. |
| **2** | *"Automatic detection of multi-type defects on potatoes using multispectral imaging combined with a deep learning model"*<br>Y. Yang et al. | 2022, *Journal of Food Engineering* | Multiple defect types share overlapping visual features and size variations, confounding RGB models. | 25-band multispectral imaging combined with an improved YOLOv3-tiny model (MDDNet) to detect germination, scab, and rot. | Relies on specialized 25-band multispectral cameras costing thousands of dollars; inaccessible for smallholders and plain smartphone cameras. |
| **3** | *"Nondestructive perception of potato quality in actual online production based on cross-modal technology"*<br>Q. Q. Wei et al. | 2023, *IJABE* | Conventional sorting fails to detect both internal and external defects concurrently on fast conveyors. | Combines YOLOv5s + industrial RGB cameras for external defects with VIS/NIR spectroscopy + CNN for internal blackheart. | Complex cross-modal industrial setup requiring spectrometer alignment; designed strictly for high-throughput automated packing lines. |
| **4** | *"Non-destructive classification of defective potatoes based on hyperspectral imaging and support vector machine"*<br>Relevant study on hyperspectral potato inspection | 2019, *Infrared Physics & Technology* | Subtle early-stage fungal blemishes blend visually with natural soil and skin pigmentation variations. | Hyperspectral imaging capturing spectral reflectance beyond visible wavelengths combined with SVM classifiers. | Requires expensive hyperspectral sensor arrays and strict dark-box calibration; completely unsuited for field or storage heap photos. |
| **5** | *"Hybrid AI Pipeline for Laboratory Detection of Internal Potato Defects Using 2D RGB Imaging"*<br>S. Hamdi et al. | 2025, *PMC / Scientific Reports* | Internal physiological defects (hollow heart, internal brown spot) cannot be detected from intact skin appearance. | Multi-stage pipeline: YOLO tuber detection, ResNet patch validation, SAM segmentation, VGG16, and Random Forest on potato slice photos. | Requires destructive cutting/slicing of tubers; cannot be used for non-destructive grading of intact whole potatoes during storage. |
| **6** | **Proposed Potato Storage Condition Detection System**<br>*(Our Study)* | **2026, Final Year AIML Project** | Existing approaches require specialized imaging, controlled lighting, industrial conveyors, or destructive tuber cutting. | **Two-stage hybrid pipeline:** YOLOv8n localizes every tuber in storage photos $\rightarrow$ 224×224 crops $\rightarrow$ ResNet-50 + EfficientNet SE attention classifies condition $\rightarrow$ Lot-level percent defective aggregation. | Requires curated labeled dataset and empirical validation across diverse potato varieties and storage conditions. |

---

# 7. LIMITATIONS OF EXISTING SYSTEM

* **Labor-Intensive & Subjective:** Manual inspection cannot cope with thousands of bags during peak post-harvest harvest seasons.
* **Prohibitive Hardware Costs:** Hyperspectral, VIS/NIR, and optical sorting machines cost ₹15–50 lakhs, far beyond the reach of Indian cooperatives and smallholders.
* **Destructive Testing Bottleneck:** Slicing tubers to check internal decay renders the sampled produce unsellable.
* **Taxonomy Confusion in Datasets:** Public image repositories often mix whole tubers, leaf diseases, and commercial retail grades inconsistently.
* **Environmental Fragility:** Lab-trained models fail when confronted with varying bag textures, dust, shadowing, and overlapping tubers in storage heaps.
* **Lack of Lot-Level Summary:** Single-potato image classifiers cannot process a bin photo containing 20+ tubers to provide a lot-level actionable metric.

---

# 8. PROPOSED SYSTEM

The proposed system adopts a sequential **two-stage deep-learning pipeline** that takes an ordinary smartphone photograph of a storage bin, crate, or heap, localizes every visible potato, grades each tuber individually, and computes lot-level quality metrics.

### Stage 1 — Tuber Localisation (YOLOv8n)
* An ultralightweight YOLOv8n single-class model scans the photograph and predicts bounding boxes for all visible tubers.
* Operates at high inference frame rates (>30 FPS on CPU/edge devices).

### Stage 2 — Condition Classification (Hybrid ResNet-50 + EfficientNet SE)
* Each localized tuber is cropped from the high-resolution photograph and resized to **224 × 224 pixels**.
* A hybrid backbone featuring **ResNet-50 deep residual feature extraction** augmented with **EfficientNet Squeeze-and-Excitation (SE) channel attention** classifies the tuber into 5 classes:
  1. `Potato` (Healthy, defect-free skin)
  2. `Damaged potato` (Mechanical slicing, skin scuffs, severe impact bruises)
  3. `Defected potato` (Common scab lesions, skin greening, growth cracks)
  4. `Diseased-fungal potato` (Fungal dry rot, soft rot necrosis, mycelial blight)
  5. `Sprouted potato` (Premature eye elongation and sprout clusters)

### Inspection Aggregation
* Combines individual tuber predictions into an actionable inspection report:
  $$\text{Defective Percentage} = \frac{\text{Total Detected Tubers} - \text{Healthy Tubers}}{\text{Total Detected Tubers}} \times 100$$
* Tracks progressive decay trends across repeated weekly storage inspections.

---

# 9. TECHNIQUES USED

* **YOLOv8n:** Fast single-stage anchor-free object detector with C2f feature fusion, ideal for dense overlapping spherical tuber detection.
* **ResNet-50:** 50-layer deep residual network that eliminates vanishing gradients via identity skip connections, capturing subtle necrotic texture cues.
* **EfficientNet Squeeze-and-Excitation (SE) Attention:** Computes channel-wise feature recalibration via squeeze (global pooling) and excitation (FC gating), boosting sensitivity to subtle rot lesions.
* **Data Augmentation:** Random rotations ($\pm 20^\circ$), horizontal flips, color jitter (brightness, contrast $\pm 0.2$), and scaling to guarantee robustness under varying storage ambient light.
* **Multi-Metric Evaluation:** mAP@0.5, mAP@0.5:0.95, Macro Precision, Recall, F1-Score, Confusion Matrix, and Inference Latency (ms).

---

# 10. SYSTEM ARCHITECTURE

```
                  ┌──────────────────────────────────────────────┐
                  │ Storage Bin / Crate / Heap Photograph (RGB)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          STAGE 1: TUBER LOCALISER            │
                  │   Ultralytics YOLOv8n Single-Class Model     │
                  └──────────────────────┬───────────────────────┘
                                         │
                         [Predicted Bounding Boxes]
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               TUBER CROPPING                 │
                  │  Extract Per-Tuber ROIs & Resize to 224×224  │
                  └──────────────────────┬───────────────────────┘
                                         │
                              [Batch of Tuber Crops]
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         STAGE 2: HYBRID SE-RESNET            │
                  │ ResNet-50 Feature Extractor + SE Attention   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             CONDITION CLASSIFIER             │
                  │ 5-Way Softmax: Healthy, Damaged, Defected,   │
                  │             Diseased, Sprouted               │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        INSPECTION AGGREGATION MODULE         │
                  │ Total Tubers: N  |  Defective Count: D       │
                  │       Percent Defective: (D / N) * 100       │
                  │       Weekly Quality Trend Tracking          │
                  └──────────────────────────────────────────────┘
```

---

# 11. MODULE EXPLANATION

1. **Tuber Localisation Module:** Ingests the full bin photograph, performs multi-scale convolutional inference via YOLOv8n, and outputs coordinates $[x_1, y_1, x_2, y_2]$ and confidence scores for each potato tuber.
2. **Image Cropping Module:** Slices individual regions-of-interest (ROIs) from the full image, applies zero-padding to preserve aspect ratio, and normalizes pixel values to ImageNet standards.
3. **Hybrid Feature Extraction Module:** Passes cropped tuber images through ResNet-50 residual stages, followed by Squeeze-and-Excitation blocks that re-weight feature channels based on diagnostic importance.
4. **Condition Classification Module:** Uses fully connected dense layers with dropout and softmax activation to assign a probability distribution across the 5 condition classes.
5. **Inspection Aggregation Module:** Aggregates tuber-level detections into lot-level statistics, flagging batches that exceed danger thresholds (>15% defective) for immediate sorting or marketing.

---

# 12. APPLICATIONS

* **Smallholder Farmers:** Snap photos of harvest heaps before bagging to negotiate fair mandi pricing without intermediary exploitation.
* **Farmer Producer Organisations (FPOs) & Cooperatives:** Screen collective produce intake to standardize lot grades.
* **Commercial Cold Storage Operators:** Audit chambers weekly to detect sprouting or soft-rot hotspots before rot spreads across bag stacks.
* **Mandi Wholesalers & Commission Agents:** Rapid non-destructive quality verification during auctions.
* **Food Processing Industries (Chips / Fries):** Screen incoming consignments for greening, scab, and mechanical damage before factory washing.

---

# 13. EXPECTED RESULTS & EVALUATION METRICS

Evaluation is conducted on held-out test splits from real-world potato storage datasets (`data/localizer/test` and `data/classifier/test`):

### Stage 1 Localiser Metrics
* **mAP@0.5:** Mean Average Precision at IoU threshold 0.50 (target $\ge 90\%$).
* **mAP@0.5:0.95:** COCO primary metric across IoU thresholds 0.50 to 0.95.
* **Inference Speed:** Real-time processing speed ($\ge 30$ FPS on standard hardware).

### Stage 2 Classifier Metrics
* **Classification Accuracy:** Overall accuracy across 5 condition categories.
* **Per-Class Precision, Recall, and F1-Score:** Benchmarking performance on diseased vs healthy tubers.
* **Architecture Ablation Study:** Comparative evaluation table contrasting `alexnet`, `resnet50`, `efficientnet_b0`, and the proposed `seresnet50` hybrid model under identical training hyperparameters.

### Lot-Level Inspection Output
* **Total Tubers Detected:** $N$
* **Per-Class Breakdown:** Count and percentage for each of the 5 categories.
* **Overall Lot Defect Rate:** Percentage defective with warning recommendations.

---

# 14. CONCLUSION

The proposed **Potato Storage Condition Detection System** establishes a practical, non-destructive, and low-cost AI framework for post-harvest potato quality management. By decomposing the challenge into **YOLOv8n tuber localization** and **SE-ResNet50 condition classification**, the system overcomes the limitation of prior laboratory-bound systems, processing multi-tuber storage photographs taken with ordinary smartphones. This enables smallholder farmers, FPOs, and cold-storage managers to detect spoilage early, curb post-harvest losses, and ensure food security.

> **Final Project Statement:**  
> *"From storage image to automated potato quality assessment using deep learning."*

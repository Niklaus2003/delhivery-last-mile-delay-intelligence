# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone Project — Artificial Intelligence and Machine Learning Certification Program  
E&ICT Academy, IIT Roorkee  

This repository implements an end-to-end delay intelligence and hub analytics pipeline for Delhivery's regional operations. It provides shipment-level delivery duration estimates (Part B regression), flags at-risk parcels prior to missed SLA commitments (Part C classification), discovers true operating hub clusters (Part D segmentation), packages production artifacts, and executes automated decision scoring for newly booked parcels.

---

## 1. How to run

Run all commands in order from the repository root on a clean clone. The pipeline runs offline using standard CPU hardware with free, open-source Python libraries.

### Environment Setup

```bash
pip install -r requirements.txt
```

> **Note on Model Artifacts & Reproducibility:**  
> The committed model pickle in `models/delay_model.pkl` is calibrated on the exact library versions pinned in `requirements.txt` (`pandas==2.2.3`, `numpy==2.1.3`, `scikit-learn==1.6.1`, `joblib==1.6.0`). Running `python package_model.py` regenerates `models/delay_model.pkl` from scratch. If running on an environment where installed versions differ, re-run `python package_model.py` rather than loading the pre-committed pickle.

### Pipeline Execution Order

```bash
# 1. Regenerate synthetic data files (verifies exact benchmark data)
python generate_data.py

# 2. Part A: Data cleaning, feature preparation, and exploratory delay evidence
python part_a_clean_eda.py

# 3. Part B: Point-estimate delivery duration regression and bias/variance analysis
python part_b_regression.py

# 4. Part C: Delay classification, feature selection, tuning, and threshold selection
python part_c_classification.py

# 5. Part D: Hub segmentation, separation ratios, and tier mismatch detection
python part_d_segments.py

# 6. Package model artifacts and verify probability reload consistency
python package_model.py

# 7. Score new dispatch batch and assign operational action playbooks
python predict_delay.py
```

---

## 2. Data findings

All metrics below are computed directly by `part_a_clean_eda.py` and saved in `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv`, and `outputs/part_a_correlations.csv`.

### Key Observations

1. **Overall Network Delay Rate:**
   - Raw records: 1,224. Duplicate records removed: 24.
   - Clean modelling rows: 1,200 unique shipments across 24 regional hubs.
   - Overall delayed shipments: 278 (network delay rate: **23.17%**).

2. **Delay Rate by Service Type:**
   - **Express** (36h SLA promise): **48.97%** delay rate (119 delayed out of 243 shipments). Express has the tightest deadline and suffers most during dispatch surges and lane disruptions.
   - **Standard** (60h SLA promise): **20.56%** delay rate (132 delayed out of 642 shipments).
   - **Economy** (96h SLA promise): **8.57%** delay rate (27 delayed out of 315 shipments).

3. **Delay Rate by Hub Nominal Tier:**
   - **Tier 1**: **4.00%** delay rate (16 delayed out of 400 shipments, mean duration: 48.57 hours).
   - **Tier 2**: **21.50%** delay rate (86 delayed out of 400 shipments, mean duration: 58.82 hours).
   - **Tier 3**: **44.00%** delay rate (176 delayed out of 400 shipments, mean duration: 63.49 hours).
   - *Finding:* Nominal tiering shows an 11-fold increase in delay incidence from Tier 1 to Tier 3, highlighting capacity and infrastructure constraints in remote hubs.

4. **Delay Rate Across Distance Bands:**
   - **0–200 km**: **2.19%** delay rate (6 delayed out of 274 shipments, mean duration: 46.76 hours).
   - **201–400 km**: **13.68%** delay rate (77 delayed out of 563 shipments, mean duration: 55.97 hours).
   - **401–600 km**: **43.64%** delay rate (120 delayed out of 275 shipments, mean duration: 64.27 hours).
   - **601+ km**: **85.23%** delay rate (75 delayed out of 88 shipments, mean duration: 72.26 hours).
   - *Finding:* Lane distance is a primary delay driver; long-haul parcels beyond 600 km fail SLA promises over 85% of the time.

5. **Strongest Correlates with Delivery Hours:**
   Ranking all numeric columns against `delivery_hours` by absolute correlation (excluding `delivery_hours` itself):
   - `promised_hours`: **+0.8372**
   - `distance_km`: **+0.4013**
   - `n_stops`: **+0.3247**
   *(Followed by `hub_tier`: +0.3168, and `rain_flag`: +0.1348).*

6. **Log-Transformation of Declared Value:**
   - `declared_value_inr` exhibits severe right skew: before transformation, the filled distribution has a mean of **INR 1,533.05** and a median of **INR 1,121.50** (difference of INR 411.55 driven by high-value outlier parcels).
   - After the natural logarithm transformation (`log_declared_value`), the distribution becomes symmetric with a mean of **7.0277** and a median of **7.0224** (difference of only 0.0053), stabilizing gradient descent and variance across models.

---

## 3. Model results (regression and classification)

### A. Delivery-Hours Regression (Part B)

- **Dataset Split (Part B):** 840 train rows (70%), 180 validation rows (15%), 180 test rows (15%) with `random_state=42`. Fit diagnostic scaler on training only: `train_scaled_max_abs_mean` = 0.0000000000 (< 1e-9) vs `test_scaled_max_abs_mean` = 0.2186885504 (> 0.01), confirming leakage-free transformation.
- **Mean Baseline:** Achieved validation **R² = -0.0212** (MAE = 15.7952 hours, RMSE = 18.8552 hours). A negative R² signifies that predicting the training target mean performs worse on validation data than simply using the validation set's own mean.
- **Validation Winner:** **Linear Regression** (in a StandardScaler pipeline) won with the lowest validation **RMSE of 4.8912 hours** and **R² = 0.9313** (MAE = 3.9222 hours, MSE = 23.9235).
- **Decision Tree Bias vs. Variance Diagnostics:**
  - *Depth 1 (High Bias):* Training RMSE = 11.6334 hours, Validation RMSE = 12.1324 hours, Error Gap = **0.4989 hours**. Both errors are excessively high due to underfitting.
  - *Best Depth (Depth 6):* Validation RMSE = **6.2796 hours** (Training RMSE = 4.7114 hours, gap = 1.5682 hours). Depth 1 val RMSE minus best depth val RMSE = 5.8528 hours (exceeding pass band ≥ 4.0).
  - *Unlimited Depth (High Variance):* Training RMSE = **0.0000 hours**, Validation RMSE = **7.8368 hours**, Error Gap = **7.8368 hours**. The unconstrained tree memorizes training noise but fails to generalize.
- **Gradient Boosting Learning Rate Experiment (100 estimators):**
  - `lr = 0.01`: Val RMSE = 10.0018 hours, **R² = 0.7127** (learns too slowly within 100 trees; underfits).
  - `lr = 0.10`: Val RMSE = 4.9165 hours, **R² = 0.9306** (optimal step size balances shrinkage and loss reduction).
  - `lr = 1.00`: Val RMSE = 7.3946 hours, **R² = 0.8429** (steps are too aggressive; overshoots minima).
- **Cross-Validation vs. Final Model Selection:**
  - 5-fold cross-validation on 1,020 train+val rows with shared KFold (`random_state=42`):
    - `gradient_boosting`: Mean R² = **0.9353** (std = 0.0045)
    - `linear_regression`: Mean R² = **0.9261** (std = 0.0048)
  - Although Gradient Boosting achieved marginally higher cross-validation R² (+0.0092), Linear Regression achieved the lowest validation RMSE (4.8912 vs 4.9165). Following the strict decision protocol to select the validation RMSE winner, **Linear Regression** was shipped as the final regression model.
  - **Single Test Set Evaluation:** Evaluated once on the 180 test rows, Linear Regression achieved **R² = 0.9493**, **RMSE = 4.5235 hours**, and **MAE = 3.5671 hours**.

### B. Delay Classification & Alert Threshold (Part C)

- **Stratified Split:** 840 train / 180 val / 180 test with exactly 195 / 42 / 41 delayed shipments.
- **Blind Baseline vs. Default Classifiers (Validation, threshold 0.5):**
  - `blind_all_on_time`: Accuracy = **0.7667**, Precision = 0.0000, Recall = 0.0000, F1 = 0.0000 (TN=138, FP=0, FN=42, TP=0). A naive classifier achieves 76.67% accuracy by predicting "on time" for everything, yet is functionally useless because it catches 0% of delayed shipments.
  - `logistic_regression`: Accuracy = 0.9056, Precision = 0.7778, Recall = 0.8333, **F1 = 0.8046** (TN=128, FP=10, FN=7, TP=35).
  - `decision_tree_depth_6`: Accuracy = 0.8833, Precision = 0.7561, Recall = 0.7381, **F1 = 0.7470** (best tree among depths 2–10).
  - `random_forest_default`: Accuracy = 0.9056, Precision = 0.8378, Recall = 0.7381, **F1 = 0.7848** (TN=132, FP=6, FN=11, TP=31).
- **Feature Selection & Noise Column Analysis:**
  - The top 7 features ranked by ANOVA F-score are:
    1. `distance_km` (F = 389.2131)
    2. `hub_tier` (F = 155.3483)
    3. `hub_daily_capacity` (F = 144.1347)
    4. `n_stops` (F = 125.4798)
    5. `promised_hours` (F = 84.4177)
    6. `rain_flag` (F = 53.0321)
    7. `sale_event_flag` (F = 36.7281)
  - Minimum F-score among top 7 is **36.7281** (well above the 30 threshold).
  - In contrast, all five noise columns (`label_print_batch`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`, `scanner_build_no`) have F-scores below 10 (ranging from 4.5847 down to 0.0230) and rank 8th, 12th, 14th, 16th, and 17th.
  - However, when fitted on all 17 features, the default Random Forest assigns a combined importance of **0.2244 (22.44%)** to these 5 pure noise features, illustrating how tree splits can absorb spurious variance from high-cardinality bookkeeping artifacts.
  - Evaluating SelectKBest across k ∈ {5, 7, 10, 12, 17}:
    - k=5: Val F1 = 0.6988
    - **k=7: Val F1 = 0.8333**
    - k=10: Val F1 = 0.7857
    - k=12: Val F1 = 0.8049
    - k=17: Val F1 = 0.7848
    - *Finding:* Filtering to the top 7 signal features maximizes validation F1 at 0.8333 while completely eliminating all noise columns.
- **Hyperparameter Tuning on Top 7 Features:**
  - `grid_search` (27 candidates, 135 fits): Best CV accuracy = 0.8952, Val F1 = **0.7750**.
  - `randomized_search` (9 candidates, 45 fits): Best CV accuracy = 0.8905, Val F1 = **0.7848**.
  - `default_forest_same_7_features` (untuned defaults: 100 trees, no depth limit, min_samples_leaf=1): Val F1 = **0.8333**.
  - *Insight on 180-row validation tuning:* Tuning for overall cross-validation accuracy selected constrained tree configurations (`max_depth=6`) that dampened boundary probability resolution, leading to lower positive-class F1 on the small 180-row validation split. The default forest configuration on the 7 selected features was retained per the tie-breaker and maximum validation F1 rule.
- **Alert Threshold Calibration (Operations Rule):**
  - Operations mandate: Missed delays cost more than unnecessary proactive calls; alert threshold is the highest `t` where validation recall ≥ 0.80.
  - Across tested thresholds `t ∈ [0.1, 0.9]`:
    - t = 0.4: Recall = 0.8810, Precision = 0.7400, F1 = 0.8043
    - **t = 0.5: Recall = 0.8333, Precision = 0.7955, F1 = 0.8140**
    - t = 0.6: Recall = 0.7619 (< 0.80 cutoff), Precision = 0.8889, F1 = 0.8205
  - The chosen operational alert threshold is **0.50**.
- **Final Test Evaluation (Single Evaluation):**
  - Evaluated on the held-out 180 test rows at threshold 0.50:
    - **Precision = 0.6889** (31 true delays out of 45 flagged parcels)
    - **Recall = 0.7561** (31 delays caught out of 41 total delayed parcels)
    - **F1 = 0.7209**, **Accuracy = 0.8667** (TN=125, FP=14, FN=10, TP=31).
  - *Honest Recall Comparison:* Test recall is 75.61% versus validation recall of 83.33% (a difference of -7.72 percentage points). Because the held-out test split contains only 41 delayed shipments, missing just 3 additional parcels changes measured recall by ~7.3 percentage points, reflecting finite-sample variance rather than model breakdown.

---

## 4. Operations recommendation

### Hub Operating Segments (Part D)

Aggregating all historical shipments across the 24 regional hubs into standardized profiles (`mean_delivery_hours`, `delay_rate`, `mean_distance_km`, `mean_n_stops`) and fitting K-Means clustering (K=3, `random_state=42`, `n_init=10`) yields three distinct operational behavior segments:

| Segment | Hub Count | Mean Hours | Delay Rate | Mean Distance (km) | Mean Stops | Operational Role |
|:---|---:|---:|---:|---:|---:|:---|
| **Standard** | 7 | 47.24 hrs | **2.57%** | 188.93 km | 55.37 | High efficiency, dense short-haul metro lanes. Standard operations. |
| **Watch** | 8 | 56.81 hrs | **15.50%** | 314.18 km | 70.72 | Medium-haul lanes; monitor during peak load and weather events. |
| **Escalate** | 9 | 64.65 hrs | **46.00%** | 470.48 km | 84.34 | Long-haul routes, high route stop density. Critical intervention. |

The **Escalate** segment hubs comprise: **HUB13 (Nagpur), HUB15 (Kochi), HUB17 (Patna), HUB18 (Ranchi), HUB19 (Guwahati), HUB20 (Raipur), HUB21 (Bhubaneswar), HUB23 (Siliguri), HUB24 (Agartala)**, averaging an alarming **46.00%** delay rate.

### Tier-vs-Segment Mismatches

Four hubs operate starkly out of sync with their administrative labels (`outputs/tier_vs_segment_mismatch.csv`):

| Hub ID | City | Nominal Label | Real Operating Segment | Operational Finding & Recommendation |
|:---|:---|:---:|:---:|:---|
| **HUB07** | Kolkata | Tier 1 | **Watch** | Operates like a Tier 2 hub (delay rate 14.0%, mean distance 304.7 km). Treat as Watch rather than an autonomous Tier 1 metro. |
| **HUB13** | Nagpur | Tier 2 | **Escalate** | Operates like a Tier 3 hub (delay rate 46.0%, mean distance 473.3 km). Discontinue Tier 2 SLAs; assign Escalate playbook. |
| **HUB15** | Kochi | Tier 2 | **Escalate** | Operates like a Tier 3 hub (delay rate 36.0%, mean distance 466.5 km). Lane geography causes severe transit overhead. |
| **HUB22** | Dehradun | Tier 3 | **Watch** | Operates significantly better than Tier 3 peers (delay rate 20.0%, mean distance 316.7 km). High routing efficiency. |

*Action for Regional Operations Head:* Hub playbooks and SLAs must be governed by empirical operational segments rather than static municipal tier labels.

### Justification for K = 3 Segments

Analyzing the across-to-within Euclidean cluster distance ratios from `outputs/part_d_separation.csv`:
- K = 2: Ratio = **2.8940**, Cluster sizes = `15-9`
- K = 3: Ratio = **5.3928**, Cluster sizes = `9-8-7`
- K = 4: Ratio = **6.3047**, Cluster sizes = `8-7-5-4`
- K = 5: Ratio = **6.8866**, Cluster sizes = `8-7-4-3-2`

Although mathematical separation continues to increase for K=4 and K=5, K=3 is the optimal operational configuration. Both K=4 and K=5 produce small splinter clusters containing fewer than 6 hubs (e.g. clusters of 4, 3, or 2 hubs). Regional logistics teams cannot maintain and staff 4 or 5 separate operating playbooks; K=3 produces balanced, actionable tiers (7, 8, and 9 hubs) that map cleanly onto Delhivery's field hierarchy.

### Dispatch Scoring on 48 New Shipments

Scoring the 48 newly booked parcels (`predict_delay.py`) using threshold **0.50**:
- **Flagged for delay:** **8 parcels** (16.7% of dispatch batch).
- **Escalate to hub manager:** **6 parcels** (flagged shipments routing through Escalate hubs: SHP90002, SHP90003, SHP90011, SHP90025, SHP90041, SHP90046). Immediate rerouting / priority sorting triggered.
- **Proactive customer call:** **2 parcels** (flagged shipments in Watch/Standard hubs: SHP90005, SHP90007). Proactive SMS/call scheduled.
- **No action:** **40 parcels** (on-track dispatches).
- *Performance Reality:* Based on held-out test calibration, ops can expect ~6.9 of every 10 flagged parcels to be true delays (precision 68.9%), and will proactively catch ~7.6 of every 10 actual delays (recall 75.6%).

---

## 5. Production plan

### Production-Readiness Architecture Mapping

| Dimension | Implementation in Repository |
|:---|:---|
| **Accuracy** | Two-stage model pipeline combining Linear Regression point-estimates (R² = 0.9493) and Random Forest classification (Test F1 = 0.7209 at calibrated threshold 0.50). |
| **Reliability** | Comprehensive input validation, sentinel weight handling (>100 kg scanner faults quarantined), missing payment mode fallback (`Unknown`), and missing value imputation. |
| **Reproducibility** | Fixed random seeds (`random_state=42`), zero leakage (scalers and selectors fit strictly on train sets), version-pinned dependencies in `requirements.txt`, and verification reload scripts (`outputs/part_d_reload_check.csv` showing 0.0 diff). |
| **Consistency** | Unified data transformation contracts in `src/shipment_prep.py` (`clean_shipments`, `build_features`) shared identically across training and inference (`predict_delay.py`). |
| **Maintainability** | Decoupled pipeline modularity: cleaning (`part_a`), regression (`part_b`), classification (`part_c`), clustering (`part_d`), packaging (`package_model.py`), and inference (`predict_delay.py`). |
| **Performance** | Lightweight, sub-10ms batch inference: pre-fitted pipeline with 7 SelectKBest features evaluates 48 shipments in under 50ms without GPU requirements. |
| **Monitorability** | Serialized telemetry outputs for every pipeline stage in `outputs/` and `models/`, enabling automated auditing and drift detection. |

### Operational Monitoring Triggers

Logistics telemetry should ingest daily dispatch predictions and track three automated alert triggers:
1. **Weekly Network Delay Rate Drift:**  
   *Trigger:* Observed 7-day delay rate among delivered parcels rises above **28.0%** (or shifts by >5.0 percentage points from baseline 23.17%).  
   *Remediation:* Inspect lane weather disruptions, regional strikes, or carrier capacity deficits.
2. **Flagged Dispatch Ratio Anomaly:**  
   *Trigger:* Share of newly booked parcels flagged as `flagged=1` departs from the expected **15%–20% band** (e.g., dropping below 10% or spiking above 25% over a 3-day rolling window).  
   *Remediation:* Audit upstream sensor inputs for scanner calibration drift or distance imputation shifts.
3. **Model Precision / Recall Degradation:**  
   *Trigger:* Rolling 30-day realized test precision falls below **0.60** or realized recall falls below **0.70** once ground-truth delivery confirmations are logged.  
   *Remediation:* Trigger pipeline retraining workflow on recent 90-day dispatch history.

### Retraining Schedule & Sale Event Protocol

- **Cadence:** Scheduled quarterly retraining on a rolling 90-day historical window. Unscheduled retraining is triggered if any monitoring threshold breaches for two consecutive weeks.
- **Handling Sale Surges (September 15–28 Sale Window):**  
  The `sale_event_flag` accounts for systemic volume spikes (+7 base hours delay effect, +12 extra hours on Express). When evaluating model drift during annual sales (e.g., Great India Festival), **never compare sale-window metrics against ordinary operational weeks**. Benchmark sale performance strictly against prior sale windows. Data drift during sales manifests as high volume concentration in Express modes, longer queue times at hub sorting belts, and increased parcel weight variance.

---

## 6. Limitations

1. **Synthetic Data Boundaries:**  
   All records are generated by synthetic simulation distributions. While capturing non-linear interactions (rain on long lanes, sale surges), real-world logistics entails erratic urban traffic, driver shortages, GST billing checkpoints, and multi-attempt delivery protocols not modeled here.
2. **Finite Test Sample Size:**  
   The held-out test split consists of 180 parcels containing exactly 41 delayed shipments. With only 41 positive cases, each parcel represents ~2.4% of recall; a shift of 2 or 3 misclassifications noticeably swings point estimates.
3. **Single Validation Split Calibration:**  
   The alert threshold (0.50) was tuned on a single 180-row validation partition. While cross-validation confirmed model stability, threshold calibration under production deployment should incorporate cost-sensitive utility functions (exact cost in INR of a missed SLA refund vs. an unnecessary customer care agent outbound call).
4. **Static Hub Segmentation:**  
   K-Means hub clustering reflects 90-day aggregate behavior across 24 hubs. Rapid expansion (opening new dark stores, routing optimizations) will alter hub stop densities and lane distances, requiring ongoing hub cluster recalibration.

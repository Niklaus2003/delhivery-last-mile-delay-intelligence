# Delhivery Capstone Project

This project builds a delay prediction and hub analytics pipeline for Delhivery shipment data. It covers data cleaning and exploratory data analysis, regression to estimate transit hours, classification to catch delayed shipments before they fail SLA, and clustering to segment hubs based on real performance.

## How to run

To run everything from scratch, install requirements and run the scripts in order:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

Note: If scikit-learn or pandas versions differ from requirements.txt, running package_model.py will regenerate models/delay_model.pkl so it matches your local environment.

## Data findings

After dropping 24 duplicate rows from raw_shipments.csv, we have 1,200 unique shipments across 24 hubs. The overall delay rate across the dataset is 23.17% (278 delayed shipments).

Here are the key takeaways from part A:

1. Service type delay rates:
   - Express (36 hr promise): 48.97% delayed (119 out of 243). This is the hardest SLA to hit so it has the highest delay rate.
   - Standard (60 hr promise): 20.56% delayed (132 out of 642).
   - Economy (96 hr promise): 8.57% delayed (27 out of 315).

2. Hub tiers:
   - Tier 1 delay rate is 4.00% (mean time ~48.6 hrs).
   - Tier 2 delay rate is 21.50% (mean time ~58.8 hrs).
   - Tier 3 delay rate is 44.00% (mean time ~63.5 hrs).
   There is an 11x jump in delay rate going from Tier 1 to Tier 3.

3. Distance bands:
   - 0-200 km: 2.19% delayed
   - 201-400 km: 13.68% delayed
   - 401-600 km: 43.64% delayed
   - 601+ km: 85.23% delayed
   Distance is clearly one of the biggest drivers of delivery delay.

4. Correlation with delivery_hours:
   The top three numeric columns most correlated with delivery_hours are:
   - promised_hours (+0.8372)
   - distance_km (+0.4013)
   - n_stops (+0.3247)

5. Declared value transformation:
   The raw declared value had a lot of skew (mean was INR 1,533.05 while median was INR 1,121.50). Taking log(declared_value_inr) brought the mean to 7.028 and median to 7.022, which makes the distribution much more balanced.

## Model results

### Regression (Part B)

We used an 840/180/180 train-val-test split. The diagnostic scaler fit on train gave a max absolute mean of ~0.0 on train and ~0.22 on test, confirming no leakage.

- Mean baseline on validation had an R2 of -0.0212 and RMSE of 18.86 hours.
- Tree depth diagnostic showed high bias at depth 1 (train RMSE 11.63, val RMSE 12.13, gap 0.50 hrs) and high variance with unlimited depth (train RMSE 0.0, val RMSE 7.84, gap 7.84 hrs). The best depth on validation was depth 6 (val RMSE 6.28 hrs).
- Candidate comparison on validation:
  - Linear Regression: val RMSE 4.8912, R2 0.9313
  - Gradient Boosting: val RMSE 4.9165, R2 0.9306
  - Random Forest: val RMSE 5.5095, R2 0.9128
  - Decision Tree (depth 6): val RMSE 6.2796, R2 0.8867
- Learning rate experiment for GB (100 trees):
  - lr 0.01: val R2 0.7127 (too slow, underfits)
  - lr 0.10: val R2 0.9306 (best balance)
  - lr 1.00: val R2 0.8429 (overshoots)
- 5-fold CV on train+val: GB had a slightly higher mean R2 (0.9353 vs 0.9261 for Linear Regression), but Linear Regression had the lower validation RMSE (4.8912), so Linear Regression was selected as final model.
- Final test evaluation for Linear Regression: R2 of 0.9493 and RMSE of 4.5235 hours.

### Classification (Part C)

We used a stratified 840/180/180 split with 195/42/41 delayed shipments.

- Baseline: Predicting all shipments on-time gives 76.67% accuracy but 0 recall, which is useless for operations.
- Logistic Regression had val F1 of 0.8046, Decision Tree (depth 6) had 0.7470, and default Random Forest had 0.7848.
- Feature selection:
  The 5 noise columns had very low F-scores (<10), while the real features scored much higher (distance_km was 389.21, hub_tier was 155.35). But when fitting Random Forest on all 17 features, the noise columns took up about 22.4% of total feature importance.
  Testing k in [5, 7, 10, 12, 17] showed k=7 had the highest validation F1 (0.8333).
- Tuning on 7 features:
  Grid search gave val F1 of 0.7750, randomized search gave 0.7848, while default forest parameters reached 0.8333. Tuning constrained max_depth to 6, which slightly hurt probability sharpness on the small validation set. So we kept default forest parameters on the 7 features.
- Alert threshold:
  Ops prefers catching delays over avoiding unnecessary calls. Scanning thresholds from 0.1 to 0.9, threshold 0.50 was the highest threshold that kept validation recall >= 0.80 (recall was 0.8333, precision 0.7955).
- Single test evaluation at threshold 0.50:
  Precision = 0.6889, Recall = 0.7561, F1 = 0.7209, Accuracy = 0.8667. Test recall dipped slightly from validation (75.6% vs 83.3%), but since test only has 41 delayed parcels, missing 3 more packages explains the drop.

## Operations recommendations

### Hub Segments

Clustering the 24 hubs using KMeans (K=3) on scaled metrics (mean delivery hours, delay rate, distance, stops) gave three clear operating groups:
- Standard (7 hubs): low delay rate (2.57%), average 47.2 hrs transit, short lanes (~189 km).
- Watch (8 hubs): medium delay rate (15.50%), average 56.8 hrs transit, lanes ~314 km.
- Escalate (9 hubs): high delay rate (46.00%), average 64.7 hrs transit, long haul (~470 km), 84 stops.

The 9 hubs in the Escalate segment are HUB13 (Nagpur), HUB15 (Kochi), HUB17 (Patna), HUB18 (Ranchi), HUB19 (Guwahati), HUB20 (Raipur), HUB21 (Bhubaneswar), HUB23 (Siliguri), and HUB24 (Agartala).

### Tier Mismatches

Four hubs do not match their nominal tier label:
- HUB07 (Kolkata): Labeled Tier 1, but acts like a Watch hub (14.0% delay rate).
- HUB13 (Nagpur): Labeled Tier 2, but acts like an Escalate hub (46.0% delay rate).
- HUB15 (Kochi): Labeled Tier 2, but acts like an Escalate hub (36.0% delay rate).
- HUB22 (Dehradun): Labeled Tier 3, but performs like a Watch hub (20.0% delay rate).

Ops should assign playbooks according to these empirical segments rather than relying on municipal tier names.

### Why K=3 was chosen

Across-to-within separation ratios were:
- K=2: 2.8940
- K=3: 5.3928
- K=4: 6.3047
- K=5: 6.8866

Even though K=4 and K=5 have slightly higher separation ratios, they create very small splinter clusters with only 2 or 3 hubs. Managing 4 or 5 different playbooks in regional operations is impractical, so K=3 was kept as the cleanest operational grouping.

### New Shipment Scoring (48 dispatches)

Running predict_delay.py on the 48 incoming shipments at threshold 0.50 gave:
- 8 shipments flagged as high risk of delay.
- 6 assigned to "Escalate to hub manager" (parcels in Escalate hubs: SHP90002, SHP90003, SHP90011, SHP90025, SHP90041, SHP90046).
- 2 assigned to "Proactive customer call" (parcels in Watch/Standard hubs: SHP90005, SHP90007).
- 40 shipments had no action needed.

## Production plan

1. System properties:
   - Leakage prevention: Scalers and feature selection are fit only on training splits.
   - Robust cleaning: Missing values are filled with saved training medians; sentinel weights (>100 kg) are replaced with NaN before imputation.
   - Fast inference: The 7-feature pipeline scores 48 dispatches in under 50ms on standard CPU.

2. Monitoring triggers:
   - Weekly network delay rate exceeds 28% (baseline is ~23.2%).
   - Percentage of flagged dispatches drops below 10% or rises above 25% over a 3-day window.
   - Rolling 30-day realized precision drops below 0.60 or recall drops below 0.70.

3. Retraining and sale events:
   - Standard quarterly retraining on a rolling 90-day window.
   - During the September sale surge window, expect Express volume to spike and queue times to increase. Do not compare sale weeks against normal operating weeks; compare against previous sale events instead.

## Limitations

1. The data is synthetic, so real-world complexities like driver shortages, weather road blocks, or failed delivery attempts are simplified.
2. The test split is relatively small (180 shipments with 41 delays), so a couple of misclassified parcels can noticeably affect recall and precision percentages.
3. Hub clustering is based on a single historical aggregate. As new routes or dark stores open, hub behavior will change and clusters will need to be re-run.

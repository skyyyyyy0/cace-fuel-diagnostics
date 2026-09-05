# CACE V1 Methodology

## 1. Method Overview

CACE V1 uses a physics-first residual-learning approach to estimate expected fuel consumption from sparse, event-driven vehicle telemetry.

```text
Physics Expected Fuel = f(Average RPM, Engine Load, RPM × Load)
Physics Residual      = Actual Fuel - Physics Expected Fuel
CACE Expected Fuel   = Physics Expected Fuel + Predicted Residual
```

The physics model provides a transparent baseline. Machine learning is used only to model the remaining error that can be explained by additional operating conditions.

## 2. Modeling Population

Four anonymized vehicles were evaluated during dataset construction. Three vehicles produced valid fuel windows and were retained for modeling.

| Dataset stage | Observations |
|---|---:|
| Chronological train | 254 |
| Chronological validation | 84 |
| Untouched final test | 87 |
| Total modeling observations | 425 |

`VEHICLE_01` was excluded because it did not produce valid windows under the fixed target-matching rules. The exclusion rule was based on data quality, not model performance.

## 3. Window and Target Construction

Engine Load observations are used as modeling anchors. Each anchor creates a fixed ±60-second window.

### Average RPM

- Use only real RPM observations inside the window.
- Require at least three valid RPM observations.
- Calculate the arithmetic mean as `avg_rpm`.
- Do not interpolate or forward-fill missing RPM values.

### Actual Fuel Used

Actual fuel is calculated from cumulative fuel observations near the window boundaries:

```text
Actual Fuel Used = Fuel at End Boundary - Fuel at Start Boundary
```

A target is retained only when:

- valid start and end fuel observations both exist;
- each observation is within 60 seconds of its boundary;
- start and end use different source observations;
- the fuel interval is positive;
- the cumulative-fuel difference is nonnegative.

These rules prevent synthetic targets and preserve the temporal order of the original telemetry.

## 4. Data Quality Controls

Before modeling, the pipeline checks:

- timestamp parsing and chronological order;
- duplicate observations;
- required signal availability;
- invalid or out-of-range values;
- cumulative-fuel resets and negative fuel differences;
- minimum RPM and fuel-boundary coverage;
- vehicle-level observation counts.

Raw proprietary telemetry and identifying fields are excluded from public outputs.

## 5. Physics Baseline

The baseline is an Ordinary Least Squares regression trained on the chronological training split.

| Feature | Role |
|---|---|
| `avg_rpm` | Average engine speed inside the window |
| `engine_load` | Observed load at the anchor timestamp |
| `rpm_load` | Interaction term: `avg_rpm × engine_load` |

The baseline equation is:

```text
Physics Expected Fuel = β0
                      + β1(RPM × Load)
                      + β2(Average RPM)
                      + β3(Engine Load)
```

The ML target is the remaining physics residual:

```text
Physics Residual = Actual Fuel - Physics Expected Fuel
```

## 6. ML Residual Correction

Random Forest and XGBoost residual models were compared. The expanded Random Forest was selected because it produced the strongest validation RMSE and R² among the tested correction models.

### Final ML Features

| Feature | Treatment |
|---|---|
| `engine_torque` | Observed operating feature |
| `avg_vehicle_speed` | Training-median imputation if missing |
| `avg_coolant_temperature` | Training-median imputation if missing |
| `avg_vehicle_speed_missing` | Binary missing-value indicator |
| `avg_coolant_temperature_missing` | Binary missing-value indicator |

All imputation values are fitted on the training split only and then applied unchanged to validation and test data.

### Random Forest Configuration

| Parameter | Value |
|---|---:|
| Number of trees | 300 |
| Maximum depth | 6 |
| Minimum samples per leaf | 5 |
| Random seed | 42 |

Depth and leaf-size constraints reduce overfitting on the small dataset.

## 7. Validation Design

### Chronological Validation

Each vehicle is sorted by time and divided approximately 60%/20%/20% into train, validation, and final-test partitions.

- Training data fits the baseline, preprocessing values, and ML model.
- Validation data supports feature and model selection.
- Final-test data remains untouched until the selected pipeline is evaluated.

### Leave-One-Vehicle-Out Validation

For each LOVO fold, one vehicle is excluded completely from training and used as the test vehicle. This evaluates whether CACE generalizes to an unseen vehicle rather than only to later observations from known vehicles.

Final model performance is reported with MAE, RMSE, and R². Cross-vehicle conclusions are based on LOVO out-of-sample predictions.

## 8. Interpretation and Business Metrics

SHAP values are calculated on the untouched final test to explain the ML residual correction. They explain how the correction model behaves; they do not measure causal effects on total fuel consumption.

Vehicle review priority is calculated from LOVO predictions:

```text
Aggregate Fuel Deviation (%) =
    (Sum of Actual Fuel - Sum of CACE Expected Fuel)
    / Sum of CACE Expected Fuel × 100
```

A positive value indicates actual fuel above the CACE expectation. It is treated as a diagnostic review signal, not confirmed fuel waste.

Operating-condition analysis uses the residual:

```text
CACE Residual = Actual Fuel - CACE Expected Fuel
```

Feature values are divided into data-relative Low, Medium, and High bands. A condition is considered a stronger review signal when positive mean residuals repeat across both LOVO and final-test views.

## 9. Methodological Limitations

- Event-driven signals are irregular and not perfectly synchronized.
- The dataset is small and contains only three modeling-eligible vehicles.
- ±60-second windows can overlap, so summed window fuel is not total fleet fuel.
- Median imputation preserves rows but cannot recreate unobserved signal behavior.
- SHAP and residual patterns are associative, not causal.
- Production use requires more vehicles, higher-frequency telemetry, and consistent improvement in unseen-vehicle validation.

## 10. Reproducibility Rules

- Keep raw data and identifying information outside the public repository.
- Fit preprocessing and models using training data only.
- Preserve chronological splits and fixed random seeds.
- Apply the same window and target rules to all vehicles and future datasets.
- Use untouched final-test data for final evaluation only.
- Use LOVO out-of-sample predictions for vehicle-level business analysis.

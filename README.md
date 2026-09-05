# CACE — Physics-First Fuel Consumption Diagnostics

CACE (Cero Adaptive Calibration Engine) is a physics-first machine-learning pipeline that estimates expected fuel consumption from sparse, event-driven vehicle telemetry.

The system combines a transparent Physics Baseline with a Random Forest residual correction model. Actual fuel is compared with condition-adjusted expected fuel to identify vehicles and operating conditions that may require further review.

> **Current status:** Validated diagnostic prototype. CACE V1 is not presented as a production fuel-efficiency score.

## Business Problem

Fuel consumption cannot be compared fairly across vehicles without considering operating conditions.

A vehicle may consume more fuel because it operated at different RPM, engine load, torque, speed, or temperature—not necessarily because it was inefficient.

CACE addresses this problem by calculating:

```text
CACE Expected Fuel = Physics Expected Fuel + ML Residual Correction
Fuel Deviation     = Actual Fuel - CACE Expected Fuel
```

A positive deviation indicates that actual fuel was above the model expectation. It is used as a diagnostic review signal, not as confirmed fuel waste.

## Solution Approach

CACE V1 follows a reusable pipeline:

1. Audit signal availability and reporting frequency.
2. Validate timestamps, duplicates, outliers, and cumulative-fuel behavior.
3. Create ±60-second windows around Engine Load observations.
4. Calculate Actual Fuel from valid cumulative-fuel boundary observations.
5. Build a transparent OLS Physics Baseline.
6. Train a Random Forest to predict the remaining physics residual.
7. Validate on later observations and completely unseen vehicles.
8. Interpret results with SHAP, vehicle deviation, and operating-condition analysis.

No RPM or fuel values are created through interpolation or forward filling.

## Model Architecture

| Component           | Method                 | Features                                                                              |
| ------------------- | ---------------------- | ------------------------------------------------------------------------------------- |
| Physics Baseline    | Ordinary Least Squares | Average RPM, Engine Load, RPM × Load                                                  |
| Residual Correction | Random Forest          | Engine Torque, Average Vehicle Speed, Average Coolant Temperature, missing indicators |
| Final Prediction    | Physics + ML           | Physics Expected Fuel + Predicted Residual                                            |

Random Forest and XGBoost residual models were compared. The expanded Random Forest was selected based on validation RMSE and R².

## Data and Validation

Four anonymized vehicles were evaluated during dataset construction. Three vehicles produced valid modeling windows under the fixed fuel-target rules.

| Dataset                     | Observations |
| --------------------------- | ------------ |
| Chronological train         | 254          |
| Chronological validation    | 84           |
| Untouched final test        | 87           |
| Total modeling observations | 425          |

Two validation methods were used:

- **Chronological split:** tests performance on later observations and reduces temporal leakage.
- **Leave-One-Vehicle-Out validation:** tests generalization to a vehicle completely excluded from training.

SHAP analysis uses the untouched final test. Vehicle ranking and business analysis use LOVO out-of-sample predictions.

## Key Results

### Untouched Final Test

| Metric | Physics Baseline | CACE V1         | Result                |
| ------ | ---------------- | --------------- | --------------------- |
| MAE    | 0.1727 L/window  | 0.1682 L/window | **2.64% improvement** |
| RMSE   | 0.2133 L/window  | 0.2116 L/window | **0.79% improvement** |
| R²     | 0.2009           | 0.2135          | +0.0126               |

CACE V1 improved all three final-test metrics, although the improvement was modest.

LOVO performance was mixed: the observation-weighted MAE increased slightly from 0.1844 to 0.1857 L/window. This indicates that generalization to unseen vehicles still requires improvement.

Actual versus expected fuel comparison

### Model Interpretation

Average Vehicle Speed was the most influential feature within the ML residual correction.

| Rank | Feature                     | SHAP importance |
| ---- | --------------------------- | --------------- |
| 1    | Average Vehicle Speed       | **54.4%**       |
| 2    | Average Coolant Temperature | **24.9%**       |
| 3    | Engine Torque               | **18.8%**       |

This ranking explains the ML correction only. It does not mean that vehicle speed is the largest causal driver of total fuel consumption.

Global SHAP feature importance

### Vehicle Review Priority

LOVO out-of-sample predictions identified:

- `VEHICLE_04`: **+8.72%** aggregate fuel deviation
- `VEHICLE_03`: **+5.48%** aggregate fuel deviation
- `VEHICLE_02`: **−12.02%** aggregate fuel deviation

`VEHICLE_04` is the first diagnostic review priority. These results do not represent confirmed fuel waste or realized savings.

### Operating-Condition Finding

Low RPM was the strongest positive residual pattern repeated across both LOVO and final-test analysis.

This makes Low RPM the first condition for additional investigation, but it does not prove that Low RPM directly causes excess fuel consumption.

## Repository Structure

```text
cace/                    Core analysis and modeling package
config/                  Portable pipeline configuration
docs/                    Project and methodology documentation
pipelines/               Reusable pipeline workflows
reports/interpretation/  Public tables and figures
tests/                   Validation and regression tests
run_pipeline.py          Main pipeline entry point
```

## Documentation

- [Project Overview](docs/project_overview.md)
- [CACE V1 Methodology](docs/cace_v1_methodology.md)
- [Window and Target Design](docs/cace_v1_window_design.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Business & Interpretation Report](docs/cace_v1_business_interpretation_report.md)

## Data Access and Reproducibility

Raw vehicle telemetry and row-level modeling artifacts are proprietary and are not included in this repository.

This repository provides the reusable pipeline code, methodology, anonymized public summaries, and interpretation outputs.

Public tables and figures are available under:

```text
reports/interpretation/
```

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- AWS S3
- Git and GitHub

## Limitations

- Source telemetry is sparse and event-driven.
- Only three vehicles met the final modeling requirements.
- The untouched final test contains 87 observations.
- ±60-second windows may overlap.
- SHAP explains model behavior but does not establish causality.
- LOVO performance was not consistently better than the Physics Baseline.
- More vehicles and higher-frequency telemetry are required for production readiness.

## Privacy and Public-Repository Safety

The public repository does not include:

- raw proprietary telemetry;
- credentials or environment secrets;
- VINs, device IDs, or customer identifiers;
- private source filenames;
- row-level private prediction files.

All public vehicle identifiers and analysis outputs are anonymized.

## Conclusion

CACE V1 demonstrates a complete physics-first ML workflow for sparse vehicle telemetry: reusable data processing, transparent baseline modeling, residual correction, chronological validation, unseen-vehicle testing, SHAP interpretation, and business-oriented diagnostics.

The current model provides a small final-test improvement and useful diagnostic signals, but cross-vehicle generalization remains mixed. CACE V1 should therefore be treated as a validated diagnostic prototype while additional data and model improvements are developed.

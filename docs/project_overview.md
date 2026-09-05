# CACE: Physics-First Fuel Consumption Diagnostics

## Project Summary

CACE (Cero Adaptive Calibration Engine) is a physics-first machine-learning system that estimates expected fuel consumption from sparse, event-driven vehicle telemetry. It compares actual fuel use with expected fuel use to identify vehicles and operating conditions that may require further review.

The project was designed as a reusable diagnostic pipeline rather than a one-time analysis. New telemetry can be processed with the same validation, windowing, feature-engineering, modeling, and reporting rules.

**Current status:** CACE V1 is a validated diagnostic prototype, not a production fuel-efficiency score.

## Business Problem

Raw fuel consumption cannot be compared fairly across vehicles because fuel use changes with operating conditions such as RPM, engine load, torque, speed, and temperature. A vehicle may consume more fuel simply because it operated under more demanding conditions.

CACE addresses this problem by estimating a condition-adjusted fuel expectation and measuring the difference between actual and expected fuel use.

```text
CACE Expected Fuel = Physics Expected Fuel + ML Residual Correction
Fuel Deviation     = Actual Fuel - CACE Expected Fuel
```

A positive deviation indicates actual fuel above the model expectation. It is a diagnostic review signal, not confirmed fuel waste.

## Project Objectives

- Build a reusable pipeline for irregular GeoTab telemetry.
- Create reliable modeling observations without interpolation or forward filling.
- Establish a transparent physics-based fuel baseline.
- Use machine learning only to correct the remaining residual error.
- Validate performance on future observations and unseen vehicles.
- Translate model outputs into explainable vehicle and operating-condition insights.

## Data and Main Challenge

The source data consists of anonymized commercial-vehicle telemetry, including cumulative fuel, engine RPM, engine load, engine torque, vehicle speed, and coolant temperature.

The main challenge is that the signals are recorded at irregular, event-driven timestamps. Different parameters do not arrive at the same time, so each modeling row must be constructed from nearby real observations while preserving temporal order and preventing leakage.

The public repository excludes proprietary raw telemetry, credentials, VINs, device IDs, and customer identifiers.

## CACE V1 Approach

1. Validate timestamps, duplicates, signal ranges, missing periods, and cumulative-fuel resets.
2. Use Engine Load observations as modeling anchors.
3. Create ±60-second windows around each anchor.
4. Calculate average RPM from real observations with minimum coverage requirements.
5. Calculate actual fuel from valid cumulative-fuel observations near the window boundaries.
6. Estimate Physics Expected Fuel using average RPM, Engine Load, and their interaction.
7. Train a Random Forest to predict the remaining physics residual using torque, speed, coolant temperature, and missing-value indicators.
8. Combine both estimates to produce CACE Expected Fuel.

No synthetic signal values are created through interpolation or forward filling.

## Validation Strategy

The model uses two complementary validation methods:

| Validation                                      | Purpose                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| Chronological train/validation/final-test split | Tests performance on later observations and reduces temporal leakage    |
| Leave-One-Vehicle-Out validation                | Tests whether the model generalizes to a vehicle excluded from training |

SHAP interpretation is calculated only on the untouched final test. Vehicle rankings and business analysis use LOVO out-of-sample predictions.

## Key Results

| Result                                        |                          Finding |
| --------------------------------------------- | -------------------------------: |
| Final-test observations                       |                               87 |
| MAE improvement over Physics Baseline         |                        **2.64%** |
| RMSE improvement over Physics Baseline        |                        **0.79%** |
| R²                                            |                **0.201 → 0.214** |
| Most important ML correction feature          | **Average Vehicle Speed: 54.4%** |
| Highest positive vehicle deviation            |           **VEHICLE_04: +8.72%** |
| Strongest repeated operating-condition signal |                      **Low RPM** |

The final-test improvement was modest, and overall LOVO performance did not improve consistently across vehicles. The current evidence supports diagnostic use and continued development rather than production deployment.

## Business Use

CACE V1 can support:

- prioritizing vehicles for engineering or operational review;
- identifying operating conditions associated with repeated positive fuel deviations;
- separating raw fuel use from condition-adjusted model expectations;
- tracking whether future model versions improve beyond a transparent physics baseline.

The output should not be used to claim confirmed fuel waste, realized savings, or emissions reduction without additional operational validation.

## Main Limitations

- Sparse and event-driven telemetry limits signal alignment.
- The untouched final test contains 87 observations.
- LOVO validation covers three vehicles that met the modeling requirements.
- Modeling windows may overlap, so aggregated window fuel is not total fleet fuel.
- SHAP explains model behavior but does not prove causality.
- Additional vehicles and higher-frequency telemetry are needed for production readiness.

## Documentation

- [CACE V1 Methodology](cace_v1_methodology.md)
- [CACE V1 Window Design](cace_v1_window_design.md)
- [Data Dictionary](data_dictionary.md)
- [CACE V1 Business & Interpretation Report](cace_v1_business_interpretation_report.md)

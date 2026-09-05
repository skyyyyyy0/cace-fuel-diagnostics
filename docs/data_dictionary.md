# CACE V1 Data Dictionary

This document defines the main fields needed to understand the public CACE V1 workflow. Timestamps use UTC, fuel is measured in liters, and vehicle identifiers are anonymized.

## Core Modeling Fields

| Column                                | Unit     | Definition                                               |
| ------------------------------------- | -------- | -------------------------------------------------------- |
| `vehicle`                             | —        | Anonymized vehicle ID, such as `VEHICLE_02`              |
| `anchor_time_utc`                     | UTC      | Engine Load timestamp used as the window anchor          |
| `window_start_utc` / `window_end_utc` | UTC      | Boundaries of the ±60-second modeling window             |
| `avg_rpm`                             | rpm      | Average of valid RPM observations inside the window      |
| `engine_load`                         | %        | Engine Load observed at the anchor timestamp             |
| `rpm_load`                            | rpm × %  | Interaction feature: `avg_rpm × engine_load`             |
| `engine_torque`                       | %        | Torque observation matched to the anchor                 |
| `avg_vehicle_speed`                   | km/h     | Average valid vehicle speed inside the window            |
| `avg_coolant_temperature`             | °C       | Average valid coolant temperature inside the window      |
| `actual_fuel_used`                    | L/window | Primary target: cumulative fuel end minus fuel start     |
| `derived_fuel_rate`                   | L/hour   | Supporting diagnostic fuel rate                          |
| `split`                               | —        | Chronological `train`, `validation`, or `test` partition |

Speed and coolant-temperature missing values are imputed with training-set medians. Binary missing indicators preserve whether each value was originally absent.

## Model Output Fields

| Column                  | Unit     | Definition                                         |
| ----------------------- | -------- | -------------------------------------------------- |
| `physics_expected_fuel` | L/window | Expected fuel from the OLS Physics Baseline        |
| `physics_residual`      | L/window | `actual_fuel_used - physics_expected_fuel`         |
| `predicted_residual`    | L/window | Residual correction predicted by the Random Forest |
| `cace_expected_fuel`    | L/window | `physics_expected_fuel + predicted_residual`       |
| `cace_residual`         | L/window | `actual_fuel_used - cace_expected_fuel`            |

## Vehicle-Level Metrics

| Metric                         | Definition                                                            |
| ------------------------------ | --------------------------------------------------------------------- |
| Aggregate deviation (%)        | `(Σ Actual Fuel - Σ CACE Expected Fuel) / Σ CACE Expected Fuel × 100` |
| Positive-deviation window rate | Percentage of windows where Actual Fuel exceeds CACE Expected Fuel    |
| Review-priority rank           | Descending rank among vehicles with positive aggregate deviation      |

Positive deviation is a diagnostic review signal, not confirmed fuel waste.

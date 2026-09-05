# CACE V1 — Outlier Detection and Data Quality Rules

## 1. Purpose

This document describes the outlier detection and data quality framework used in CACE V1.

The source telematics data is event-driven rather than recorded at fixed time intervals. Reporting behavior therefore varies across signals and operating conditions.

For this reason, CACE does not rely on a single statistical rule to identify outliers. Data quality rules are defined using a combination of:

- Observed signal distributions
- Physical and diagnostic meaning
- Expected value ranges
- Signal-specific behavior
- Conservative validation thresholds

Potential anomalies are flagged for review rather than automatically removed from the source data.

## 2. Data Quality Strategy

The CACE V1 data quality process follows several principles:

- Profile each signal before defining thresholds.
- Use physical and diagnostic context in addition to statistical distributions.
- Preserve original observations.
- Flag suspicious observations instead of automatically deleting them.
- Handle different signal types using appropriate validation rules.
- Keep QC thresholds separate from processing code.
- Reevaluate rules as additional data and signals become available.

The overall approach is:

`Profile → Define Rules → Detect → Flag → Review → Decide`

## 3. Signal Profiling

Before defining QC thresholds, signal distributions were profiled at both vehicle and fleet levels.

The following statistics were calculated:

- Minimum
- 1st percentile (P1)
- 5th percentile (P5)
- 25th percentile (P25)
- Median
- Mean
- 75th percentile (P75)
- 95th percentile (P95)
- 99th percentile (P99)
- Maximum
- Standard deviation

Percentiles were reviewed together with minimum and maximum values to identify extreme observations without allowing a small number of unusual records to define the normal operating range.

Percentiles such as P99 were not used directly as outlier cutoffs. A value in the upper tail may represent a valid high-load or high-RPM operating condition rather than a data error.

Final QC boundaries were therefore defined using both the observed distributions and the expected behavior of each signal.

## 4. Initial QC Rules

### Engine Speed (RPM)

The observed RPM distribution was relatively consistent across the available vehicles, with the upper tail remaining well below the initial QC boundary.

Initial rule:

`0 <= RPM <= 3,000`

The upper threshold provides a conservative margin above the observed operating range.

RPM values of zero are retained because they may represent valid stopped or engine-off conditions.

### Engine Load

Engine Load is represented as a percentage.

Initial rule:

`0 <= Engine Load <= 100`

Values outside this range are flagged for review.

### Engine Torque

The currently observed torque values were within the expected percentage range.

However, depending on the underlying diagnostic definition, valid negative percentage torque values may be possible.

Initial rule:

`-125 <= Engine Torque <= 125`

A wider boundary is therefore used rather than treating every negative observation as invalid.

### Vehicle Speed

The observed vehicle-speed distribution remained comfortably below the initial upper boundary.

Initial rule:

`0 <= Vehicle Speed <= 160`

A speed of zero is retained because it is required for stopped and idle-state analysis.

### Engine Coolant Temperature

Signal profiling identified a small cluster of unusually high coolant-temperature observations that was clearly separated from the normal operating distribution.

Initial rule:

`-40°C <= Coolant Temperature <= 130°C`

Values outside this range are flagged as potential sensor or data-quality anomalies.

They are not automatically removed from the source data.

### Outside Air Temperature

Some relatively high outside-air temperature readings were observed across multiple vehicles.

Because sensor placement and vehicle-generated heat may influence the measurement, a wider validation range is used.

Initial rule:

`-40°C <= Outside Air Temperature <= 70°C`

### DPF Active Regeneration Status

The available regeneration-status signal behaves as a binary indicator.

Initial rule:

`Allowed values = {0, 1}`

Values outside the expected set are flagged for review.

## 5. Signals Without Fixed Range Rules

Not every signal is validated using a fixed minimum and maximum.

Applying the same QC method to every signal could incorrectly classify valid observations.

### Fuel Rate

A fixed Fuel Rate range is not currently applied.

The observed values require additional confirmation of the diagnostic definition and unit before a reliable physical boundary can be established.

Until that validation is complete, the pipeline retains the observations without applying an arbitrary range threshold.

### DPF Soot Load

The observed DPF Soot Load distribution contains valid-looking observations above 100.

A simple rule such as:

`Soot Load > 100 = Outlier`

could therefore remove meaningful operating conditions.

A fixed range rule is deferred until the diagnostic definition is sufficiently validated.

### Total Fuel Used

Total Fuel Used is a cumulative counter rather than a conventional continuous feature.

A fixed upper boundary is therefore not appropriate because the value naturally increases as the vehicle operates.

Instead, consecutive observations are evaluated using:

`Fuel Delta = Current Total Fuel Used - Previous Total Fuel Used`

A negative delta is flagged:

`Fuel Delta < 0`

because a cumulative fuel counter would normally be expected not to decrease.

## 6. Automated Quality Checks

The CACE data pipeline currently performs automated checks for:

1. Invalid timestamps
2. Missing values
3. Duplicate observations
4. Signal range violations
5. Invalid categorical values
6. Cumulative fuel counter decreases

Quality checks are performed after CACE signal filtering and before modeling dataset generation.

The source observations are preserved throughout the process.

## 7. Initial Findings

The initial QC analysis identified several useful data-quality patterns.

### Coolant Temperature

A small cluster of coolant-temperature observations exceeded the configured operating boundary.

The observations occurred within a short time period rather than being distributed throughout the dataset.

These records were flagged as:

`range_outlier`

The observations remain preserved for traceability and downstream review.

### Cumulative Fuel Counter

A small number of minor decreases were detected in the Total Fuel Used counter.

The magnitude of the decreases was small, so they were not automatically classified as full counter resets.

Possible explanations include:

- Measurement precision
- Telemetry corrections
- Counter behavior
- Data transmission effects

These observations were flagged as:

`fuel_counter_decrease`

rather than automatically corrected.

## 8. Flag-First Outlier Handling

CACE uses a flag-first approach to data quality.

The QC layer does not automatically delete suspicious observations.

The workflow is:

Raw Observation  
→ Signal Filtering  
→ QC Rule Evaluation  
→ Issue Flag  
→ Preserve Original Observation  
→ Downstream Treatment

This approach is important for vehicle telemetry because an extreme observation may represent either:

- A sensor or data-quality problem, or
- A rare but valid operating condition.

Automatically removing all extreme values could remove useful information from the modeling dataset.

## 9. Modeling Treatment

Flagged observations are handled according to the type of issue.

Examples include:

- Clear sensor or parsing errors may be excluded.
- Valid extreme operating conditions may be retained.
- Signals with uncertain definitions may require additional validation.
- Invalid target intervals are excluded from model training.

This distinction is particularly important for the fuel-consumption target.

Actual fuel consumption is derived from the cumulative fuel counter:

`Actual Fuel Used = Fuel_end - Fuel_start`

If a counter anomaly produces an invalid negative fuel delta, the corresponding target interval is flagged and excluded from model training rather than modifying the original counter values.

## 10. Configuration-Driven QC

Quality rules are maintained separately from the processing code in:

`config/quality_rules.yaml`

For example:

```yaml
rpm:
  type: range
  min: 0
  max: 3000
  action: flag

engine_load:
  type: range
  min: 0
  max: 100
  action: flag

coolant_temperature:
  type: range
  min: -40
  max: 130
  action: flag
```

Keeping the rules in configuration allows thresholds to evolve without changing the core pipeline logic.

The same QC code can therefore be reused as new vehicles and observations are added.

## 11. Updating Quality Rules

As the dataset grows, quality rules are reevaluated using the following process:

```text
New Data
    ↓
Signal Profiling
    ↓
Existing Rule Validation
    ↓
New Signal Review
    ↓
Threshold Review
    ↓
QC Configuration Update
```

New signals are not automatically assigned thresholds.

Their distribution, diagnostic definition, unit, and physical meaning are reviewed before they are added to the automated QC framework.

## 12. Reproducibility and Data Privacy

The public repository contains the reusable QC methodology and processing code but does not expose proprietary fleet data.

The following are excluded from the public repository:

- Raw telematics data
- VINs
- Device identifiers
- Customer information
- Credentials
- Internal source filenames
- Private vehicle-level QC reports

Public examples use anonymized identifiers and non-sensitive sample outputs where needed.

Private QC reports are excluded through `.gitignore` and can be regenerated internally from the source data.

## 13. Summary

The CACE V1 data quality framework combines statistical profiling with signal-specific validation rules.

The main design decisions are:

- Signal distributions are reviewed before thresholds are defined.
- Physical and diagnostic context is considered alongside statistical behavior.
- Different signal types use different validation strategies.
- Potential anomalies are flagged rather than automatically deleted.
- Cumulative counters are validated using temporal behavior rather than fixed ranges.
- QC rules are configuration-driven and reusable.
- Raw and proprietary data remain separate from the public project repository.

This framework provides a reproducible validation layer between raw telematics ingestion and CACE modeling dataset generation.

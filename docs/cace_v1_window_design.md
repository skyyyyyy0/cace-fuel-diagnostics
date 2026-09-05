# CACE V1 Window & Target Design

## 1. Objective

The purpose of CACE V1 is to build a modeling dataset that can estimate expected fuel consumption under observed vehicle operating conditions and compare it with actual fuel consumption.

The GeoTab engine signals used in this project are not reported at the same timestamps or at a consistent frequency. Some signals are relatively frequent, while others are sparse or event-driven.

Because of this, the raw signals cannot be directly combined into a standard time-based modeling dataset.

To address this issue, Engine Load is used as the anchor signal, and surrounding RPM, Engine Torque, and Total Fuel Used observations are grouped into a fixed time window.

This analysis compares several candidate window sizes and defines the final window and minimum coverage rules for CACE V1.

## 2. Data Characteristics and Problem Definition

The initial assumption was that when relatively sparse signals such as Engine Load and Engine Torque were reported, the other core engine signals would also be available at approximately the same timestamp.

After reviewing the actual reporting patterns, however, the signals were found to have different reporting frequencies.

The main observations were:

- Engine Load and Engine Torque are relatively sparse and are generally well aligned with each other.
- RPM is reported more frequently than Load and Torque, but it does not always occur at the exact Load timestamp.
- Fuel Rate is available for some observations, but it is not used directly as the primary actual fuel target in CACE V1.
- Total Fuel Used is a cumulative signal and is not reported frequently enough to align directly with every Engine Load observation.

This last point is particularly important because Actual Fuel Used is calculated from the cumulative Total Fuel Used signal:

`Actual Fuel Used = Total Fuel Used(end) - Total Fuel Used(start)`

If the window is too short, there may not be enough Total Fuel Used observations around the window boundaries to calculate a reliable fuel target.

On the other hand, using a very large window increases target availability but makes it more difficult for a single Engine Load observation to represent the operating conditions across the entire period.

For this reason, the final window was selected based on the trade-off between:

**Temporal Alignment and Target Availability**

## 3. Anchor Signal

CACE V1 uses each observed Engine Load timestamp as an anchor.

A fixed time window is created around each anchor, and the surrounding signals are used to construct one modeling observation.

Conceptually:

```text
                    Engine Load
                        ●
                        │
          Window Start  │  Window End
                |-------●-------|
```

Within this window:

- RPM observations are aggregated.
- Engine Torque availability is checked.
- Total Fuel Used observations near the start and end boundaries are matched.
- Actual Fuel Used is calculated from the cumulative fuel difference.

Only actual Engine Load observations are used as anchors.

No interpolated or forward-filled Engine Load values are created.

---

## 4. Window Candidate Analysis

The following candidate windows were evaluated using the core engine signals:

| Candidate | Half Window | Total Window |
| --------- | ----------: | -----------: |
| ±30 sec   |      30 sec |        1 min |
| ±45 sec   |      45 sec |      1.5 min |
| ±60 sec   |      60 sec |        2 min |
| ±90 sec   |      90 sec |        3 min |

The primary signals considered during the window design were:

- Engine Load
- Engine Torque
- RPM
- Total Fuel Used
- Fuel Rate

Other signals such as Vehicle Speed, Temperature, and DPF-related parameters may be used later as model features, but they were not used as the primary criteria for selecting the base CACE V1 window.

---

## 5. Window Analysis Results

RPM and Engine Torque could generally be matched around the Engine Load anchors.

The main constraint was the reporting frequency of Total Fuel Used.

As the window size increased, more Engine Load anchors could be associated with usable Total Fuel Used observations.

The shorter ±30-second and ±45-second windows frequently did not provide enough cumulative fuel observations to create a meaningful start-to-end fuel interval. In some cases, both boundaries were matched to the same cumulative fuel observation, resulting in no usable fuel interval.

The ±90-second window improved target coverage, but it also increased the period represented by a single Engine Load observation to three minutes.

The ±60-second window provided a better balance. It improved Actual Fuel target availability compared with the shorter windows while keeping the total observation period limited to two minutes.

Based on this trade-off, the ±60-second window was selected for CACE V1.

---

## 6. Total Fuel Used Boundary Matching

Total Fuel Used is not always available exactly at the desired window start and end timestamps.

For each Engine Load anchor, the desired boundaries are:

- `Anchor Time - 60 seconds`
- `Anchor Time + 60 seconds`

The nearest actual Total Fuel Used observation is identified for each boundary.

However, simply selecting the nearest observation without a limit can result in fuel observations being matched from timestamps that are too far away from the intended window.

To control this, a Fuel Boundary Tolerance was evaluated.

The following tolerance levels were compared:

- 30 seconds
- 60 seconds
- 90 seconds
- 120 seconds

Increasing the tolerance retained more observations but reduced temporal alignment with the intended window.

A 60-second boundary tolerance provided a reasonable balance between retaining usable observations and keeping the fuel measurements close to the intended boundaries.

Therefore, CACE V1 requires both fuel boundary matches to be within 60 seconds of their respective target timestamps.

---

## 7. RPM Minimum Coverage

RPM is reported multiple times within many of the candidate windows.

CACE V1 uses the mean RPM within each window as the representative RPM feature.

However, using only one RPM observation to represent an entire two-minute period may not provide enough information about the operating condition.

The following minimum RPM coverage rules were evaluated:

- `RPM observations >= 1`
- `RPM observations >= 3`
- `RPM observations >= 5`

Requiring only one observation retained the most data but provided relatively weak temporal representation.

Requiring five observations provided stronger coverage but removed a substantial number of otherwise usable windows.

For CACE V1, a minimum of three actual RPM observations was selected as a practical balance between observation quality and dataset coverage.

Therefore:

`Minimum RPM Observations = 3`

---

## 8. Final CACE V1 Window Rule

The final CACE V1 modeling window is defined as follows.

### Anchor

- Use an actual Engine Load observation as the anchor.
- Do not create interpolated Engine Load values.

### Window

- ±60 seconds around the Engine Load anchor.
- Total window length: 120 seconds.

### RPM

- Use actual RPM observations within the window.
- Require at least three RPM observations.
- Calculate the window mean as the representative RPM value.

### Engine Torque

- Require an actual Engine Torque observation.
- Do not generate Torque values through interpolation or forward filling.

### Total Fuel Used

- Match the nearest actual cumulative fuel observation to the desired window start.
- Match the nearest actual cumulative fuel observation to the desired window end.
- Each matched observation must be within 60 seconds of its intended boundary.
- The start and end matches must be different observations.

### Actual Fuel Used

Calculate:

`Actual Fuel Used = Fuel_end - Fuel_start`

A valid target must satisfy:

- `Actual Fuel Used >= 0`
- `Actual Fuel Interval > 0`
- Fuel observations must be in the correct chronological order.

### Data Integrity

The CACE V1 window generation process does not use:

- Interpolation
- Forward filling
- Artificial Engine Load observations
- Artificial Total Fuel Used observations

The modeling dataset is constructed from actual reported observations whenever possible.

---

## 9. Final Pipeline Rule

Each CACE V1 modeling observation is generated using the following process:

```text
Engine Load Observation
        ↓
Create ±60 sec Window
        ↓
Check RPM Observations >= 3
        ↓
Calculate Mean RPM
        ↓
Check Engine Torque Availability
        ↓
Match Total Fuel Used Near Start / End
        ↓
Check Fuel Boundary Distance <= 60 sec
        ↓
Check Start / End Are Different Observations
        ↓
Actual Fuel = Fuel_end - Fuel_start
        ↓
Check Actual Fuel >= 0 and Fuel Interval > 0
        ↓
Valid CACE V1 Observation
```

---

## 10. Design Rationale

The final two-minute window was not selected simply to maximize the number of usable observations.

A shorter window provides stronger temporal alignment with the Engine Load anchor, but the current Total Fuel Used reporting frequency does not provide enough observations to generate reliable fuel targets for many anchors.

A longer window improves target availability but increases the risk that a single Engine Load observation does not adequately represent the operating conditions across the entire period.

The final CACE V1 design therefore uses:

**±60-second Window + 60-second Fuel Boundary Tolerance + Minimum 3 RPM Observations**

This rule was selected to balance temporal alignment, target availability, and observation quality.

---

## 11. Current Limitations

The current CACE V1 window design has several limitations.

First, Total Fuel Used is not fully synchronized with Engine Load, which means that a valid Actual Fuel target cannot be generated for every Engine Load observation.

Second, applying minimum coverage rules reduces the number of observations available for modeling. This is an intentional trade-off to improve the reliability of the final dataset.

Third, CACE V1 is designed around the current event-driven and sparse reporting characteristics of the available telematics data.

If the reporting frequency of RPM, Engine Load, Engine Torque, or fuel-related signals improves in the future, shorter and more precisely synchronized windows can be evaluated as part of a later CACE version.

---

## 12. CACE V1 Design Decision

The final CACE V1 Window & Target Design is:

| Component                | V1 Rule                     |
| ------------------------ | --------------------------- |
| Anchor Signal            | Engine Load                 |
| Window                   | ±60 seconds                 |
| Total Window             | 2 minutes                   |
| RPM Aggregation          | Mean                        |
| Minimum RPM Observations | 3                           |
| Engine Torque            | Actual observation required |
| Fuel Boundary Tolerance  | 60 seconds                  |
| Actual Fuel Target       | `Fuel_end - Fuel_start`     |
| Interpolation            | Not used                    |
| Forward Filling          | Not used                    |

These rules are treated as fixed CACE V1 pipeline rules.

As additional raw data becomes available, new observations can be generated using the same methodology without changing the underlying V1 window and target definitions.

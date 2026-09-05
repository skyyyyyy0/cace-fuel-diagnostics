# Audits the candidate CACE V1 modeling dataset before finalization.
#
# This step does not rebuild windows, rematch signals, or recalculate targets.
# It reviews only observations that already passed the Phase 4 window rules
# and summarizes feature distributions, missing values, potential outliers,
# and vehicle-level coverage before the final V1 dataset is created.
#
# Potential outliers are flagged for review rather than automatically removed.
# This keeps data-cleaning decisions separate from the validated window and
# target-generation logic.

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "CACE_Dataset_v1.0_candidate_private.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling_dataset"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DISTRIBUTION_REPORT = (
    REPORT_DIR
    / "cace_v1_distribution_summary_private.csv"
)

OUTLIER_REPORT = (
    REPORT_DIR
    / "cace_v1_outlier_review_private.csv"
)

VEHICLE_REPORT = (
    REPORT_DIR
    / "cace_v1_vehicle_observation_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "audit_v1_modeling_dataset.log"
)


MODEL_COLUMNS = [
    "avg_rpm",
    "engine_load",
    "rpm_load",
    "engine_torque",
    "actual_fuel_used",
    "derived_fuel_rate",
    "actual_fuel_interval_sec",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


def load_candidate_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Candidate dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    logger.info(
        "Loaded candidate dataset | rows=%d",
        len(df),
    )

    return df


def select_valid_observations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_flags = [
        "window_rule_valid",
        "core_quality_flag",
    ]

    missing_flags = [
        column
        for column in required_flags
        if column not in df.columns
    ]

    if missing_flags:
        raise ValueError(
            "Missing validation columns: "
            f"{missing_flags}"
        )

    valid_df = df.loc[
        df["window_rule_valid"].eq(True)
        & df["core_quality_flag"].eq(True)
    ].copy()

    logger.info(
        "Selected valid observations | rows=%d",
        len(valid_df),
    )

    return valid_df


def build_distribution_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in MODEL_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        rows.append(
            {
                "feature": column,
                "count": int(values.count()),
                "missing_count": int(values.isna().sum()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p01": values.quantile(0.01),
                "p05": values.quantile(0.05),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
            }
        )

    return pd.DataFrame(rows)


def add_iqr_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    outlier_columns = [
        "avg_rpm",
        "engine_load",
        "engine_torque",
        "actual_fuel_used",
        "derived_fuel_rate",
    ]

    flag_columns = []

    for column in outlier_columns:
        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        flag_column = (
            f"{column}_iqr_flag"
        )

        result[flag_column] = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        flag_columns.append(
            flag_column
        )

    result["any_iqr_flag"] = (
        result[flag_columns]
        .any(axis=1)
    )

    return result


def build_vehicle_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    total_rows = len(df)

    summary = (
        df.groupby(
            "vehicle",
            as_index=False,
        )
        .agg(
            observation_count=(
                "anchor_time_utc",
                "size",
            ),
            median_avg_rpm=(
                "avg_rpm",
                "median",
            ),
            median_engine_load=(
                "engine_load",
                "median",
            ),
            median_engine_torque=(
                "engine_torque",
                "median",
            ),
            median_actual_fuel_used=(
                "actual_fuel_used",
                "median",
            ),
        )
    )

    summary[
        "observation_share_pct"
    ] = (
        summary["observation_count"]
        / total_rows
        * 100
    )

    return summary


def main():
    logger.info(
        "Starting CACE V1 modeling dataset audit"
    )

    candidate_df = (
        load_candidate_dataset()
    )

    valid_df = (
        select_valid_observations(
            candidate_df
        )
    )

    if valid_df.empty:
        raise RuntimeError(
            "No valid CACE V1 observations available"
        )

    distribution_summary = (
        build_distribution_summary(
            valid_df
        )
    )

    audited_df = (
        add_iqr_flags(
            valid_df
        )
    )

    outlier_review = (
        audited_df.loc[
            audited_df[
                "any_iqr_flag"
            ]
        ]
        .copy()
    )

    vehicle_summary = (
        build_vehicle_summary(
            valid_df
        )
    )

    distribution_summary.to_csv(
        DISTRIBUTION_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    outlier_review.to_csv(
        OUTLIER_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    vehicle_summary.to_csv(
        VEHICLE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Audit complete | valid_rows=%d | "
        "iqr_flagged_rows=%d",
        len(valid_df),
        len(outlier_review),
    )

    logger.info(
        "Saved distribution report: %s",
        DISTRIBUTION_REPORT.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved outlier review: %s",
        OUTLIER_REPORT.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved vehicle summary: %s",
        VEHICLE_REPORT.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
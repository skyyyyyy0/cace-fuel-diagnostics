# Finalizes the CACE V1 modeling dataset after the candidate dataset has passed
# window, target, feature, and data-quality review. This step does not rebuild
# modeling windows or recalculate features. It selects validated observations,
# preserves statistical outlier flags for later residual analysis, and records
# the final dataset version and summary metadata.

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "1.0"

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / f"CACE_Dataset_v{DATASET_VERSION}_candidate_private.csv"
)

OUTLIER_REVIEW_FILE = (
    PROJECT_ROOT
    / "reports"
    / "modeling_dataset"
    / "cace_v1_outlier_review_private.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "metadata"
    / "datasets"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

FINAL_DATASET_FILE = (
    OUTPUT_DIR
    / f"CACE_Dataset_v{DATASET_VERSION}_private.csv"
)

DATASET_METADATA_FILE = (
    METADATA_DIR
    / f"CACE_Dataset_v{DATASET_VERSION}_private.yaml"
)

LOG_FILE = (
    LOG_DIR
    / "finalize_v1_modeling_dataset.log"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


FINAL_COLUMNS = [
    "vehicle",
    "anchor_time_utc",
    "window_start_utc",
    "window_end_utc",
    "avg_rpm",
    "rpm_observation_count",
    "engine_load",
    "engine_torque",
    "torque_distance_sec",
    "rpm_load",
    "fuel_start",
    "fuel_end",
    "fuel_start_time_utc",
    "fuel_end_time_utc",
    "fuel_start_boundary_gap_sec",
    "fuel_end_boundary_gap_sec",
    "actual_fuel_interval_sec",
    "actual_fuel_used",
    "derived_fuel_rate",
]


def load_candidate_dataset() -> pd.DataFrame:
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate dataset not found: {CANDIDATE_FILE}"
        )

    df = pd.read_csv(
        CANDIDATE_FILE,
        low_memory=False,
    )

    logger.info(
        "Loaded candidate dataset | rows=%d",
        len(df),
    )

    return df


def select_final_observations(
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
            "Candidate dataset is missing validation flags: "
            f"{missing_flags}"
        )

    final_df = df.loc[
        df["window_rule_valid"].eq(True)
        & df["core_quality_flag"].eq(True)
    ].copy()

    return final_df


def add_outlier_review_flag(
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    result = final_df.copy()

    result["statistical_outlier_flag"] = False

    if not OUTLIER_REVIEW_FILE.exists():
        logger.warning(
            "Outlier review file not found; "
            "statistical_outlier_flag will remain False"
        )
        return result

    outliers = pd.read_csv(
        OUTLIER_REVIEW_FILE,
        low_memory=False,
    )

    if outliers.empty:
        return result

    key_columns = [
        "vehicle",
        "anchor_time_utc",
    ]

    missing_columns = [
        column
        for column in key_columns
        if column not in outliers.columns
    ]

    if missing_columns:
        raise ValueError(
            "Outlier review file is missing key columns: "
            f"{missing_columns}"
        )

    outlier_keys = set(
        zip(
            outliers["vehicle"].astype(str),
            outliers["anchor_time_utc"].astype(str),
        )
    )

    row_keys = list(
        zip(
            result["vehicle"].astype(str),
            result["anchor_time_utc"].astype(str),
        )
    )

    result["statistical_outlier_flag"] = [
        key in outlier_keys
        for key in row_keys
    ]

    return result


def validate_final_dataset(
    df: pd.DataFrame,
) -> None:
    required_model_columns = [
        "avg_rpm",
        "engine_load",
        "rpm_load",
        "engine_torque",
        "actual_fuel_used",
        "derived_fuel_rate",
    ]

    missing_columns = [
        column
        for column in required_model_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Final dataset is missing required columns: "
            f"{missing_columns}"
        )

    missing_values = (
        df[required_model_columns]
        .isna()
        .sum()
    )

    if missing_values.any():
        invalid = (
            missing_values[
                missing_values > 0
            ]
            .to_dict()
        )

        raise ValueError(
            "Final dataset contains missing modeling values: "
            f"{invalid}"
        )

    if (
        df["actual_fuel_used"] < 0
    ).any():
        raise ValueError(
            "Negative actual_fuel_used found "
            "in final dataset"
        )

    if (
        df["actual_fuel_interval_sec"] <= 0
    ).any():
        raise ValueError(
            "Non-positive fuel interval found "
            "in final dataset"
        )

    duplicate_count = (
        df.duplicated(
            subset=[
                "vehicle",
                "anchor_time_utc",
            ]
        )
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate vehicle/anchor observations found: "
            f"{duplicate_count}"
        )


def prepare_final_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        column
        for column in FINAL_COLUMNS
        if column in df.columns
    ]

    columns.append(
        "statistical_outlier_flag"
    )

    return (
        df[columns]
        .sort_values(
            [
                "vehicle",
                "anchor_time_utc",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_dataset_metadata(
    df: pd.DataFrame,
) -> dict:
    vehicle_counts = (
        df.groupby(
            "vehicle"
        )
        .size()
        .to_dict()
    )

    return {
        "dataset_name": "CACE_Dataset",
        "dataset_version": DATASET_VERSION,
        "status": "final",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "observation_count": int(
            len(df)
        ),
        "vehicle_count": int(
            df["vehicle"].nunique()
        ),
        "vehicle_observation_counts": {
            str(vehicle): int(count)
            for vehicle, count
            in vehicle_counts.items()
        },
        "primary_target": (
            "actual_fuel_used"
        ),
        "supporting_metric": (
            "derived_fuel_rate"
        ),
        "core_features": [
            "avg_rpm",
            "engine_load",
            "rpm_load",
            "engine_torque",
        ],
        "window_rule": {
            "anchor_signal": "engine_load",
            "half_window_sec": 60,
            "total_window_sec": 120,
            "minimum_rpm_observations": 3,
            "fuel_boundary_tolerance_sec": 60,
            "interpolation": False,
            "forward_fill": False,
        },
        "outlier_policy": {
            "method": "IQR review flag",
            "automatic_removal": False,
            "flagged_observation_count": int(
                df[
                    "statistical_outlier_flag"
                ].sum()
            ),
        },
    }


def save_dataset(
    df: pd.DataFrame,
) -> None:
    df.to_csv(
        FINAL_DATASET_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved final dataset: %s",
        FINAL_DATASET_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def save_metadata(
    metadata: dict,
) -> None:
    with DATASET_METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            metadata,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info(
        "Saved dataset metadata: %s",
        DATASET_METADATA_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def main():
    logger.info(
        "Starting CACE V1 dataset finalization"
    )

    candidate_df = (
        load_candidate_dataset()
    )

    final_df = (
        select_final_observations(
            candidate_df
        )
    )

    final_df = (
        add_outlier_review_flag(
            final_df
        )
    )

    validate_final_dataset(
        final_df
    )

    final_df = (
        prepare_final_columns(
            final_df
        )
    )

    metadata = (
        build_dataset_metadata(
            final_df
        )
    )

    save_dataset(
        final_df
    )

    save_metadata(
        metadata
    )

    logger.info(
        "CACE V1 dataset finalized | "
        "rows=%d | vehicles=%d | "
        "outlier_flags=%d",
        len(final_df),
        final_df[
            "vehicle"
        ].nunique(),
        int(
            final_df[
                "statistical_outlier_flag"
            ].sum()
        ),
    )


if __name__ == "__main__":
    main()
# Reviews coverage of candidate ML residual features in the finalized CACE V1 dataset.
# This step does not change the Physics Baseline or dataset rows.
# It only identifies which additional operating variables are currently available
# and whether their coverage is sufficient for ML residual correction.

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "CACE_Dataset_v1.0_private.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "ml_residual"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

REPORT_FILE = (
    REPORT_DIR
    / "ml_feature_coverage_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "audit_ml_feature_coverage.log"
)


CANDIDATE_FEATURES = {
    "engine_torque": [
        "engine_torque",
    ],
    "vehicle_speed": [
        "vehicle_speed",
        "avg_speed",
        "speed",
    ],
    "idle_ratio": [
        "idle_ratio",
    ],
    "outside_temperature": [
        "outside_temperature",
        "outside_temp",
    ],
    "coolant_temperature": [
        "coolant_temperature",
        "coolant_temp",
    ],
    "dpf_soot_load": [
        "dpf_soot_load",
        "soot_load",
    ],
}


REPORT_DIR.mkdir(
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


def find_column(
    columns,
    aliases,
):
    for column in aliases:
        if column in columns:
            return column

    return None


def calculate_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    total_rows = len(df)

    for feature_name, aliases in CANDIDATE_FEATURES.items():
        matched_column = find_column(
            df.columns,
            aliases,
        )

        if matched_column is None:
            rows.append(
                {
                    "feature": feature_name,
                    "matched_column": None,
                    "available_rows": 0,
                    "missing_rows": total_rows,
                    "coverage_percent": 0.0,
                    "status": "not_in_dataset",
                }
            )

            continue

        available_rows = int(
            df[
                matched_column
            ].notna().sum()
        )

        missing_rows = (
            total_rows
            - available_rows
        )

        coverage_percent = (
            available_rows
            / total_rows
            * 100
        )

        if coverage_percent >= 90:
            status = "strong_candidate"
        elif coverage_percent >= 70:
            status = "review_candidate"
        else:
            status = "low_coverage"

        rows.append(
            {
                "feature": feature_name,
                "matched_column": matched_column,
                "available_rows": available_rows,
                "missing_rows": missing_rows,
                "coverage_percent": round(
                    coverage_percent,
                    2,
                ),
                "status": status,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    logger.info(
        "Starting ML feature coverage audit"
    )

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"CACE V1 dataset not found: {DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE,
        low_memory=False,
    )

    logger.info(
        "Loaded CACE V1 dataset | rows=%d | columns=%d",
        len(df),
        len(df.columns),
    )

    logger.info(
        "Dataset columns: %s",
        list(df.columns),
    )

    coverage = calculate_coverage(
        df
    )

    coverage.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "\n%s",
        coverage.to_string(
            index=False
        ),
    )

    logger.info(
        "Saved ML feature coverage report: %s",
        REPORT_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "ML feature coverage audit complete"
    )


if __name__ == "__main__":
    main()
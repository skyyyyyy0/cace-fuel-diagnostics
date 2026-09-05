# Analyzes patterns in the CACE V1 physics residuals before ML correction.
# The analysis combines the finalized modeling features with the Physics Baseline
# residuals and checks whether the remaining prediction error is associated with
# available operating-condition features. Target-derived metrics are excluded
# to avoid leakage into the ML correction model.

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "1.0"
MODEL_VERSION = "1.0"

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / f"CACE_Dataset_v{DATASET_VERSION}_private.csv"
)

RESIDUAL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "physics_baseline"
    / f"physics_baseline_residuals_v{MODEL_VERSION}_private.csv"
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

CORRELATION_FILE = (
    REPORT_DIR
    / "residual_feature_correlation_private.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "residual_pattern_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "analyze_residual_patterns.log"
)


FEATURES = [
    "avg_rpm",
    "engine_load",
    "rpm_load",
    "engine_torque",
]

RESIDUAL = "physics_residual"

KEY_COLUMNS = [
    "vehicle",
    "anchor_time_utc",
]


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {DATASET_FILE}"
        )

    if not RESIDUAL_FILE.exists():
        raise FileNotFoundError(
            f"Residual file not found: {RESIDUAL_FILE}"
        )

    dataset = pd.read_csv(
        DATASET_FILE,
        low_memory=False,
    )

    residuals = pd.read_csv(
        RESIDUAL_FILE,
        low_memory=False,
    )

    return dataset, residuals


def merge_modeling_data(
    dataset: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    required_dataset_columns = (
        KEY_COLUMNS
        + FEATURES
    )

    missing_dataset_columns = [
        column
        for column in required_dataset_columns
        if column not in dataset.columns
    ]

    if missing_dataset_columns:
        raise ValueError(
            "Modeling dataset is missing required columns: "
            f"{missing_dataset_columns}"
        )

    required_residual_columns = (
        KEY_COLUMNS
        + [RESIDUAL]
    )

    missing_residual_columns = [
        column
        for column in required_residual_columns
        if column not in residuals.columns
    ]

    if missing_residual_columns:
        raise ValueError(
            "Residual file is missing required columns: "
            f"{missing_residual_columns}"
        )

    merged = dataset[
        required_dataset_columns
    ].merge(
        residuals[
            required_residual_columns
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    logger.info(
        "Merged residual analysis dataset | rows=%d",
        len(merged),
    )

    return merged


def build_correlation_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in FEATURES:
        pearson = df[
            [feature, RESIDUAL]
        ].corr(
            method="pearson"
        ).iloc[
            0,
            1,
        ]

        spearman = df[
            [feature, RESIDUAL]
        ].corr(
            method="spearman"
        ).iloc[
            0,
            1,
        ]

        rows.append(
            {
                "feature": feature,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "absolute_pearson": abs(
                    pearson
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "absolute_pearson",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def build_residual_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        df.groupby(
            "vehicle",
            as_index=False,
        )
        .agg(
            observation_count=(
                RESIDUAL,
                "size",
            ),
            mean_residual=(
                RESIDUAL,
                "mean",
            ),
            median_residual=(
                RESIDUAL,
                "median",
            ),
            residual_std=(
                RESIDUAL,
                "std",
            ),
            mean_engine_torque=(
                "engine_torque",
                "mean",
            ),
        )
    )

    return summary


def save_reports(
    correlation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    correlation_df.to_csv(
        CORRELATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved residual correlation report: %s",
        CORRELATION_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved residual pattern summary: %s",
        SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def main():
    logger.info(
        "Starting CACE residual pattern analysis"
    )

    dataset, residuals = (
        load_data()
    )

    analysis_df = (
        merge_modeling_data(
            dataset,
            residuals,
        )
    )

    correlation_df = (
        build_correlation_report(
            analysis_df
        )
    )

    summary_df = (
        build_residual_summary(
            analysis_df
        )
    )

    save_reports(
        correlation_df,
        summary_df,
    )

    logger.info(
        "Residual pattern analysis complete"
    )


if __name__ == "__main__":
    main()
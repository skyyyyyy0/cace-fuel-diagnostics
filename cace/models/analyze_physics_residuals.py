# Analyzes prediction error from the CACE V1 physics baseline.
# The script calculates residuals and baseline performance metrics from the
# previously generated Actual and Physics Expected Fuel values. It also
# summarizes model behavior by vehicle without retraining the baseline model.

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_VERSION = "1.0"

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "physics_baseline"
    / f"physics_baseline_predictions_v{MODEL_VERSION}_private.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "physics_baseline"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

RESIDUAL_FILE = (
    REPORT_DIR
    / f"physics_baseline_residuals_v{MODEL_VERSION}_private.csv"
)

VEHICLE_SUMMARY_FILE = (
    REPORT_DIR
    / f"physics_baseline_vehicle_summary_v{MODEL_VERSION}_private.csv"
)

METRICS_FILE = (
    REPORT_DIR
    / f"physics_baseline_metrics_v{MODEL_VERSION}_private.json"
)

LOG_FILE = (
    LOG_DIR
    / "analyze_physics_residuals.log"
)


ACTUAL = "actual_fuel_used"
EXPECTED = "physics_expected_fuel"
RESIDUAL = "physics_residual"


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


def load_predictions() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required_columns = [
        "vehicle",
        ACTUAL,
        EXPECTED,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction file is missing required columns: "
            f"{missing_columns}"
        )

    if df[
        [ACTUAL, EXPECTED]
    ].isna().any().any():
        raise ValueError(
            "Missing Actual or Expected Fuel values found"
        )

    logger.info(
        "Loaded physics predictions | rows=%d",
        len(df),
    )

    return df


def calculate_residuals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result[RESIDUAL] = (
        result[ACTUAL]
        - result[EXPECTED]
    )

    result[
        "absolute_error"
    ] = (
        result[RESIDUAL]
        .abs()
    )

    result[
        "squared_error"
    ] = (
        result[RESIDUAL]
        ** 2
    )

    return result


def calculate_metrics(
    df: pd.DataFrame,
) -> dict:
    actual = df[
        ACTUAL
    ]

    expected = df[
        EXPECTED
    ]

    mae = mean_absolute_error(
        actual,
        expected,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            expected,
        )
    )

    r2 = r2_score(
        actual,
        expected,
    )

    negative_prediction_count = int(
        (
            expected < 0
        ).sum()
    )

    return {
        "model_name": "CACE Physics Baseline",
        "model_version": MODEL_VERSION,
        "evaluation_type": "in_sample_baseline_fit",
        "observation_count": int(
            len(df)
        ),
        "mae": float(
            mae
        ),
        "rmse": float(
            rmse
        ),
        "r2": float(
            r2
        ),
        "mean_residual": float(
            df[RESIDUAL].mean()
        ),
        "median_residual": float(
            df[RESIDUAL].median()
        ),
        "negative_prediction_count": (
            negative_prediction_count
        ),
    }


def build_vehicle_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for vehicle, group in df.groupby(
        "vehicle"
    ):
        actual = group[
            ACTUAL
        ]

        expected = group[
            EXPECTED
        ]

        rows.append(
            {
                "vehicle": vehicle,
                "observation_count": int(
                    len(group)
                ),
                "actual_fuel_mean": float(
                    actual.mean()
                ),
                "expected_fuel_mean": float(
                    expected.mean()
                ),
                "mean_residual": float(
                    group[
                        RESIDUAL
                    ].mean()
                ),
                "mae": float(
                    mean_absolute_error(
                        actual,
                        expected,
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            expected,
                        )
                    )
                ),
                "r2": float(
                    r2_score(
                        actual,
                        expected,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_outputs(
    residual_df: pd.DataFrame,
    vehicle_summary: pd.DataFrame,
    metrics: dict,
) -> None:
    residual_df.to_csv(
        RESIDUAL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    vehicle_summary.to_csv(
        VEHICLE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    logger.info(
        "Saved residual output: %s",
        RESIDUAL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved vehicle summary: %s",
        VEHICLE_SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved baseline metrics: %s",
        METRICS_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def main():
    logger.info(
        "Starting CACE Physics Baseline residual analysis"
    )

    df = (
        load_predictions()
    )

    residual_df = (
        calculate_residuals(
            df
        )
    )

    metrics = (
        calculate_metrics(
            residual_df
        )
    )

    vehicle_summary = (
        build_vehicle_summary(
            residual_df
        )
    )

    logger.info(
        "Baseline performance | "
        "MAE=%.4f | RMSE=%.4f | R2=%.4f",
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )

    logger.info(
        "Residual summary | "
        "mean=%.6f | negative_predictions=%d",
        metrics["mean_residual"],
        metrics["negative_prediction_count"],
    )

    save_outputs(
        residual_df=residual_df,
        vehicle_summary=vehicle_summary,
        metrics=metrics,
    )

    logger.info(
        "CACE Physics Baseline residual analysis complete"
    )


if __name__ == "__main__":
    main()
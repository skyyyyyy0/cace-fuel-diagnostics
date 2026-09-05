# Evaluates the selected CACE V1 model on the untouched Test split.
# The selected Expanded Random Forest is loaded without retraining.
# Final predicted residuals and CACE expected fuel are generated here.

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "1.0"

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ml_residual"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v2"
    / "splits"
)

PHYSICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "physics_baseline"
    / "predictions"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "final_cace"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)


MODEL_FILE = (
    MODEL_DIR
    / f"random_forest_expanded_residual_v{MODEL_VERSION}_private.joblib"
)

TEST_FEATURE_FILE = (
    DATA_DIR
    / f"CACE_ML_Expanded_test_v{MODEL_VERSION}_private.csv"
)

TEST_PHYSICS_FILE = (
    PHYSICS_DIR
    / f"physics_test_v{MODEL_VERSION}_private.csv"
)

FINAL_PREDICTIONS_FILE = (
    REPORT_DIR
    / f"CACE_v{MODEL_VERSION}_final_test_predictions_private.csv"
)

FINAL_METRICS_FILE = (
    REPORT_DIR
    / f"CACE_v{MODEL_VERSION}_final_test_metrics_private.json"
)

LOG_FILE = (
    LOG_DIR
    / "evaluate_final_cace_v1.log"
)


FEATURES = [
    "engine_torque",
    "avg_vehicle_speed",
    "avg_coolant_temperature",
    "avg_vehicle_speed_missing",
    "avg_coolant_temperature_missing",
]

PHYSICS_EXPECTED = "physics_expected_fuel"
TARGET = "actual_fuel_used"
PREDICTED_RESIDUAL = "predicted_residual"
CACE_EXPECTED = "cace_expected_fuel"


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


def load_csv(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    df["anchor_time_utc"] = pd.to_datetime(
        df["anchor_time_utc"],
        utc=True,
        errors="coerce",
    )

    return df


def load_test_dataset() -> pd.DataFrame:
    feature_df = load_csv(
        TEST_FEATURE_FILE
    )

    physics_df = load_csv(
        TEST_PHYSICS_FILE
    )

    physics_columns = physics_df[
        [
            "vehicle",
            "anchor_time_utc",
            PHYSICS_EXPECTED,
            "physics_residual",
        ]
    ].copy()

    test_df = feature_df.merge(
        physics_columns,
        on=[
            "vehicle",
            "anchor_time_utc",
        ],
        how="inner",
        validate="one_to_one",
    )

    required_columns = (
        FEATURES
        + [
            TARGET,
            PHYSICS_EXPECTED,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    missing_values = (
        test_df[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"Missing values found in required columns: "
            f"{missing_values}"
        )

    return test_df


def calculate_metrics(
    actual: pd.Series,
    predicted,
) -> dict:
    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "rmse": float(
            mean_squared_error(
                actual,
                predicted,
            )
            ** 0.5
        ),
        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
    }


def main():
    logger.info(
        "Starting final CACE V1 Test evaluation"
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Selected model not found: {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    test_df = load_test_dataset()

    logger.info(
        "Final Test observations: %d",
        len(test_df),
    )

    logger.info(
        "Selected model: Expanded Random Forest"
    )

    logger.info(
        "Features: %s",
        FEATURES,
    )

    predicted_residual = model.predict(
        test_df[
            FEATURES
        ]
    )

    test_result = test_df.copy()

    test_result[
        PREDICTED_RESIDUAL
    ] = predicted_residual

    test_result[
        CACE_EXPECTED
    ] = (
        test_result[
            PHYSICS_EXPECTED
        ]
        + test_result[
            PREDICTED_RESIDUAL
        ]
    )

    physics_metrics = calculate_metrics(
        test_result[
            TARGET
        ],
        test_result[
            PHYSICS_EXPECTED
        ],
    )

    cace_metrics = calculate_metrics(
        test_result[
            TARGET
        ],
        test_result[
            CACE_EXPECTED
        ],
    )

    mae_improvement_percent = (
        (
            physics_metrics["mae"]
            - cace_metrics["mae"]
        )
        / physics_metrics["mae"]
        * 100
    )

    rmse_improvement_percent = (
        (
            physics_metrics["rmse"]
            - cace_metrics["rmse"]
        )
        / physics_metrics["rmse"]
        * 100
    )

    metrics = {
        "model": "CACE V1",
        "model_version": MODEL_VERSION,
        "ml_correction_model": "Expanded Random Forest",
        "evaluation_scope": "final_test_only",
        "test_observations": len(
            test_result
        ),
        "features": FEATURES,
        "physics_only": physics_metrics,
        "cace_final": cace_metrics,
        "mae_improvement_percent": float(
            mae_improvement_percent
        ),
        "rmse_improvement_percent": float(
            rmse_improvement_percent
        ),
    }

    test_result.to_csv(
        FINAL_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with FINAL_METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    logger.info(
        "Physics Test MAE: %.6f",
        physics_metrics["mae"],
    )

    logger.info(
        "CACE Test MAE: %.6f",
        cace_metrics["mae"],
    )

    logger.info(
        "Physics Test RMSE: %.6f",
        physics_metrics["rmse"],
    )

    logger.info(
        "CACE Test RMSE: %.6f",
        cace_metrics["rmse"],
    )

    logger.info(
        "Physics Test R2: %.6f",
        physics_metrics["r2"],
    )

    logger.info(
        "CACE Test R2: %.6f",
        cace_metrics["r2"],
    )

    logger.info(
        "MAE improvement: %.2f%%",
        mae_improvement_percent,
    )

    logger.info(
        "RMSE improvement: %.2f%%",
        rmse_improvement_percent,
    )

    logger.info(
        "Final CACE V1 Test evaluation complete"
    )


if __name__ == "__main__":
    main()
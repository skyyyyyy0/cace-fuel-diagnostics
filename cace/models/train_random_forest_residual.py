# Trains the CACE V1 Random Forest residual correction model.
# The same training and validation logic supports both Core and Expanded
# feature experiments. Test data remains untouched.

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "1.0"

# Change only this value to run a different experiment.
EXPERIMENT = "expanded"


PHYSICS_DATA_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "physics_baseline"
    / "predictions"
)

EXPANDED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v2"
    / "splits"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ml_residual"
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


TRAIN_PHYSICS_FILE = (
    PHYSICS_DATA_DIR
    / f"physics_train_v{MODEL_VERSION}_private.csv"
)

VALIDATION_PHYSICS_FILE = (
    PHYSICS_DATA_DIR
    / f"physics_validation_v{MODEL_VERSION}_private.csv"
)

TRAIN_EXPANDED_FILE = (
    EXPANDED_DATA_DIR
    / f"CACE_ML_Expanded_train_v{MODEL_VERSION}_private.csv"
)

VALIDATION_EXPANDED_FILE = (
    EXPANDED_DATA_DIR
    / f"CACE_ML_Expanded_validation_v{MODEL_VERSION}_private.csv"
)


CORE_FEATURES = [
    "avg_rpm",
    "engine_load",
    "rpm_load",
    "engine_torque",
]

EXPANDED_FEATURES = [
    "engine_torque",
    "avg_vehicle_speed",
    "avg_coolant_temperature",
    "avg_vehicle_speed_missing",
    "avg_coolant_temperature_missing",
]


TARGET = "physics_residual"
PHYSICS_EXPECTED = "physics_expected_fuel"
CACE_EXPECTED = "cace_expected_fuel"


if EXPERIMENT == "core":
    FEATURES = CORE_FEATURES
elif EXPERIMENT == "expanded":
    FEATURES = EXPANDED_FEATURES
else:
    raise ValueError(
        f"Unsupported experiment: {EXPERIMENT}"
    )


MODEL_FILE = (
    MODEL_DIR
    / f"random_forest_{EXPERIMENT}_residual_v{MODEL_VERSION}_private.joblib"
)

METRICS_FILE = (
    REPORT_DIR
    / f"random_forest_{EXPERIMENT}_validation_metrics_v{MODEL_VERSION}_private.json"
)

PREDICTIONS_FILE = (
    REPORT_DIR
    / f"random_forest_{EXPERIMENT}_validation_predictions_v{MODEL_VERSION}_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / f"train_random_forest_{EXPERIMENT}_residual.log"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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


def load_dataset(
    physics_file: Path,
    expanded_file: Path | None = None,
) -> pd.DataFrame:
    physics_df = load_csv(
        physics_file
    )

    if EXPERIMENT == "core":
        df = physics_df.copy()

    else:
        expanded_df = load_csv(
            expanded_file
        )

        physics_columns = physics_df[
            [
                "vehicle",
                "anchor_time_utc",
                PHYSICS_EXPECTED,
                TARGET,
            ]
        ].copy()

        df = expanded_df.merge(
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
            "actual_fuel_used",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    missing_values = (
        df[
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

    return df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


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
        "Starting Random Forest residual correction | experiment=%s",
        EXPERIMENT,
    )

    train_df = load_dataset(
        TRAIN_PHYSICS_FILE,
        TRAIN_EXPANDED_FILE
        if EXPERIMENT == "expanded"
        else None,
    )

    validation_df = load_dataset(
        VALIDATION_PHYSICS_FILE,
        VALIDATION_EXPANDED_FILE
        if EXPERIMENT == "expanded"
        else None,
    )

    logger.info(
        "Dataset rows | train=%d | validation=%d",
        len(train_df),
        len(validation_df),
    )

    logger.info(
        "Features: %s",
        FEATURES,
    )

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_validation = validation_df[
        FEATURES
    ]

    model = train_model(
        X_train,
        y_train,
    )

    predicted_residual = model.predict(
        X_validation
    )

    validation_result = (
        validation_df.copy()
    )

    validation_result[
        "rf_predicted_residual"
    ] = predicted_residual

    validation_result[
        CACE_EXPECTED
    ] = (
        validation_result[
            PHYSICS_EXPECTED
        ]
        + validation_result[
            "rf_predicted_residual"
        ]
    )

    physics_metrics = calculate_metrics(
        validation_result[
            "actual_fuel_used"
        ],
        validation_result[
            PHYSICS_EXPECTED
        ],
    )

    rf_metrics = calculate_metrics(
        validation_result[
            "actual_fuel_used"
        ],
        validation_result[
            CACE_EXPECTED
        ],
    )

    mae_improvement_percent = (
        (
            physics_metrics["mae"]
            - rf_metrics["mae"]
        )
        / physics_metrics["mae"]
        * 100
    )

    rmse_improvement_percent = (
        (
            physics_metrics["rmse"]
            - rf_metrics["rmse"]
        )
        / physics_metrics["rmse"]
        * 100
    )

    feature_importance = {
        feature: float(importance)
        for feature, importance
        in zip(
            FEATURES,
            model.feature_importances_,
        )
    }

    metrics = {
        "model": "Random Forest Residual Correction",
        "model_version": MODEL_VERSION,
        "experiment": EXPERIMENT,
        "training_observations": len(
            train_df
        ),
        "validation_observations": len(
            validation_df
        ),
        "features": FEATURES,
        "parameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "physics_only": physics_metrics,
        "physics_plus_random_forest": rf_metrics,
        "mae_improvement_percent": float(
            mae_improvement_percent
        ),
        "rmse_improvement_percent": float(
            rmse_improvement_percent
        ),
        "feature_importance": feature_importance,
    }

    joblib.dump(
        model,
        MODEL_FILE,
    )

    validation_result.to_csv(
        PREDICTIONS_FILE,
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
        "Physics-only Validation MAE: %.6f",
        physics_metrics["mae"],
    )

    logger.info(
        "Physics + RF Validation MAE: %.6f",
        rf_metrics["mae"],
    )

    logger.info(
        "Physics-only Validation RMSE: %.6f",
        physics_metrics["rmse"],
    )

    logger.info(
        "Physics + RF Validation RMSE: %.6f",
        rf_metrics["rmse"],
    )

    logger.info(
        "Physics-only Validation R2: %.6f",
        physics_metrics["r2"],
    )

    logger.info(
        "Physics + RF Validation R2: %.6f",
        rf_metrics["r2"],
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
        "Feature importance: %s",
        feature_importance,
    )

    logger.info(
        "Random Forest residual correction training complete"
    )


if __name__ == "__main__":
    main()
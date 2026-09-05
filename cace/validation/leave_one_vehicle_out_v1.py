# Runs Leave-One-Vehicle-Out validation for CACE V1.
# Each vehicle is held out completely while Physics and the selected Expanded
# Random Forest are trained only on the remaining vehicles.

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from cace.models.train_physics_baseline import (
    FEATURES as PHYSICS_FEATURES,
    TARGET,
    prepare_training_data,
    train_model as train_physics_model,
)

from cace.models.train_random_forest_residual import (
    FEATURES as RF_FEATURES,
    train_model as train_rf_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "1.0"

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v2"
    / f"CACE_ML_Features_v{MODEL_VERSION}_private.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "lovo"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

SUMMARY_FILE = (
    REPORT_DIR
    / f"CACE_v{MODEL_VERSION}_lovo_summary_private.csv"
)

DETAIL_FILE = (
    REPORT_DIR
    / f"CACE_v{MODEL_VERSION}_lovo_predictions_private.csv"
)

METRICS_FILE = (
    REPORT_DIR
    / f"CACE_v{MODEL_VERSION}_lovo_metrics_private.json"
)

LOG_FILE = (
    LOG_DIR
    / "leave_one_vehicle_out_v1.log"
)


PHYSICS_EXPECTED = "physics_expected_fuel"
PHYSICS_RESIDUAL = "physics_residual"
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


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    required_columns = (
        PHYSICS_FEATURES
        + [
            "engine_torque",
            "avg_vehicle_speed",
            "avg_coolant_temperature",
            TARGET,
            "vehicle",
            "anchor_time_utc",
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

    logger.info(
        "Loaded LOVO dataset | rows=%d | vehicles=%d",
        len(df),
        df["vehicle"].nunique(),
    )

    return df


def calculate_metrics(
    actual,
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


def prepare_ml_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    continuous_features = [
        "engine_torque",
        "avg_vehicle_speed",
        "avg_coolant_temperature",
    ]

    missing_features = [
        "avg_vehicle_speed",
        "avg_coolant_temperature",
    ]

    for feature in missing_features:
        indicator = (
            f"{feature}_missing"
        )

        train_df[
            indicator
        ] = (
            train_df[
                feature
            ]
            .isna()
            .astype(int)
        )

        test_df[
            indicator
        ] = (
            test_df[
                feature
            ]
            .isna()
            .astype(int)
        )

    medians = {}

    for feature in continuous_features:
        median = float(
            train_df[
                feature
            ].median()
        )

        medians[
            feature
        ] = median

        train_df[
            feature
        ] = (
            train_df[
                feature
            ]
            .fillna(median)
        )

        test_df[
            feature
        ] = (
            test_df[
                feature
            ]
            .fillna(median)
        )

    return (
        train_df,
        test_df,
        medians,
    )


def run_fold(
    df: pd.DataFrame,
    holdout_vehicle: str,
):
    train_df = df[
        df["vehicle"]
        != holdout_vehicle
    ].copy()

    test_df = df[
        df["vehicle"]
        == holdout_vehicle
    ].copy()

    logger.info(
        "LOVO fold | holdout=%s | train=%d | test=%d",
        holdout_vehicle,
        len(train_df),
        len(test_df),
    )

    # Fit Physics only on the remaining vehicles.
    X_physics_train, y_physics_train = (
        prepare_training_data(
            train_df
        )
    )

    physics_model = (
        train_physics_model(
            X_physics_train,
            y_physics_train,
        )
    )

    train_df[
        PHYSICS_EXPECTED
    ] = physics_model.predict(
        train_df[
            PHYSICS_FEATURES
        ]
    )

    test_df[
        PHYSICS_EXPECTED
    ] = physics_model.predict(
        test_df[
            PHYSICS_FEATURES
        ]
    )

    train_df[
        PHYSICS_RESIDUAL
    ] = (
        train_df[
            TARGET
        ]
        - train_df[
            PHYSICS_EXPECTED
        ]
    )

    test_df[
        PHYSICS_RESIDUAL
    ] = (
        test_df[
            TARGET
        ]
        - test_df[
            PHYSICS_EXPECTED
        ]
    )

    # Missing-value preprocessing is fitted only on fold training vehicles.
    train_df, test_df, medians = (
        prepare_ml_features(
            train_df,
            test_df,
        )
    )

    # Train the already-selected Expanded Random Forest.
    rf_model = train_rf_model(
        train_df[
            RF_FEATURES
        ],
        train_df[
            PHYSICS_RESIDUAL
        ],
    )

    test_df[
        PREDICTED_RESIDUAL
    ] = rf_model.predict(
        test_df[
            RF_FEATURES
        ]
    )

    test_df[
        CACE_EXPECTED
    ] = (
        test_df[
            PHYSICS_EXPECTED
        ]
        + test_df[
            PREDICTED_RESIDUAL
        ]
    )

    physics_metrics = (
        calculate_metrics(
            test_df[
                TARGET
            ],
            test_df[
                PHYSICS_EXPECTED
            ],
        )
    )

    cace_metrics = (
        calculate_metrics(
            test_df[
                TARGET
            ],
            test_df[
                CACE_EXPECTED
            ],
        )
    )

    mae_improvement = (
        (
            physics_metrics["mae"]
            - cace_metrics["mae"]
        )
        / physics_metrics["mae"]
        * 100
    )

    rmse_improvement = (
        (
            physics_metrics["rmse"]
            - cace_metrics["rmse"]
        )
        / physics_metrics["rmse"]
        * 100
    )

    fold_metrics = {
        "holdout_vehicle": holdout_vehicle,
        "training_observations": len(
            train_df
        ),
        "test_observations": len(
            test_df
        ),
        "physics_only": physics_metrics,
        "cace": cace_metrics,
        "mae_improvement_percent": float(
            mae_improvement
        ),
        "rmse_improvement_percent": float(
            rmse_improvement
        ),
        "training_medians": medians,
    }

    return (
        fold_metrics,
        test_df,
    )


def main():
    logger.info(
        "Starting CACE V1 Leave-One-Vehicle-Out validation"
    )

    df = load_dataset()

    vehicles = sorted(
        df[
            "vehicle"
        ].unique()
    )

    all_metrics = []
    all_predictions = []

    for vehicle in vehicles:
        fold_metrics, predictions = (
            run_fold(
                df,
                vehicle,
            )
        )

        all_metrics.append(
            fold_metrics
        )

        all_predictions.append(
            predictions
        )

        logger.info(
            "%s | Physics MAE=%.6f | CACE MAE=%.6f | "
            "MAE improvement=%.2f%%",
            vehicle,
            fold_metrics[
                "physics_only"
            ][
                "mae"
            ],
            fold_metrics[
                "cace"
            ][
                "mae"
            ],
            fold_metrics[
                "mae_improvement_percent"
            ],
        )

    summary_rows = []

    for result in all_metrics:
        summary_rows.append(
            {
                "holdout_vehicle": (
                    result[
                        "holdout_vehicle"
                    ]
                ),
                "training_observations": (
                    result[
                        "training_observations"
                    ]
                ),
                "test_observations": (
                    result[
                        "test_observations"
                    ]
                ),
                "physics_mae": (
                    result[
                        "physics_only"
                    ][
                        "mae"
                    ]
                ),
                "cace_mae": (
                    result[
                        "cace"
                    ][
                        "mae"
                    ]
                ),
                "physics_rmse": (
                    result[
                        "physics_only"
                    ][
                        "rmse"
                    ]
                ),
                "cace_rmse": (
                    result[
                        "cace"
                    ][
                        "rmse"
                    ]
                ),
                "physics_r2": (
                    result[
                        "physics_only"
                    ][
                        "r2"
                    ]
                ),
                "cace_r2": (
                    result[
                        "cace"
                    ][
                        "r2"
                    ]
                ),
                "mae_improvement_percent": (
                    result[
                        "mae_improvement_percent"
                    ]
                ),
                "rmse_improvement_percent": (
                    result[
                        "rmse_improvement_percent"
                    ]
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        DETAIL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "model": "CACE V1",
        "validation": (
            "leave_one_vehicle_out"
        ),
        "vehicle_count": len(
            vehicles
        ),
        "folds": all_metrics,
    }

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    logger.info(
        "LOVO summary:\n%s",
        summary_df.to_string(
            index=False
        ),
    )

    logger.info(
        "CACE V1 Leave-One-Vehicle-Out validation complete"
    )


if __name__ == "__main__":
    main()
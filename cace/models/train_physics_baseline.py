# Trains the CACE V1 physics baseline using the finalized modeling dataset.
# The model follows the CACE reference equation using Avg RPM, Engine Load,
# and RPM × Load to predict Actual Fuel Used. The fitted model and coefficients
# are saved for downstream prediction, performance evaluation, and residual analysis.

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "1.0"
MODEL_VERSION = "1.0"

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / f"CACE_Dataset_v{DATASET_VERSION}_private.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "physics_baseline"
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

MODEL_FILE = (
    MODEL_DIR
    / f"physics_baseline_v{MODEL_VERSION}_private.joblib"
)

COEFFICIENT_FILE = (
    REPORT_DIR
    / f"physics_baseline_coefficients_v{MODEL_VERSION}_private.json"
)

LOG_FILE = (
    LOG_DIR
    / "train_physics_baseline.log"
)


FEATURES = [
    "rpm_load",
    "avg_rpm",
    "engine_load",
]

TARGET = "actual_fuel_used"


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


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required_columns = (
        FEATURES
        + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    logger.info(
        "Loaded modeling dataset | rows=%d",
        len(df),
    )

    return df


def prepare_training_data(
    df: pd.DataFrame,
):
    modeling_df = df[
        FEATURES + [TARGET]
    ].copy()

    missing_counts = (
        modeling_df
        .isna()
        .sum()
    )

    if missing_counts.any():
        invalid = (
            missing_counts[
                missing_counts > 0
            ]
            .to_dict()
        )

        raise ValueError(
            "Missing values found in modeling data: "
            f"{invalid}"
        )

    X = modeling_df[
        FEATURES
    ]

    y = modeling_df[
        TARGET
    ]

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> LinearRegression:
    model = LinearRegression()

    model.fit(
        X,
        y,
    )

    return model


def build_coefficient_report(
    model: LinearRegression,
    observation_count: int,
) -> dict:
    coefficients = {
        feature: float(coefficient)
        for feature, coefficient
        in zip(
            FEATURES,
            model.coef_,
        )
    }

    return {
        "model_name": "CACE Physics Baseline",
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "method": "Ordinary Least Squares Linear Regression",
        "target": TARGET,
        "features": FEATURES,
        "observation_count": int(
            observation_count
        ),
        "coefficients": coefficients,
        "intercept": float(
            model.intercept_
        ),
    }


def log_fitted_equation(
    model: LinearRegression,
) -> None:
    coefficient_map = dict(
        zip(
            FEATURES,
            model.coef_,
        )
    )

    logger.info(
        "Physics baseline fitted"
    )

    logger.info(
        "beta_1 | rpm_load = %.10f",
        coefficient_map[
            "rpm_load"
        ],
    )

    logger.info(
        "beta_2 | avg_rpm = %.10f",
        coefficient_map[
            "avg_rpm"
        ],
    )

    logger.info(
        "beta_3 | engine_load = %.10f",
        coefficient_map[
            "engine_load"
        ],
    )

    logger.info(
        "beta_4 | intercept = %.10f",
        model.intercept_,
    )


def save_model(
    model: LinearRegression,
) -> None:
    joblib.dump(
        model,
        MODEL_FILE,
    )

    logger.info(
        "Saved physics baseline model: %s",
        MODEL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def save_coefficient_report(
    report: dict,
) -> None:
    with COEFFICIENT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    logger.info(
        "Saved coefficient report: %s",
        COEFFICIENT_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def main():
    logger.info(
        "Starting CACE Physics Baseline v%s training",
        MODEL_VERSION,
    )

    df = (
        load_dataset()
    )

    X, y = (
        prepare_training_data(
            df
        )
    )

    logger.info(
        "Training physics baseline | "
        "observations=%d | features=%d",
        len(X),
        len(FEATURES),
    )

    model = (
        train_model(
            X,
            y,
        )
    )

    coefficient_report = (
        build_coefficient_report(
            model=model,
            observation_count=len(X),
        )
    )

    log_fitted_equation(
        model
    )

    save_model(
        model
    )

    save_coefficient_report(
        coefficient_report
    )

    logger.info(
        "CACE Physics Baseline training complete"
    )


if __name__ == "__main__":
    main()
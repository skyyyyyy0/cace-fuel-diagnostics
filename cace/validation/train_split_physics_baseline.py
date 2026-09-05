# Trains the CACE V1 physics baseline using only the chronological training split.
# The existing physics-baseline training functions are reused so the validation
# workflow follows the same feature definition and least-squares model as Phase 7.
# Validation and test observations are not used during coefficient fitting.

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from cace.models.train_physics_baseline import (
    FEATURES,
    TARGET,
    build_coefficient_report,
    prepare_training_data,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "1.0"
MODEL_VERSION = "1.0"

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "splits"
    / f"CACE_Dataset_v{DATASET_VERSION}_train_private.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "physics_baseline"
    / "validation"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "physics_baseline"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

MODEL_FILE = (
    MODEL_DIR
    / f"physics_baseline_train_v{MODEL_VERSION}_private.joblib"
)

COEFFICIENT_FILE = (
    REPORT_DIR
    / f"physics_baseline_train_coefficients_v{MODEL_VERSION}_private.json"
)

LOG_FILE = (
    LOG_DIR
    / "train_split_physics_baseline.log"
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


def load_train_dataset() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training split not found: {TRAIN_FILE}"
        )

    df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False,
    )

    logger.info(
        "Loaded training split | rows=%d",
        len(df),
    )

    return df


def save_model(
    model,
) -> None:
    joblib.dump(
        model,
        MODEL_FILE,
    )

    logger.info(
        "Saved train-only physics model: %s",
        MODEL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def save_coefficient_report(
    report: dict,
) -> None:
    report = report.copy()

    report[
        "training_scope"
    ] = "chronological_train_split"

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
        "Saved train-only coefficient report: %s",
        COEFFICIENT_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def log_coefficients(
    model,
) -> None:
    coefficient_map = dict(
        zip(
            FEATURES,
            model.coef_,
        )
    )

    logger.info(
        "Train-only physics baseline fitted"
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


def main():
    logger.info(
        "Starting train-only CACE Physics Baseline"
    )

    train_df = (
        load_train_dataset()
    )

    X_train, y_train = (
        prepare_training_data(
            train_df
        )
    )

    logger.info(
        "Training physics baseline | "
        "observations=%d | features=%d",
        len(X_train),
        len(FEATURES),
    )

    model = (
        train_model(
            X_train,
            y_train,
        )
    )

    coefficient_report = (
        build_coefficient_report(
            model=model,
            observation_count=len(X_train),
        )
    )

    log_coefficients(
        model
    )

    save_model(
        model
    )

    save_coefficient_report(
        coefficient_report
    )

    logger.info(
        "Train-only CACE Physics Baseline complete"
    )


if __name__ == "__main__":
    main()
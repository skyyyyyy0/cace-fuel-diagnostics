# Evaluates the trained CACE V1 physics baseline using the finalized modeling
# dataset. The saved model is used to generate Physics Expected Fuel from
# Avg RPM, Engine Load, and RPM × Load. This step does not retrain the model
# or recalculate any Phase 6 features.

import logging
from pathlib import Path

import joblib
import pandas as pd


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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "physics_baseline"
    / f"physics_baseline_v{MODEL_VERSION}_private.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "physics_baseline"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"physics_baseline_predictions_v{MODEL_VERSION}_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "evaluate_physics_baseline.log"
)


FEATURES = [
    "rpm_load",
    "avg_rpm",
    "engine_load",
]

TARGET = "actual_fuel_used"


OUTPUT_DIR.mkdir(
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


def load_physics_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Physics model not found: {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    logger.info(
        "Loaded Physics Baseline v%s",
        MODEL_VERSION,
    )

    return model


def generate_predictions(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    result = df.copy()

    X = result[
        FEATURES
    ]

    result[
        "physics_expected_fuel"
    ] = model.predict(
        X
    )

    return result


def prepare_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        "vehicle",
        "anchor_time_utc",
        "avg_rpm",
        "engine_load",
        "rpm_load",
        TARGET,
        "physics_expected_fuel",
    ]

    if (
        "statistical_outlier_flag"
        in df.columns
    ):
        output_columns.append(
            "statistical_outlier_flag"
        )

    return (
        df[
            output_columns
        ]
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


def main():
    logger.info(
        "Starting CACE Physics Baseline evaluation"
    )

    df = (
        load_dataset()
    )

    model = (
        load_physics_model()
    )

    prediction_df = (
        generate_predictions(
            df=df,
            model=model,
        )
    )

    output_df = (
        prepare_output(
            prediction_df
        )
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Physics Expected Fuel generated | rows=%d",
        len(output_df),
    )

    logger.info(
        "Saved prediction output: %s",
        OUTPUT_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "CACE Physics Baseline evaluation complete"
    )


if __name__ == "__main__":
    main()
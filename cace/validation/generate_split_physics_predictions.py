# Generates Physics Expected Fuel and residuals for the CACE V1 train,
# validation, and test splits. The Physics Baseline fitted only on the
# chronological training split is reused for all three datasets so validation
# and test observations remain unseen during coefficient estimation.

import logging
from pathlib import Path

import joblib
import pandas as pd

from cace.models.train_physics_baseline import (
    FEATURES,
    TARGET,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "1.0"
MODEL_VERSION = "1.0"

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "splits"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "physics_baseline"
    / "validation"
    / f"physics_baseline_train_v{MODEL_VERSION}_private.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "physics_baseline"
    / "predictions"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

LOG_FILE = (
    LOG_DIR
    / "generate_split_physics_predictions.log"
)


SPLITS = [
    "train",
    "validation",
    "test",
]

EXPECTED = "physics_expected_fuel"
RESIDUAL = "physics_residual"


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


def load_physics_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Train-only physics model not found: {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    logger.info(
        "Loaded train-only Physics Baseline v%s",
        MODEL_VERSION,
    )

    return model


def load_split(
    split_name: str,
) -> pd.DataFrame:
    input_file = (
        SPLIT_DIR
        / f"CACE_Dataset_v{DATASET_VERSION}_{split_name}_private.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"{split_name} split not found: {input_file}"
        )

    df = pd.read_csv(
        input_file,
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
            f"{split_name} split is missing required columns: "
            f"{missing_columns}"
        )

    return df


def generate_physics_predictions(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    result = df.copy()

    X = result[
        FEATURES
    ]

    result[
        EXPECTED
    ] = model.predict(
        X
    )

    result[
        RESIDUAL
    ] = (
        result[TARGET]
        - result[EXPECTED]
    )

    return result


def save_split(
    df: pd.DataFrame,
    split_name: str,
) -> None:
    output_file = (
        OUTPUT_DIR
        / f"physics_{split_name}_v{MODEL_VERSION}_private.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved %s physics predictions | rows=%d",
        split_name,
        len(df),
    )


def main():
    logger.info(
        "Starting split-level Physics Baseline prediction"
    )

    model = (
        load_physics_model()
    )

    for split_name in SPLITS:
        df = (
            load_split(
                split_name
            )
        )

        result = (
            generate_physics_predictions(
                df=df,
                model=model,
            )
        )

        save_split(
            df=result,
            split_name=split_name,
        )

    logger.info(
        "Split-level Physics Baseline prediction complete"
    )


if __name__ == "__main__":
    main()
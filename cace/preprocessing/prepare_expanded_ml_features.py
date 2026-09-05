# Prepares expanded residual-model features using train-only preprocessing.
# Median imputation parameters are learned from the Train split and reused
# unchanged for Validation and Test to avoid data leakage.

import json
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v2"
    / "CACE_ML_Features_v1.0_private.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "splits"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v2"
    / "splits"
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

TRAIN_SPLIT_FILE = (
    SPLIT_DIR
    / "CACE_Dataset_v1.0_train_private.csv"
)

VALIDATION_SPLIT_FILE = (
    SPLIT_DIR
    / "CACE_Dataset_v1.0_validation_private.csv"
)

TEST_SPLIT_FILE = (
    SPLIT_DIR
    / "CACE_Dataset_v1.0_test_private.csv"
)

TRAIN_OUTPUT_FILE = (
    OUTPUT_DIR
    / "CACE_ML_Expanded_train_v1.0_private.csv"
)

VALIDATION_OUTPUT_FILE = (
    OUTPUT_DIR
    / "CACE_ML_Expanded_validation_v1.0_private.csv"
)

TEST_OUTPUT_FILE = (
    OUTPUT_DIR
    / "CACE_ML_Expanded_test_v1.0_private.csv"
)

PREPROCESSING_FILE = (
    REPORT_DIR
    / "expanded_ml_preprocessing_v1.0_private.json"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "expanded_ml_preprocessing_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "prepare_expanded_ml_features.log"
)


FEATURES_TO_IMPUTE = [
    "avg_vehicle_speed",
    "avg_coolant_temperature",
]

ML_FEATURES = [
    "engine_torque",
    "avg_vehicle_speed",
    "avg_coolant_temperature",
    "avg_vehicle_speed_missing",
    "avg_coolant_temperature_missing",
]


OUTPUT_DIR.mkdir(
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


def load_split_keys(
    path: Path,
) -> pd.DataFrame:
    split = pd.read_csv(
        path,
        usecols=[
            "vehicle",
            "anchor_time_utc",
        ],
        low_memory=False,
    )

    split["anchor_time_utc"] = pd.to_datetime(
        split["anchor_time_utc"],
        utc=True,
        errors="coerce",
    )

    split = split.dropna(
        subset=[
            "vehicle",
            "anchor_time_utc",
        ]
    )

    return split


def attach_split(
    feature_dataset: pd.DataFrame,
    split_keys: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    split = feature_dataset.merge(
        split_keys,
        on=[
            "vehicle",
            "anchor_time_utc",
        ],
        how="inner",
        validate="one_to_one",
    )

    split["split"] = split_name

    return split


def fit_train_medians(
    train: pd.DataFrame,
) -> dict[str, float]:
    medians = {}

    for feature in FEATURES_TO_IMPUTE:
        median_value = train[
            feature
        ].median()

        if pd.isna(
            median_value
        ):
            raise RuntimeError(
                f"Train median is missing for {feature}."
            )

        medians[
            feature
        ] = float(
            median_value
        )

    return medians


def apply_preprocessing(
    dataset: pd.DataFrame,
    medians: dict[str, float],
) -> pd.DataFrame:
    output = dataset.copy()

    for feature in FEATURES_TO_IMPUTE:
        missing_column = (
            f"{feature}_missing"
        )

        output[
            missing_column
        ] = (
            output[
                feature
            ]
            .isna()
            .astype(int)
        )

        output[
            feature
        ] = (
            output[
                feature
            ]
            .fillna(
                medians[
                    feature
                ]
            )
        )

    return output


def build_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    medians: dict[str, float],
) -> pd.DataFrame:
    rows = []

    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for split_name, frame in split_frames.items():
        for feature in FEATURES_TO_IMPUTE:
            missing_column = (
                f"{feature}_missing"
            )

            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "rows": len(frame),
                    "imputation_value_train_median": medians[
                        feature
                    ],
                    "original_missing_rows": int(
                        frame[
                            missing_column
                        ].sum()
                    ),
                    "missing_after_imputation": int(
                        frame[
                            feature
                        ]
                        .isna()
                        .sum()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    logger.info(
        "Starting expanded ML preprocessing"
    )

    features = pd.read_csv(
        FEATURE_DATASET_FILE,
        low_memory=False,
    )

    features["anchor_time_utc"] = pd.to_datetime(
        features["anchor_time_utc"],
        utc=True,
        errors="coerce",
    )

    features = features.dropna(
        subset=[
            "vehicle",
            "anchor_time_utc",
        ]
    )

    train_keys = load_split_keys(
        TRAIN_SPLIT_FILE
    )

    validation_keys = load_split_keys(
        VALIDATION_SPLIT_FILE
    )

    test_keys = load_split_keys(
        TEST_SPLIT_FILE
    )

    train = attach_split(
        features,
        train_keys,
        "train",
    )

    validation = attach_split(
        features,
        validation_keys,
        "validation",
    )

    test = attach_split(
        features,
        test_keys,
        "test",
    )

    logger.info(
        "Split rows | train=%d | validation=%d | test=%d",
        len(train),
        len(validation),
        len(test),
    )

    if len(train) != len(train_keys):
        raise RuntimeError(
            "Train row count mismatch."
        )

    if len(validation) != len(validation_keys):
        raise RuntimeError(
            "Validation row count mismatch."
        )

    if len(test) != len(test_keys):
        raise RuntimeError(
            "Test row count mismatch."
        )

    medians = fit_train_medians(
        train
    )

    train = apply_preprocessing(
        train,
        medians,
    )

    validation = apply_preprocessing(
        validation,
        medians,
    )

    test = apply_preprocessing(
        test,
        medians,
    )

    for split_name, frame in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        missing_ml_values = (
            frame[
                ML_FEATURES
            ]
            .isna()
            .sum()
            .sum()
        )

        if missing_ml_values > 0:
            raise RuntimeError(
                f"Missing ML values remain in {split_name}: "
                f"{missing_ml_values}"
            )

    preprocessing_report = {
        "version": "1.0",
        "fit_scope": "train_only",
        "imputation_method": "median",
        "features": medians,
        "missing_indicators": [
            "avg_vehicle_speed_missing",
            "avg_coolant_temperature_missing",
        ],
        "ml_features": ML_FEATURES,
    }

    summary = build_summary(
        train,
        validation,
        test,
        medians,
    )

    train.to_csv(
        TRAIN_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    validation.to_csv(
        VALIDATION_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    test.to_csv(
        TEST_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        PREPROCESSING_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            preprocessing_report,
            file,
            indent=2,
        )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Train medians: %s",
        medians,
    )

    logger.info(
        "\n%s",
        summary.to_string(
            index=False
        ),
    )

    logger.info(
        "Expanded ML preprocessing complete"
    )


if __name__ == "__main__":
    main()
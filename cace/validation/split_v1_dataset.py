# Splits the finalized CACE V1 modeling dataset into chronological train,
# validation, and test sets for each vehicle. The split preserves time order
# within each vehicle and avoids random sampling so future observations remain
# separated from the data used for model fitting and model selection.

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "1.0"

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / f"CACE_Dataset_v{DATASET_VERSION}_private.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "splits"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / f"CACE_Dataset_v{DATASET_VERSION}_train_private.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / f"CACE_Dataset_v{DATASET_VERSION}_validation_private.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / f"CACE_Dataset_v{DATASET_VERSION}_test_private.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / f"cace_v1_split_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "split_v1_dataset.log"
)


TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


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


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Finalized dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required_columns = [
        "vehicle",
        "anchor_time_utc",
    ]

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

    df[
        "anchor_time_utc"
    ] = pd.to_datetime(
        df[
            "anchor_time_utc"
        ],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "vehicle",
            "anchor_time_utc",
        ]
    )

    logger.info(
        "Loaded finalized dataset | rows=%d",
        len(df),
    )

    return df


def split_vehicle_data(
    vehicle_df: pd.DataFrame,
):
    vehicle_df = (
        vehicle_df
        .sort_values(
            "anchor_time_utc"
        )
        .reset_index(
            drop=True
        )
    )

    total_count = len(
        vehicle_df
    )

    train_end = int(
        total_count
        * TRAIN_RATIO
    )

    validation_end = (
        train_end
        + int(
            total_count
            * VALIDATION_RATIO
        )
    )

    train_df = (
        vehicle_df
        .iloc[
            :train_end
        ]
        .copy()
    )

    validation_df = (
        vehicle_df
        .iloc[
            train_end:
            validation_end
        ]
        .copy()
    )

    test_df = (
        vehicle_df
        .iloc[
            validation_end:
        ]
        .copy()
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


def split_dataset(
    df: pd.DataFrame,
):
    train_frames = []
    validation_frames = []
    test_frames = []
    summary_rows = []

    for vehicle, vehicle_df in df.groupby(
        "vehicle"
    ):
        (
            train_df,
            validation_df,
            test_df,
        ) = split_vehicle_data(
            vehicle_df
        )

        train_frames.append(
            train_df
        )

        validation_frames.append(
            validation_df
        )

        test_frames.append(
            test_df
        )

        summary_rows.append(
            {
                "vehicle": vehicle,
                "total_count": len(
                    vehicle_df
                ),
                "train_count": len(
                    train_df
                ),
                "validation_count": len(
                    validation_df
                ),
                "test_count": len(
                    test_df
                ),
            }
        )

    train_df = pd.concat(
        train_frames,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_frames,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_frames,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        train_df,
        validation_df,
        test_df,
        summary_df,
    )


def save_outputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    train_df.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved train dataset | rows=%d",
        len(train_df),
    )

    logger.info(
        "Saved validation dataset | rows=%d",
        len(validation_df),
    )

    logger.info(
        "Saved test dataset | rows=%d",
        len(test_df),
    )


def main():
    logger.info(
        "Starting CACE V1 chronological dataset split"
    )

    df = (
        load_dataset()
    )

    (
        train_df,
        validation_df,
        test_df,
        summary_df,
    ) = split_dataset(
        df
    )

    save_outputs(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        summary_df=summary_df,
    )

    logger.info(
        "Dataset split complete | "
        "train=%d | validation=%d | test=%d",
        len(train_df),
        len(validation_df),
        len(test_df),
    )


if __name__ == "__main__":
    main()
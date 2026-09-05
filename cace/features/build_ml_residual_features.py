# Builds additional operating-condition features for the CACE residual model.
# Features are calculated from actual GeoTab observations within ±60 seconds
# of each existing CACE V1 anchor. Missing observations remain missing.

import logging
from pathlib import Path

import pandas as pd

from cace.utils.modeling_windows import (
    build_vehicle_labels,
    load_vehicle_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v1"
    / "CACE_Dataset_v1.0_private.csv"
)

SIGNAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_signals"
    / "geotab"
)

VEHICLE_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "vehicles.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_v2"
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "CACE_ML_Features_v1.0_private.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "ml_feature_build_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "build_ml_residual_features.log"
)


WINDOW_SECONDS = 60

NEW_FEATURES = {
    "vehicle_speed": "avg_vehicle_speed",
    "coolant_temperature": "avg_coolant_temperature",
}


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


def load_signal_data(
    vehicle_id: str,
) -> pd.DataFrame:
    vehicle_dir = (
        SIGNAL_ROOT
        / f"vehicle_{vehicle_id}"
    )

    frames = []

    for file in vehicle_dir.rglob("*.csv"):
        df = pd.read_csv(
            file,
            usecols=[
                "datetime_utc",
                "signal_key",
                "value",
            ],
            low_memory=False,
        )

        df = df[
            df["signal_key"].isin(
                NEW_FEATURES
            )
        ]

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "datetime_utc",
                "signal_key",
                "value",
            ]
        )

    signals = pd.concat(
        frames,
        ignore_index=True,
    )

    signals["datetime_utc"] = pd.to_datetime(
        signals["datetime_utc"],
        utc=True,
        errors="coerce",
    )

    signals["value"] = pd.to_numeric(
        signals["value"],
        errors="coerce",
    )

    signals = signals.dropna(
        subset=[
            "datetime_utc",
            "signal_key",
            "value",
        ]
    )

    signals = signals.drop_duplicates(
        subset=[
            "datetime_utc",
            "signal_key",
            "value",
        ]
    )

    return signals.sort_values(
        "datetime_utc"
    ).reset_index(
        drop=True
    )


def calculate_window_features(
    anchors: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for anchor in anchors.itertuples(
        index=False
    ):
        window_start = (
            anchor.anchor_time_utc
            - pd.Timedelta(
                seconds=WINDOW_SECONDS
            )
        )

        window_end = (
            anchor.anchor_time_utc
            + pd.Timedelta(
                seconds=WINDOW_SECONDS
            )
        )

        window = signals[
            signals["datetime_utc"].between(
                window_start,
                window_end,
                inclusive="both",
            )
        ]

        row = {
            "vehicle": anchor.vehicle,
            "anchor_time_utc": anchor.anchor_time_utc,
        }

        for signal_key, feature_name in NEW_FEATURES.items():
            values = (
                window.loc[
                    window["signal_key"]
                    == signal_key,
                    "value",
                ]
                .dropna()
            )

            row[feature_name] = (
                values.mean()
                if not values.empty
                else float("nan")
            )

            row[
                f"{feature_name}_observation_count"
            ] = len(values)

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def build_summary(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    feature_names = list(
        NEW_FEATURES.values()
    )

    for feature in feature_names:
        available = (
            dataset[feature]
            .notna()
            .sum()
        )

        missing = (
            dataset[feature]
            .isna()
            .sum()
        )

        rows.append(
            {
                "feature": feature,
                "total_rows": len(dataset),
                "available_rows": available,
                "missing_rows": missing,
                "coverage_percent": round(
                    available
                    / len(dataset)
                    * 100,
                    2,
                ),
                "mean": dataset[
                    feature
                ].mean(),
                "std": dataset[
                    feature
                ].std(),
                "min": dataset[
                    feature
                ].min(),
                "max": dataset[
                    feature
                ].max(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    logger.info(
        "Starting ML residual feature build"
    )

    dataset = pd.read_csv(
        DATASET_FILE,
        low_memory=False,
    )

    dataset["anchor_time_utc"] = pd.to_datetime(
        dataset["anchor_time_utc"],
        utc=True,
        errors="coerce",
    )

    dataset = dataset.dropna(
        subset=[
            "vehicle",
            "anchor_time_utc",
        ]
    )

    dataset["vehicle"] = (
        dataset["vehicle"]
        .astype(str)
    )

    logger.info(
        "Loaded CACE V1 dataset | rows=%d",
        len(dataset),
    )

    vehicles = load_vehicle_ids(
        VEHICLE_CONFIG
    )

    vehicle_labels = build_vehicle_labels(
        vehicles
    )

    label_to_vehicle = {
        label: vehicle_id
        for vehicle_id, label
        in vehicle_labels.items()
    }

    feature_frames = []

    for vehicle_label in sorted(
        dataset["vehicle"].unique()
    ):
        if vehicle_label not in label_to_vehicle:
            logger.warning(
                "No vehicle mapping found for %s",
                vehicle_label,
            )
            continue

        vehicle_id = (
            label_to_vehicle[
                vehicle_label
            ]
        )

        anchors = dataset[
            dataset["vehicle"]
            == vehicle_label
        ][
            [
                "vehicle",
                "anchor_time_utc",
            ]
        ].copy()

        signals = load_signal_data(
            vehicle_id
        )

        logger.info(
            (
                "Vehicle %s | "
                "anchors=%d | "
                "signal_rows=%d"
            ),
            vehicle_label,
            len(anchors),
            len(signals),
        )

        vehicle_features = (
            calculate_window_features(
                anchors,
                signals,
            )
        )

        feature_frames.append(
            vehicle_features
        )

    if not feature_frames:
        raise RuntimeError(
            "No ML features were generated."
        )

    features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    duplicate_keys = features.duplicated(
        subset=[
            "vehicle",
            "anchor_time_utc",
        ]
    ).sum()

    if duplicate_keys > 0:
        raise RuntimeError(
            "Duplicate vehicle-anchor keys "
            f"found: {duplicate_keys}"
        )

    output = dataset.merge(
        features,
        on=[
            "vehicle",
            "anchor_time_utc",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(output) != len(dataset):
        raise RuntimeError(
            "Row count changed after feature merge."
        )

    summary = build_summary(
        output
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "\n%s",
        summary.to_string(
            index=False
        ),
    )

    logger.info(
        "Output rows=%d",
        len(output),
    )

    logger.info(
        "Saved ML feature dataset: %s",
        OUTPUT_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved feature summary: %s",
        SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "ML residual feature build complete"
    )


if __name__ == "__main__":
    main()
# Checks whether additional operating-condition signals are available around
# the existing CACE V1 anchor timestamps. This audit reuses the same vehicle
# labeling logic as the modeling pipeline and does not modify the dataset,
# Physics Baseline, or train/validation/test splits.

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

SUMMARY_FILE = (
    REPORT_DIR
    / "anchor_feature_coverage_summary_private.csv"
)

DETAIL_FILE = (
    REPORT_DIR
    / "anchor_feature_coverage_detail_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "audit_anchor_feature_coverage.log"
)


WINDOW_SECONDS = 60

CANDIDATE_SIGNALS = [
    "vehicle_speed",
    "outside_temperature",
    "coolant_temperature",
    "soot_load",
    "active_regen_status",
]


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

    files = list(
        vehicle_dir.rglob("*.csv")
    )

    frames = []

    for file in files:
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
                CANDIDATE_SIGNALS
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


def check_anchor_coverage(
    anchors: pd.DataFrame,
    signals: pd.DataFrame,
) -> list[dict]:
    rows = []

    for anchor in anchors.itertuples(
        index=False
    ):
        anchor_time = (
            anchor.anchor_time_utc
        )

        window_start = (
            anchor_time
            - pd.Timedelta(
                seconds=WINDOW_SECONDS
            )
        )

        window_end = (
            anchor_time
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

        for signal_key in CANDIDATE_SIGNALS:
            signal_window = window[
                window["signal_key"]
                == signal_key
            ]

            observation_count = len(
                signal_window
            )

            rows.append(
                {
                    "vehicle": anchor.vehicle,
                    "anchor_time_utc": anchor_time,
                    "signal_key": signal_key,
                    "observation_count": observation_count,
                    "available": observation_count > 0,
                }
            )

    return rows


def build_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        detail.groupby(
            [
                "vehicle",
                "signal_key",
            ]
        )
        .agg(
            total_anchors=(
                "available",
                "size",
            ),
            available_anchors=(
                "available",
                "sum",
            ),
            median_observations=(
                "observation_count",
                "median",
            ),
            mean_observations=(
                "observation_count",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["missing_anchors"] = (
        summary["total_anchors"]
        - summary["available_anchors"]
    )

    summary["coverage_percent"] = (
        summary["available_anchors"]
        / summary["total_anchors"]
        * 100
    ).round(2)

    summary["mean_observations"] = (
        summary["mean_observations"]
        .round(2)
    )

    return summary[
        [
            "vehicle",
            "signal_key",
            "total_anchors",
            "available_anchors",
            "missing_anchors",
            "coverage_percent",
            "median_observations",
            "mean_observations",
        ]
    ]


def main():
    logger.info(
        "Starting anchor feature coverage audit"
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

    vehicle_labels = (
        build_vehicle_labels(
            vehicles
        )
    )

    label_to_vehicle = {
        label: vehicle_id
        for vehicle_id, label
        in vehicle_labels.items()
    }

    logger.info(
        "Loaded vehicle mapping | vehicles=%d",
        len(label_to_vehicle),
    )

    all_rows = []

    dataset_labels = sorted(
        dataset["vehicle"].unique()
    )

    for vehicle_label in dataset_labels:
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

        vehicle_anchors = dataset[
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
                "source_vehicle=%s | "
                "anchors=%d | "
                "candidate_signal_rows=%d"
            ),
            vehicle_label,
            vehicle_id,
            len(vehicle_anchors),
            len(signals),
        )

        all_rows.extend(
            check_anchor_coverage(
                vehicle_anchors,
                signals,
            )
        )

    if not all_rows:
        raise RuntimeError(
            "No anchor coverage records were generated."
        )

    detail = pd.DataFrame(
        all_rows
    )

    summary = build_summary(
        detail
    )

    detail.to_csv(
        DETAIL_FILE,
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
        "Saved detail report: %s",
        DETAIL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved summary report: %s",
        SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Anchor feature coverage audit complete"
    )


if __name__ == "__main__":
    main()
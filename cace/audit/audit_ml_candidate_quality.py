# Reviews the value quality and variability of additional ML candidate signals
# around the existing CACE V1 anchors. This audit uses the same vehicle mapping
# and ±60 second window as the modeling pipeline and does not modify the dataset.

import logging
from pathlib import Path

import numpy as np
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
    / "ml_candidate_quality_summary_private.csv"
)

DETAIL_FILE = (
    REPORT_DIR
    / "ml_candidate_quality_detail_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "audit_ml_candidate_quality.log"
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


def collect_anchor_values(
    anchors: pd.DataFrame,
    signals: pd.DataFrame,
) -> list[dict]:
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

        for signal_key in CANDIDATE_SIGNALS:
            values = (
                window.loc[
                    window["signal_key"]
                    == signal_key,
                    "value",
                ]
                .dropna()
            )

            if values.empty:
                rows.append(
                    {
                        "vehicle": anchor.vehicle,
                        "anchor_time_utc": anchor.anchor_time_utc,
                        "signal_key": signal_key,
                        "observation_count": 0,
                        "mean_value": np.nan,
                        "min_value": np.nan,
                        "max_value": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "vehicle": anchor.vehicle,
                    "anchor_time_utc": anchor.anchor_time_utc,
                    "signal_key": signal_key,
                    "observation_count": len(values),
                    "mean_value": values.mean(),
                    "min_value": values.min(),
                    "max_value": values.max(),
                }
            )

    return rows


def build_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    available = detail[
        detail["observation_count"] > 0
    ].copy()

    summary = (
        available.groupby(
            "signal_key"
        )
        .agg(
            available_anchors=(
                "mean_value",
                "size",
            ),
            mean_value=(
                "mean_value",
                "mean",
            ),
            std_value=(
                "mean_value",
                "std",
            ),
            min_value=(
                "min_value",
                "min",
            ),
            max_value=(
                "max_value",
                "max",
            ),
            median_observations=(
                "observation_count",
                "median",
            ),
        )
        .reset_index()
    )

    summary["zero_value_anchors"] = (
        available.assign(
            is_zero=(
                available["mean_value"]
                == 0
            )
        )
        .groupby(
            "signal_key"
        )["is_zero"]
        .sum()
        .reindex(
            summary["signal_key"]
        )
        .to_numpy()
    )

    summary["zero_percent"] = (
        summary["zero_value_anchors"]
        / summary["available_anchors"]
        * 100
    ).round(2)

    summary["above_100_anchors"] = (
        available.assign(
            above_100=(
                available["max_value"]
                > 100
            )
        )
        .groupby(
            "signal_key"
        )["above_100"]
        .sum()
        .reindex(
            summary["signal_key"]
        )
        .to_numpy()
    )

    summary["above_100_percent"] = (
        summary["above_100_anchors"]
        / summary["available_anchors"]
        * 100
    ).round(2)

    numeric_columns = [
        "mean_value",
        "std_value",
        "min_value",
        "max_value",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(4)
    )

    return summary


def main():
    logger.info(
        "Starting ML candidate quality audit"
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

    all_rows = []

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
                "candidate_signal_rows=%d"
            ),
            vehicle_label,
            len(anchors),
            len(signals),
        )

        all_rows.extend(
            collect_anchor_values(
                anchors,
                signals,
            )
        )

    if not all_rows:
        raise RuntimeError(
            "No candidate feature records were generated."
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
        "ML candidate quality audit complete"
    )


if __name__ == "__main__":
    main()
# Compares RPM aggregation methods around observed Engine Load timestamps.
# The analysis evaluates whether mean or median RPM is more appropriate for
# representing engine behavior within the candidate CACE windows. Vehicle IDs
# are anonymized in generated reports so the outputs can be reviewed safely.

import logging
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CONFIG = PROJECT_ROOT / "config" / "vehicles.yaml"

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_signals"
    / "geotab"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "window_design"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "rpm_aggregation_detail_private.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "rpm_aggregation_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "rpm_aggregation_analysis.log"
)


# Candidate half-window sizes around each Engine Load observation.
WINDOWS = {
    "pm_30s": 30,
    "pm_60s": 60,
    "pm_120s": 120,
}


REQUIRED_COLUMNS = {
    "vehicle",
    "datetime_utc",
    "signal_key",
    "value",
}


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


def load_vehicle_config():
    if not VEHICLE_CONFIG.exists():
        raise FileNotFoundError(
            f"Vehicle config not found: {VEHICLE_CONFIG}"
        )

    with VEHICLE_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    vehicles = config.get(
        "vehicles",
        [],
    )

    if not vehicles:
        raise ValueError(
            "No vehicles found in config/vehicles.yaml"
        )

    return [
        str(vehicle).strip()
        for vehicle in vehicles
        if str(vehicle).strip()
    ]


def build_vehicle_labels(vehicles):
    return {
        vehicle: f"VEHICLE_{index:02d}"
        for index, vehicle in enumerate(
            vehicles,
            start=1,
        )
    }


def find_vehicle_files(vehicle):
    vehicle_dir = (
        INPUT_DIR
        / f"vehicle_{vehicle}"
    )

    if not vehicle_dir.exists():
        logger.warning(
            "%s | processed signal directory not found",
            vehicle,
        )
        return []

    return sorted(
        vehicle_dir.rglob(
            "statusdata_*.csv"
        )
    )


def read_signal_file(path):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df[
        [
            "vehicle",
            "datetime_utc",
            "signal_key",
            "value",
        ]
    ].copy()


def load_vehicle_data(vehicle):
    files = find_vehicle_files(
        vehicle
    )

    if not files:
        return pd.DataFrame()

    frames = []

    for path in files:
        try:
            frames.append(
                read_signal_file(path)
            )

        except Exception as exc:
            logger.warning(
                "%s | skipped %s | %s",
                vehicle,
                path.name,
                exc,
            )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["vehicle"] = (
        df["vehicle"]
        .astype(str)
        .str.strip()
    )

    df["signal_key"] = (
        df["signal_key"]
        .astype(str)
        .str.strip()
    )

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime_utc",
            "signal_key",
            "value",
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "vehicle",
            "datetime_utc",
            "signal_key",
            "value",
        ]
    )

    return (
        df
        .sort_values(
            "datetime_utc"
        )
        .reset_index(
            drop=True
        )
    )


def select_signal(
    df,
    signal_key,
):
    return (
        df.loc[
            df["signal_key"].eq(
                signal_key
            ),
            [
                "datetime_utc",
                "value",
            ],
        ]
        .sort_values(
            "datetime_utc"
        )
        .reset_index(
            drop=True
        )
    )


def summarize_rpm_window(
    vehicle_label,
    anchor_time,
    anchor_load,
    window_name,
    half_window_sec,
    rpm_df,
):
    window_start = (
        anchor_time
        - pd.Timedelta(
            seconds=half_window_sec
        )
    )

    window_end = (
        anchor_time
        + pd.Timedelta(
            seconds=half_window_sec
        )
    )

    rpm_window = rpm_df.loc[
        rpm_df["datetime_utc"].between(
            window_start,
            window_end,
            inclusive="both",
        )
    ]

    if rpm_window.empty:
        return {
            "vehicle": vehicle_label,
            "anchor_time_utc": anchor_time,
            "window_name": window_name,
            "half_window_sec": half_window_sec,
            "total_window_sec": half_window_sec * 2,
            "anchor_load": anchor_load,
            "rpm_count": 0,
            "rpm_mean": None,
            "rpm_median": None,
            "rpm_min": None,
            "rpm_max": None,
            "rpm_std": None,
            "mean_median_diff": None,
            "mean_median_diff_pct": None,
        }

    rpm_values = rpm_window["value"]

    rpm_mean = rpm_values.mean()
    rpm_median = rpm_values.median()

    mean_median_diff = (
        rpm_mean
        - rpm_median
    )

    if rpm_median != 0:
        mean_median_diff_pct = (
            abs(mean_median_diff)
            / abs(rpm_median)
            * 100
        )
    else:
        mean_median_diff_pct = None

    return {
        "vehicle": vehicle_label,
        "anchor_time_utc": anchor_time,
        "window_name": window_name,
        "half_window_sec": half_window_sec,
        "total_window_sec": half_window_sec * 2,
        "anchor_load": anchor_load,
        "rpm_count": len(rpm_values),
        "rpm_mean": rpm_mean,
        "rpm_median": rpm_median,
        "rpm_min": rpm_values.min(),
        "rpm_max": rpm_values.max(),
        "rpm_std": rpm_values.std(),
        "mean_median_diff": mean_median_diff,
        "mean_median_diff_pct": mean_median_diff_pct,
    }


def build_vehicle_detail(
    vehicle_label,
    vehicle_df,
):
    load_df = select_signal(
        vehicle_df,
        "engine_load",
    )

    rpm_df = select_signal(
        vehicle_df,
        "rpm",
    )

    if load_df.empty:
        logger.warning(
            "%s | skipped: Engine Load anchors unavailable",
            vehicle_label,
        )
        return pd.DataFrame()

    if rpm_df.empty:
        logger.warning(
            "%s | skipped: RPM unavailable",
            vehicle_label,
        )
        return pd.DataFrame()

    rows = []

    for anchor in load_df.itertuples(
        index=False
    ):
        for (
            window_name,
            half_window_sec,
        ) in WINDOWS.items():

            rows.append(
                summarize_rpm_window(
                    vehicle_label=vehicle_label,
                    anchor_time=anchor.datetime_utc,
                    anchor_load=anchor.value,
                    window_name=window_name,
                    half_window_sec=half_window_sec,
                    rpm_df=rpm_df,
                )
            )

    return pd.DataFrame(
        rows
    )


def build_summary(detail_df):
    detail_df = detail_df.copy()

    detail_df["has_rpm"] = (
        detail_df["rpm_count"]
        > 0
    )

    summary = (
        detail_df
        .groupby(
            [
                "vehicle",
                "window_name",
                "half_window_sec",
                "total_window_sec",
            ],
            as_index=False,
        )
        .agg(
            anchor_count=(
                "anchor_time_utc",
                "size",
            ),
            rpm_coverage_pct=(
                "has_rpm",
                "mean",
            ),
            median_rpm_count=(
                "rpm_count",
                "median",
            ),
            median_rpm_mean=(
                "rpm_mean",
                "median",
            ),
            median_rpm_median=(
                "rpm_median",
                "median",
            ),
            median_rpm_std=(
                "rpm_std",
                "median",
            ),
            median_mean_median_diff=(
                "mean_median_diff",
                lambda values: (
                    values.abs().median()
                ),
            ),
            median_mean_median_diff_pct=(
                "mean_median_diff_pct",
                "median",
            ),
            p95_mean_median_diff_pct=(
                "mean_median_diff_pct",
                lambda values: (
                    values.quantile(0.95)
                ),
            ),
        )
    )

    summary["rpm_coverage_pct"] = (
        summary["rpm_coverage_pct"]
        * 100
    )

    numeric_columns = [
        "rpm_coverage_pct",
        "median_rpm_count",
        "median_rpm_mean",
        "median_rpm_median",
        "median_rpm_std",
        "median_mean_median_diff",
        "median_mean_median_diff_pct",
        "p95_mean_median_diff_pct",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(2)
    )

    return (
        summary
        .sort_values(
            [
                "vehicle",
                "half_window_sec",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def save_reports(
    detail_df,
    summary_df,
):
    detail_df.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    logger.info(
        "Saved RPM detail report: %s",
        DETAIL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved RPM summary report: %s",
        SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def print_summary(summary_df):
    columns = [
        "vehicle",
        "window_name",
        "anchor_count",
        "rpm_coverage_pct",
        "median_rpm_count",
        "median_rpm_mean",
        "median_rpm_median",
        "median_mean_median_diff_pct",
        "p95_mean_median_diff_pct",
    ]

    print()
    print("CACE RPM Aggregation Summary")
    print()

    print(
        summary_df[
            columns
        ].to_string(
            index=False
        )
    )


def main():
    logger.info(
        "Starting CACE RPM aggregation analysis"
    )

    vehicles = (
        load_vehicle_config()
    )

    vehicle_labels = (
        build_vehicle_labels(
            vehicles
        )
    )

    detail_frames = []

    for vehicle in vehicles:
        vehicle_label = (
            vehicle_labels[
                vehicle
            ]
        )

        logger.info(
            "%s | analyzing RPM aggregation",
            vehicle_label,
        )

        vehicle_df = (
            load_vehicle_data(
                vehicle
            )
        )

        if vehicle_df.empty:
            logger.warning(
                "%s | skipped: no usable processed data",
                vehicle_label,
            )
            continue

        detail_df = (
            build_vehicle_detail(
                vehicle_label,
                vehicle_df,
            )
        )

        if detail_df.empty:
            continue

        detail_frames.append(
            detail_df
        )

        logger.info(
            "%s | analyzed %d Engine Load anchors",
            vehicle_label,
            detail_df[
                "anchor_time_utc"
            ].nunique(),
        )

    if not detail_frames:
        raise RuntimeError(
            "No RPM aggregation results were generated"
        )

    detail_df = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    summary_df = build_summary(
        detail_df
    )

    save_reports(
        detail_df,
        summary_df,
    )

    print_summary(
        summary_df
    )

    logger.info(
        "RPM aggregation analysis complete | "
        "vehicles=%d | anchors=%d",
        detail_df[
            "vehicle"
        ].nunique(),
        detail_df[
            [
                "vehicle",
                "anchor_time_utc",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )


if __name__ == "__main__":
    main()
# Measures the actual reporting frequency of GeoTab signals collected for each
# CACE vehicle. Since GeoTab data is event-driven, the script summarizes the
# timestamp intervals and larger gaps for each signal instead of assuming a
# fixed sampling rate. The results are used to evaluate which signals have
# enough temporal coverage for synchronization and CACE modeling.

import logging
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CONFIG = PROJECT_ROOT / "config" / "vehicles.yaml"

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "geotab"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_audit"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

FREQUENCY_REPORT_FILE = (
    OUTPUT_DIR
    / "signal_frequency_analysis.csv"
)

LOG_FILE = (
    LOG_DIR
    / "frequency_analysis.log"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


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


def load_yaml(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_target_vehicles():
    config = load_yaml(VEHICLE_CONFIG)
    vehicles = config.get("vehicles", [])

    if not vehicles:
        raise ValueError(
            "No vehicles found in config/vehicles.yaml"
        )

    return [
        str(vehicle).strip()
        for vehicle in vehicles
    ]


def find_raw_files(vehicle):
    vehicle_dir = (
        RAW_DATA_DIR
        / f"vehicle_{vehicle}"
    )

    if not vehicle_dir.exists():
        logger.warning(
            "%s | raw data directory not found",
            vehicle,
        )
        return []

    files = sorted(
        vehicle_dir.rglob("statusdata_*.csv")
    )

    logger.info(
        "%s | found %d raw file(s)",
        vehicle,
        len(files),
    )

    return files


def load_vehicle_data(vehicle):
    files = find_raw_files(vehicle)

    if not files:
        return pd.DataFrame()

    frames = []

    columns = [
        "vehicle",
        "datetime_utc",
        "diagnostic_id",
        "diagnostic_name",
        "value",
        "unit",
    ]

    for file in files:
        try:
            df = pd.read_csv(
                file,
                usecols=columns,
            )

            frames.append(df)

        except Exception as exc:
            logger.warning(
                "%s | failed to read %s | %s",
                vehicle,
                file.name,
                exc,
            )

    if not frames:
        return pd.DataFrame()

    vehicle_df = pd.concat(
        frames,
        ignore_index=True,
    )

    vehicle_df["datetime_utc"] = pd.to_datetime(
        vehicle_df["datetime_utc"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    vehicle_df = vehicle_df[
        vehicle_df["datetime_utc"].notna()
    ].copy()

    # Incremental extraction uses an overlap window, so remove any repeated
    # StatusData rows before calculating the actual reporting intervals.
    vehicle_df = vehicle_df.drop_duplicates(
        subset=[
            "vehicle",
            "datetime_utc",
            "diagnostic_id",
            "value",
        ]
    )

    return vehicle_df.reset_index(drop=True)


def calculate_signal_frequency(
    vehicle,
    diagnostic_id,
    diagnostic_name,
    signal_df,
):
    signal_df = (
        signal_df
        .sort_values("datetime_utc")
        .copy()
    )

    # Multiple values can occasionally share the same timestamp. Frequency
    # should measure the time between observations, not duplicate timestamps.
    timestamps = (
        signal_df["datetime_utc"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    record_count = len(signal_df)
    timestamp_count = len(timestamps)

    if timestamp_count < 2:
        return {
            "vehicle": vehicle,
            "diagnostic_id": diagnostic_id,
            "diagnostic_name": diagnostic_name,
            "record_count": record_count,
            "timestamp_count": timestamp_count,
            "first_datetime_utc": (
                timestamps.min()
                if timestamp_count
                else pd.NaT
            ),
            "last_datetime_utc": (
                timestamps.max()
                if timestamp_count
                else pd.NaT
            ),
            "median_interval_sec": None,
            "mean_interval_sec": None,
            "p25_interval_sec": None,
            "p75_interval_sec": None,
            "p95_interval_sec": None,
            "min_interval_sec": None,
            "max_interval_sec": None,
        }

    intervals = (
        timestamps
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    return {
        "vehicle": vehicle,
        "diagnostic_id": diagnostic_id,
        "diagnostic_name": diagnostic_name,
        "record_count": record_count,
        "timestamp_count": timestamp_count,
        "first_datetime_utc": timestamps.min(),
        "last_datetime_utc": timestamps.max(),
        "median_interval_sec": intervals.median(),
        "mean_interval_sec": intervals.mean(),
        "p25_interval_sec": intervals.quantile(0.25),
        "p75_interval_sec": intervals.quantile(0.75),
        "p95_interval_sec": intervals.quantile(0.95),
        "min_interval_sec": intervals.min(),
        "max_interval_sec": intervals.max(),
    }


def build_frequency_report(
    vehicle,
    vehicle_df,
):
    if vehicle_df.empty:
        return pd.DataFrame()

    rows = []

    grouped = vehicle_df.groupby(
        [
            "diagnostic_id",
            "diagnostic_name",
        ],
        dropna=False,
    )

    for (
        diagnostic_id,
        diagnostic_name,
    ), signal_df in grouped:

        rows.append(
            calculate_signal_frequency(
                vehicle=vehicle,
                diagnostic_id=diagnostic_id,
                diagnostic_name=diagnostic_name,
                signal_df=signal_df,
            )
        )

    report_df = pd.DataFrame(rows)

    if report_df.empty:
        return report_df

    return (
        report_df
        .sort_values(
            [
                "vehicle",
                "median_interval_sec",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def save_report(report_df):
    report_df.to_csv(
        FREQUENCY_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved frequency report: %s",
        FREQUENCY_REPORT_FILE,
    )


def print_summary(report_df):
    if report_df.empty:
        return

    columns = [
        "vehicle",
        "diagnostic_name",
        "record_count",
        "median_interval_sec",
        "mean_interval_sec",
        "p95_interval_sec",
        "max_interval_sec",
    ]

    print()
    print("Signal Frequency Summary")
    print(
        report_df[columns]
        .round(2)
        .to_string(index=False)
    )


def main():
    vehicles = load_target_vehicles()

    report_frames = []

    for vehicle in vehicles:
        logger.info(
            "%s | starting frequency analysis",
            vehicle,
        )

        vehicle_df = load_vehicle_data(
            vehicle
        )

        if vehicle_df.empty:
            logger.warning(
                "%s | no StatusData available",
                vehicle,
            )
            continue

        logger.info(
            "%s | loaded %d unique StatusData rows",
            vehicle,
            len(vehicle_df),
        )

        vehicle_report = build_frequency_report(
            vehicle,
            vehicle_df,
        )

        report_frames.append(
            vehicle_report
        )

    if not report_frames:
        logger.warning(
            "No frequency results were generated."
        )
        return

    frequency_df = pd.concat(
        report_frames,
        ignore_index=True,
    )

    save_report(frequency_df)
    print_summary(frequency_df)

    logger.info(
        "Frequency analysis complete for %d vehicles",
        len(vehicles),
    )


if __name__ == "__main__":
    main()
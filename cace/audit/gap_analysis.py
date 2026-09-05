# Checks for unusually long reporting gaps in GeoTab signals while each vehicle
# is operating. Reporting intervals are calculated separately within each driving
# period so parked or inactive time between trips is not treated as missing data.
# The results are used to evaluate temporal coverage before CACE synchronization.

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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "signal_gap_analysis.csv"
)

LOG_FILE = (
    LOG_DIR
    / "gap_analysis.log"
)


# Initial audit rule:
# flag intervals that are at least 5x the normal median interval,
# with a minimum threshold of 60 seconds.
GAP_MULTIPLIER = 5
MIN_GAP_SECONDS = 60

# Moving observations separated by more than this start a new driving period.
DRIVING_BREAK_SECONDS = 600


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

    vehicle_df["value"] = pd.to_numeric(
        vehicle_df["value"],
        errors="coerce",
    )

    vehicle_df = vehicle_df[
        vehicle_df["datetime_utc"].notna()
    ].copy()

    vehicle_df = vehicle_df.drop_duplicates(
        subset=[
            "vehicle",
            "datetime_utc",
            "diagnostic_id",
            "value",
        ]
    )

    return vehicle_df.reset_index(drop=True)


def find_speed_signal(vehicle_df):
    speed_names = {
        "engine road speed",
        "vehicle speed",
    }

    names = (
        vehicle_df["diagnostic_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return vehicle_df[
        names.isin(speed_names)
    ].copy()


def build_driving_periods(vehicle_df):
    speed_df = find_speed_signal(vehicle_df)

    if speed_df.empty:
        return pd.DataFrame()

    speed_df = (
        speed_df[
            speed_df["value"] > 0
        ]
        .sort_values("datetime_utc")
        .copy()
    )

    if speed_df.empty:
        return pd.DataFrame()

    time_diff = (
        speed_df["datetime_utc"]
        .diff()
        .dt.total_seconds()
    )

    speed_df["driving_period_id"] = (
        time_diff.gt(DRIVING_BREAK_SECONDS)
        .cumsum()
    )

    periods_df = (
        speed_df
        .groupby("driving_period_id")
        .agg(
            start_time=("datetime_utc", "min"),
            end_time=("datetime_utc", "max"),
            speed_record_count=("datetime_utc", "size"),
        )
        .reset_index()
    )

    return periods_df


def calculate_intervals_within_periods(
    signal_df,
    driving_periods,
):
    interval_frames = []
    driving_record_count = 0
    periods_with_signal = 0

    for period in driving_periods.itertuples(index=False):
        period_df = signal_df[
            signal_df["datetime_utc"].between(
                period.start_time,
                period.end_time,
            )
        ].copy()

        if period_df.empty:
            continue

        driving_record_count += len(period_df)

        timestamps = (
            period_df["datetime_utc"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )

        if timestamps.empty:
            continue

        periods_with_signal += 1

        if len(timestamps) < 2:
            continue

        intervals = (
            timestamps
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        if intervals.empty:
            continue

        interval_frames.append(
            pd.DataFrame({
                "interval_sec": intervals.values,
                "driving_period_id": period.driving_period_id,
            })
        )

    if not interval_frames:
        return (
            pd.DataFrame(
                columns=[
                    "interval_sec",
                    "driving_period_id",
                ]
            ),
            driving_record_count,
            periods_with_signal,
        )

    interval_df = pd.concat(
        interval_frames,
        ignore_index=True,
    )

    return (
        interval_df,
        driving_record_count,
        periods_with_signal,
    )


def analyze_signal_gaps(
    vehicle,
    diagnostic_id,
    diagnostic_name,
    signal_df,
    driving_periods,
):
    if driving_periods.empty:
        return {
            "vehicle": vehicle,
            "diagnostic_id": diagnostic_id,
            "diagnostic_name": diagnostic_name,
            "driving_period_count": 0,
            "periods_with_signal": 0,
            "driving_record_count": 0,
            "interval_count": 0,
            "median_interval_sec": None,
            "mean_interval_sec": None,
            "p95_interval_sec": None,
            "gap_threshold_sec": None,
            "gap_count": 0,
            "gap_pct": None,
            "max_gap_sec": None,
        }

    (
        interval_df,
        driving_record_count,
        periods_with_signal,
    ) = calculate_intervals_within_periods(
        signal_df,
        driving_periods,
    )

    if interval_df.empty:
        return {
            "vehicle": vehicle,
            "diagnostic_id": diagnostic_id,
            "diagnostic_name": diagnostic_name,
            "driving_period_count": len(driving_periods),
            "periods_with_signal": periods_with_signal,
            "driving_record_count": driving_record_count,
            "interval_count": 0,
            "median_interval_sec": None,
            "mean_interval_sec": None,
            "p95_interval_sec": None,
            "gap_threshold_sec": None,
            "gap_count": 0,
            "gap_pct": None,
            "max_gap_sec": None,
        }

    intervals = interval_df["interval_sec"]

    median_interval = intervals.median()
    mean_interval = intervals.mean()
    p95_interval = intervals.quantile(0.95)

    gap_threshold = max(
        median_interval * GAP_MULTIPLIER,
        MIN_GAP_SECONDS,
    )

    gaps = intervals[
        intervals > gap_threshold
    ]

    gap_pct = (
        len(gaps)
        / len(intervals)
        * 100
    )

    return {
        "vehicle": vehicle,
        "diagnostic_id": diagnostic_id,
        "diagnostic_name": diagnostic_name,
        "driving_period_count": len(driving_periods),
        "periods_with_signal": periods_with_signal,
        "driving_record_count": driving_record_count,
        "interval_count": len(intervals),
        "median_interval_sec": median_interval,
        "mean_interval_sec": mean_interval,
        "p95_interval_sec": p95_interval,
        "gap_threshold_sec": gap_threshold,
        "gap_count": len(gaps),
        "gap_pct": gap_pct,
        "max_gap_sec": (
            gaps.max()
            if not gaps.empty
            else 0
        ),
    }


def build_gap_report(
    vehicle,
    vehicle_df,
):
    driving_periods = build_driving_periods(
        vehicle_df
    )

    logger.info(
        "%s | identified %d driving periods",
        vehicle,
        len(driving_periods),
    )

    if driving_periods.empty:
        logger.warning(
            "%s | no driving periods found",
            vehicle,
        )
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
            analyze_signal_gaps(
                vehicle=vehicle,
                diagnostic_id=diagnostic_id,
                diagnostic_name=diagnostic_name,
                signal_df=signal_df,
                driving_periods=driving_periods,
            )
        )

    return pd.DataFrame(rows)


def save_report(report_df):
    report_df = report_df.sort_values(
        [
            "vehicle",
            "gap_pct",
            "diagnostic_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    report_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved gap report: %s",
        OUTPUT_FILE,
    )


def print_summary(report_df):
    if report_df.empty:
        return

    columns = [
        "vehicle",
        "diagnostic_name",
        "driving_record_count",
        "median_interval_sec",
        "gap_threshold_sec",
        "gap_count",
        "gap_pct",
        "max_gap_sec",
    ]

    print()
    print("Signal Gap Summary")
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
            "%s | starting gap analysis",
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

        report = build_gap_report(
            vehicle,
            vehicle_df,
        )

        if not report.empty:
            report_frames.append(report)

    if not report_frames:
        logger.warning(
            "No gap analysis results were generated."
        )
        return

    gap_df = pd.concat(
        report_frames,
        ignore_index=True,
    )

    save_report(gap_df)
    print_summary(gap_df)

    logger.info(
        "Gap analysis complete for %d vehicles",
        len(vehicles),
    )


if __name__ == "__main__":
    main()
# Evaluates minimum coverage rules for the current CACE V1 window candidate.
# The analysis tests fuel boundary tolerances and minimum RPM observation counts
# to determine which windows are reliable enough for target generation. Source
# data is not modified, and vehicle identifiers are anonymized in the reports.

import logging
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "vehicles.yaml"
)

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
    / "minimum_coverage_detail_private.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "minimum_coverage_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "minimum_coverage_analysis.log"
)


# Current leading window candidate: ±60 seconds.
HALF_WINDOW_SEC = 60

# Maximum allowed distance between the desired window boundary and the
# matched cumulative fuel observation.
FUEL_TOLERANCES_SEC = [
    30,
    60,
    90,
    120,
]

# Minimum number of RPM observations required within the full window.
RPM_MIN_COUNTS = [
    1,
    3,
    5,
]


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


def get_window(
    signal_df,
    start_time,
    end_time,
):
    if signal_df.empty:
        return signal_df

    return signal_df.loc[
        signal_df["datetime_utc"].between(
            start_time,
            end_time,
            inclusive="both",
        )
    ]


def find_nearest_observation(
    signal_df,
    target_time,
):
    if signal_df.empty:
        return None

    distances = (
        signal_df["datetime_utc"]
        .sub(target_time)
        .abs()
    )

    nearest_index = (
        distances.idxmin()
    )

    row = signal_df.loc[
        nearest_index
    ]

    return {
        "datetime_utc": row[
            "datetime_utc"
        ],
        "value": row[
            "value"
        ],
        "distance_sec": (
            abs(
                row["datetime_utc"]
                - target_time
            )
            .total_seconds()
        ),
    }


def build_anchor_base(
    vehicle_label,
    anchor_time,
    anchor_load,
    rpm_df,
    torque_df,
    fuel_df,
):
    window_start = (
        anchor_time
        - pd.Timedelta(
            seconds=HALF_WINDOW_SEC
        )
    )

    window_end = (
        anchor_time
        + pd.Timedelta(
            seconds=HALF_WINDOW_SEC
        )
    )

    rpm_window = get_window(
        rpm_df,
        window_start,
        window_end,
    )

    torque_window = get_window(
        torque_df,
        window_start,
        window_end,
    )

    start_match = (
        find_nearest_observation(
            fuel_df,
            window_start,
        )
    )

    end_match = (
        find_nearest_observation(
            fuel_df,
            window_end,
        )
    )

    result = {
        "vehicle": vehicle_label,
        "anchor_time_utc": anchor_time,
        "anchor_load": anchor_load,
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "rpm_count": len(
            rpm_window
        ),
        "torque_available": (
            not torque_window.empty
        ),
        "fuel_start_time_utc": None,
        "fuel_end_time_utc": None,
        "fuel_start": None,
        "fuel_end": None,
        "start_boundary_distance_sec": None,
        "end_boundary_distance_sec": None,
        "actual_fuel_interval_sec": None,
        "actual_fuel_used": None,
        "fuel_time_order_valid": False,
        "fuel_delta_valid": False,
        "fuel_pair_distinct": False,
    }

    if (
        start_match is None
        or end_match is None
    ):
        return result

    fuel_start_time = (
        start_match["datetime_utc"]
    )

    fuel_end_time = (
        end_match["datetime_utc"]
    )

    fuel_start = (
        start_match["value"]
    )

    fuel_end = (
        end_match["value"]
    )

    actual_fuel_interval_sec = (
        fuel_end_time
        - fuel_start_time
    ).total_seconds()

    actual_fuel_used = (
        fuel_end
        - fuel_start
    )

    result.update({
        "fuel_start_time_utc": (
            fuel_start_time
        ),
        "fuel_end_time_utc": (
            fuel_end_time
        ),
        "fuel_start": fuel_start,
        "fuel_end": fuel_end,
        "start_boundary_distance_sec": (
            start_match[
                "distance_sec"
            ]
        ),
        "end_boundary_distance_sec": (
            end_match[
                "distance_sec"
            ]
        ),
        "actual_fuel_interval_sec": (
            actual_fuel_interval_sec
        ),
        "actual_fuel_used": (
            actual_fuel_used
        ),
        "fuel_time_order_valid": (
            actual_fuel_interval_sec > 0
        ),
        "fuel_delta_valid": (
            actual_fuel_used >= 0
        ),
        "fuel_pair_distinct": (
            fuel_start_time
            != fuel_end_time
        ),
    })

    return result


def build_vehicle_base(
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

    torque_df = select_signal(
        vehicle_df,
        "engine_torque",
    )

    fuel_df = select_signal(
        vehicle_df,
        "total_fuel_used",
    )

    if load_df.empty:
        logger.warning(
            "%s | skipped: Engine Load unavailable",
            vehicle_label,
        )
        return pd.DataFrame()

    if rpm_df.empty:
        logger.warning(
            "%s | skipped: RPM unavailable",
            vehicle_label,
        )
        return pd.DataFrame()

    if fuel_df.empty:
        logger.warning(
            "%s | skipped: Total Fuel Used unavailable",
            vehicle_label,
        )
        return pd.DataFrame()

    rows = []

    for anchor in load_df.itertuples(
        index=False
    ):
        rows.append(
            build_anchor_base(
                vehicle_label=vehicle_label,
                anchor_time=anchor.datetime_utc,
                anchor_load=anchor.value,
                rpm_df=rpm_df,
                torque_df=torque_df,
                fuel_df=fuel_df,
            )
        )

    return pd.DataFrame(
        rows
    )


def evaluate_rules(base_df):
    rows = []

    for (
        fuel_tolerance_sec
    ) in FUEL_TOLERANCES_SEC:

        for (
            rpm_min_count
        ) in RPM_MIN_COUNTS:

            evaluated = (
                base_df.copy()
            )

            evaluated[
                "fuel_tolerance_sec"
            ] = fuel_tolerance_sec

            evaluated[
                "rpm_min_count"
            ] = rpm_min_count

            evaluated[
                "rpm_coverage_valid"
            ] = (
                evaluated[
                    "rpm_count"
                ]
                >= rpm_min_count
            )

            evaluated[
                "fuel_boundary_valid"
            ] = (
                evaluated[
                    "start_boundary_distance_sec"
                ].le(
                    fuel_tolerance_sec
                )
                &
                evaluated[
                    "end_boundary_distance_sec"
                ].le(
                    fuel_tolerance_sec
                )
            )

            evaluated[
                "valid_window"
            ] = (
                evaluated[
                    "rpm_coverage_valid"
                ]
                &
                evaluated[
                    "torque_available"
                ]
                &
                evaluated[
                    "fuel_boundary_valid"
                ]
                &
                evaluated[
                    "fuel_pair_distinct"
                ]
                &
                evaluated[
                    "fuel_time_order_valid"
                ]
                &
                evaluated[
                    "fuel_delta_valid"
                ]
            )

            rows.append(
                evaluated
            )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_summary(
    detail_df,
):
    summary = (
        detail_df
        .groupby(
            [
                "vehicle",
                "fuel_tolerance_sec",
                "rpm_min_count",
            ],
            as_index=False,
        )
        .agg(
            anchor_count=(
                "anchor_time_utc",
                "size",
            ),
            valid_window_count=(
                "valid_window",
                "sum",
            ),
            valid_window_pct=(
                "valid_window",
                "mean",
            ),
            rpm_valid_pct=(
                "rpm_coverage_valid",
                "mean",
            ),
            torque_valid_pct=(
                "torque_available",
                "mean",
            ),
            fuel_boundary_valid_pct=(
                "fuel_boundary_valid",
                "mean",
            ),
            median_rpm_count=(
                "rpm_count",
                "median",
            ),
            median_start_boundary_distance_sec=(
                "start_boundary_distance_sec",
                "median",
            ),
            median_end_boundary_distance_sec=(
                "end_boundary_distance_sec",
                "median",
            ),
            median_actual_fuel_interval_sec=(
                "actual_fuel_interval_sec",
                "median",
            ),
            median_actual_fuel_used=(
                "actual_fuel_used",
                "median",
            ),
        )
    )

    percentage_columns = [
        "valid_window_pct",
        "rpm_valid_pct",
        "torque_valid_pct",
        "fuel_boundary_valid_pct",
    ]

    for column in percentage_columns:
        summary[column] = (
            summary[column]
            .mul(100)
            .round(2)
        )

    numeric_columns = [
        "median_rpm_count",
        "median_start_boundary_distance_sec",
        "median_end_boundary_distance_sec",
        "median_actual_fuel_interval_sec",
        "median_actual_fuel_used",
    ]

    for column in numeric_columns:
        summary[column] = (
            pd.to_numeric(
                summary[column],
                errors="coerce",
            )
            .round(4)
        )

    return (
        summary
        .sort_values(
            [
                "vehicle",
                "fuel_tolerance_sec",
                "rpm_min_count",
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
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved minimum coverage detail report: %s",
        DETAIL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved minimum coverage summary report: %s",
        SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def print_summary(
    summary_df,
):
    if summary_df.empty:
        return

    columns = [
        "vehicle",
        "fuel_tolerance_sec",
        "rpm_min_count",
        "anchor_count",
        "valid_window_count",
        "valid_window_pct",
        "fuel_boundary_valid_pct",
        "rpm_valid_pct",
        "median_rpm_count",
    ]

    print()
    print(
        "CACE Minimum Coverage Summary"
    )
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
        "Starting CACE minimum coverage analysis"
    )

    vehicles = (
        load_vehicle_config()
    )

    vehicle_labels = (
        build_vehicle_labels(
            vehicles
        )
    )

    base_frames = []

    for vehicle in vehicles:
        vehicle_label = (
            vehicle_labels[
                vehicle
            ]
        )

        logger.info(
            "%s | building coverage base",
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

        base_df = (
            build_vehicle_base(
                vehicle_label,
                vehicle_df,
            )
        )

        if base_df.empty:
            continue

        base_frames.append(
            base_df
        )

        logger.info(
            "%s | evaluated %d Engine Load anchors",
            vehicle_label,
            len(base_df),
        )

    if not base_frames:
        raise RuntimeError(
            "No minimum coverage analysis data was generated"
        )

    base_df = pd.concat(
        base_frames,
        ignore_index=True,
    )

    detail_df = evaluate_rules(
        base_df
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
        "Minimum coverage analysis complete | "
        "vehicles=%d | anchors=%d",
        base_df[
            "vehicle"
        ].nunique(),
        len(base_df),
    )


if __name__ == "__main__":
    main()
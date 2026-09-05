# Evaluates how cumulative Total Fuel Used can be matched to the start and end
# boundaries of each candidate CACE window. The analysis checks boundary distance,
# fuel observation order, negative fuel deltas, and target coverage before the
# final window rule is selected. Source data is not modified.

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
    / "fuel_target_matching_detail_private.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "fuel_target_matching_summary_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "fuel_target_matching_analysis.log"
)


# Candidate half-window sizes selected from the core-signal window analysis.
# The final CACE V1 window will be selected based on fuel target coverage
# and the quality of cumulative fuel matching at each window boundary.
WINDOWS = {
    "pm_30s": 30,
    "pm_45s": 45,
    "pm_60s": 60,
    "pm_90s": 90,
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
        distances
        .idxmin()
    )

    row = signal_df.loc[
        nearest_index
    ]

    distance_sec = (
        abs(
            row["datetime_utc"]
            - target_time
        )
        .total_seconds()
    )

    return {
        "datetime_utc": row[
            "datetime_utc"
        ],
        "value": row[
            "value"
        ],
        "distance_sec": distance_sec,
    }


def analyze_fuel_target(
    vehicle_label,
    anchor_time,
    anchor_load,
    window_name,
    half_window_sec,
    fuel_df,
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
        "window_name": window_name,
        "half_window_sec": half_window_sec,
        "total_window_sec": half_window_sec * 2,
        "anchor_load": anchor_load,
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "fuel_start_time_utc": None,
        "fuel_end_time_utc": None,
        "fuel_start": None,
        "fuel_end": None,
        "start_boundary_distance_sec": None,
        "end_boundary_distance_sec": None,
        "actual_fuel_used": None,
        "actual_fuel_interval_sec": None,
        "derived_fuel_rate_per_hour": None,
        "fuel_pair_available": False,
        "fuel_time_order_valid": False,
        "fuel_delta_valid": False,
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

    actual_interval_sec = (
        fuel_end_time
        - fuel_start_time
    ).total_seconds()

    actual_fuel_used = (
        fuel_end
        - fuel_start
    )

    time_order_valid = (
        actual_interval_sec > 0
    )

    fuel_delta_valid = (
        actual_fuel_used >= 0
    )

    fuel_pair_available = (
        fuel_start_time
        != fuel_end_time
    )

    derived_fuel_rate = None

    if (
        fuel_pair_available
        and time_order_valid
        and fuel_delta_valid
        and actual_interval_sec > 0
    ):
        derived_fuel_rate = (
            actual_fuel_used
            / actual_interval_sec
            * 3600
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
        "actual_fuel_used": (
            actual_fuel_used
        ),
        "actual_fuel_interval_sec": (
            actual_interval_sec
        ),
        "derived_fuel_rate_per_hour": (
            derived_fuel_rate
        ),
        "fuel_pair_available": (
            fuel_pair_available
        ),
        "fuel_time_order_valid": (
            time_order_valid
        ),
        "fuel_delta_valid": (
            fuel_delta_valid
        ),
    })

    return result


def build_vehicle_detail(
    vehicle_label,
    vehicle_df,
):
    load_df = select_signal(
        vehicle_df,
        "engine_load",
    )

    fuel_df = select_signal(
        vehicle_df,
        "total_fuel_used",
    )

    if load_df.empty:
        logger.warning(
            "%s | skipped: Engine Load anchors unavailable",
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
        for (
            window_name,
            half_window_sec,
        ) in WINDOWS.items():

            rows.append(
                analyze_fuel_target(
                    vehicle_label=vehicle_label,
                    anchor_time=anchor.datetime_utc,
                    anchor_load=anchor.value,
                    window_name=window_name,
                    half_window_sec=half_window_sec,
                    fuel_df=fuel_df,
                )
            )

    return pd.DataFrame(
        rows
    )


def build_summary(detail_df):
    if detail_df.empty:
        return pd.DataFrame()

    detail_df = detail_df.copy()

    detail_df["valid_target"] = (
        detail_df[
            "fuel_pair_available"
        ]
        & detail_df[
            "fuel_time_order_valid"
        ]
        & detail_df[
            "fuel_delta_valid"
        ]
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
            fuel_pair_coverage_pct=(
                "fuel_pair_available",
                "mean",
            ),
            valid_target_pct=(
                "valid_target",
                "mean",
            ),
            negative_fuel_delta_count=(
                "fuel_delta_valid",
                lambda values: (
                    (~values).sum()
                ),
            ),
            median_start_boundary_distance_sec=(
                "start_boundary_distance_sec",
                "median",
            ),
            p95_start_boundary_distance_sec=(
                "start_boundary_distance_sec",
                lambda values: (
                    values.quantile(
                        0.95
                    )
                ),
            ),
            median_end_boundary_distance_sec=(
                "end_boundary_distance_sec",
                "median",
            ),
            p95_end_boundary_distance_sec=(
                "end_boundary_distance_sec",
                lambda values: (
                    values.quantile(
                        0.95
                    )
                ),
            ),
            median_actual_fuel_interval_sec=(
                "actual_fuel_interval_sec",
                "median",
            ),
            median_actual_fuel_used=(
                "actual_fuel_used",
                "median",
            ),
            median_derived_fuel_rate_per_hour=(
                "derived_fuel_rate_per_hour",
                "median",
            ),
        )
    )

    percentage_columns = [
        "fuel_pair_coverage_pct",
        "valid_target_pct",
    ]

    for column in percentage_columns:
        summary[column] = (
            summary[column]
            .mul(100)
            .round(2)
        )

    numeric_columns = [
        "median_start_boundary_distance_sec",
        "p95_start_boundary_distance_sec",
        "median_end_boundary_distance_sec",
        "p95_end_boundary_distance_sec",
        "median_actual_fuel_interval_sec",
        "median_actual_fuel_used",
        "median_derived_fuel_rate_per_hour",
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
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved fuel target detail report: %s",
        DETAIL_FILE.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved fuel target summary report: %s",
        SUMMARY_FILE.relative_to(
            PROJECT_ROOT
        ),
    )


def print_summary(summary_df):
    if summary_df.empty:
        return

    columns = [
        "vehicle",
        "window_name",
        "anchor_count",
        "fuel_pair_coverage_pct",
        "valid_target_pct",
        "negative_fuel_delta_count",
        "median_start_boundary_distance_sec",
        "median_end_boundary_distance_sec",
        "median_actual_fuel_interval_sec",
        "median_actual_fuel_used",
    ]

    print()
    print(
        "CACE Fuel Target Matching Summary"
    )

    print(
        summary_df[
            columns
        ].to_string(
            index=False
        )
    )


def main():
    logger.info(
        "Starting CACE fuel target matching analysis"
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
            "%s | analyzing fuel target matching",
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
            "No fuel target matching results were generated"
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
        "Fuel target matching analysis complete | "
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
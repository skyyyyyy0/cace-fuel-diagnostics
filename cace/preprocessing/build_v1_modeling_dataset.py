# Builds the candidate CACE V1 modeling dataset using the window, feature,
# target, and minimum-coverage rules validated during Phase 4 and registered
# in the project configuration. Shared loading and matching logic is imported
# from cace.utils.modeling_windows to keep the modeling pipeline consistent
# with the earlier design analysis and avoid duplicate implementations.

import logging
from pathlib import Path

import pandas as pd

from cace.utils.modeling_windows import (
    build_vehicle_labels,
    find_nearest_observation,
    get_window,
    load_vehicle_data,
    load_vehicle_ids,
    load_yaml,
    range_is_valid,
    select_signal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "vehicles.yaml"
)

FEATURE_REGISTRY = (
    PROJECT_ROOT
    / "config"
    / "feature_registry.yaml"
)

QUALITY_RULES = (
    PROJECT_ROOT
    / "config"
    / "quality_rules.yaml"
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
    / "data"
    / "processed"
    / "cace_v1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling_dataset"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)


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


LOG_FILE = (
    LOG_DIR
    / "build_v1_modeling_dataset.log"
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


def load_v1_rules():
    registry = load_yaml(
        FEATURE_REGISTRY
    )

    quality_config = load_yaml(
        QUALITY_RULES
    )

    window_config = registry.get(
        "window",
        {},
    )

    features = registry.get(
        "features",
        {},
    )

    targets = registry.get(
        "targets",
        {},
    )

    if not window_config:
        raise ValueError(
            "Window configuration is missing "
            "from feature_registry.yaml"
        )

    half_window_sec = window_config.get(
        "half_window_sec"
    )

    rpm_min_count = (
        features
        .get("avg_rpm", {})
        .get("minimum_observations")
    )

    fuel_tolerance_sec = (
        targets
        .get("actual_fuel_used", {})
        .get("fuel_boundary_tolerance_sec")
    )

    if half_window_sec is None:
        raise ValueError(
            "window.half_window_sec is required"
        )

    if rpm_min_count is None:
        raise ValueError(
            "avg_rpm.minimum_observations is required"
        )

    if fuel_tolerance_sec is None:
        raise ValueError(
            "actual_fuel_used.fuel_boundary_tolerance_sec "
            "is required"
        )

    return {
        "dataset_version": (
            registry.get(
                "version",
                "1.0",
            )
        ),
        "half_window_sec": int(
            half_window_sec
        ),
        "rpm_min_count": int(
            rpm_min_count
        ),
        "fuel_tolerance_sec": int(
            fuel_tolerance_sec
        ),
        "quality_rules": (
            quality_config.get(
                "rules",
                {},
            )
        ),
    }


def nearest_torque_in_window(
    torque_df,
    anchor_time,
    half_window_sec,
):
    match = find_nearest_observation(
        torque_df,
        anchor_time,
    )

    if match is None:
        return None

    if (
        match["distance_sec"]
        > half_window_sec
    ):
        return None

    return match


def build_anchor_row(
    vehicle_label,
    anchor_time,
    engine_load,
    rpm_df,
    torque_df,
    fuel_df,
    rules,
):
    half_window_sec = rules[
        "half_window_sec"
    ]

    rpm_min_count = rules[
        "rpm_min_count"
    ]

    fuel_tolerance_sec = rules[
        "fuel_tolerance_sec"
    ]

    quality_rules = rules[
        "quality_rules"
    ]

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

    rpm_window = get_window(
        rpm_df,
        window_start,
        window_end,
    )

    rpm_count = len(
        rpm_window
    )

    avg_rpm = (
        rpm_window["value"].mean()
        if rpm_count > 0
        else None
    )

    torque_match = (
        nearest_torque_in_window(
            torque_df=torque_df,
            anchor_time=anchor_time,
            half_window_sec=half_window_sec,
        )
    )

    engine_torque = (
        torque_match["value"]
        if torque_match is not None
        else None
    )

    torque_distance_sec = (
        torque_match["distance_sec"]
        if torque_match is not None
        else None
    )

    fuel_start_match = (
        find_nearest_observation(
            fuel_df,
            window_start,
        )
    )

    fuel_end_match = (
        find_nearest_observation(
            fuel_df,
            window_end,
        )
    )

    fuel_start = None
    fuel_end = None

    fuel_start_time = None
    fuel_end_time = None

    start_boundary_distance_sec = None
    end_boundary_distance_sec = None

    actual_fuel_interval_sec = None
    actual_fuel_used = None
    derived_fuel_rate = None

    if (
        fuel_start_match is not None
        and fuel_end_match is not None
    ):
        fuel_start = (
            fuel_start_match[
                "value"
            ]
        )

        fuel_end = (
            fuel_end_match[
                "value"
            ]
        )

        fuel_start_time = (
            fuel_start_match[
                "datetime_utc"
            ]
        )

        fuel_end_time = (
            fuel_end_match[
                "datetime_utc"
            ]
        )

        start_boundary_distance_sec = (
            fuel_start_match[
                "distance_sec"
            ]
        )

        end_boundary_distance_sec = (
            fuel_end_match[
                "distance_sec"
            ]
        )

        actual_fuel_interval_sec = (
            fuel_end_time
            - fuel_start_time
        ).total_seconds()

        actual_fuel_used = (
            fuel_end
            - fuel_start
        )

        if (
            actual_fuel_interval_sec > 0
            and actual_fuel_used >= 0
        ):
            derived_fuel_rate = (
                actual_fuel_used
                / actual_fuel_interval_sec
                * 3600
            )

    rpm_load = None

    if (
        avg_rpm is not None
        and engine_load is not None
    ):
        rpm_load = (
            avg_rpm
            * engine_load
        )

    rpm_count_valid = (
        rpm_count
        >= rpm_min_count
    )

    torque_available = (
        torque_match is not None
    )

    fuel_pair_distinct = (
        fuel_start_time is not None
        and fuel_end_time is not None
        and fuel_start_time
        != fuel_end_time
    )

    fuel_time_valid = (
        actual_fuel_interval_sec is not None
        and actual_fuel_interval_sec > 0
    )

    fuel_delta_valid = (
        actual_fuel_used is not None
        and actual_fuel_used >= 0
    )

    fuel_boundary_valid = (
        start_boundary_distance_sec is not None
        and end_boundary_distance_sec is not None
        and start_boundary_distance_sec
        <= fuel_tolerance_sec
        and end_boundary_distance_sec
        <= fuel_tolerance_sec
    )

    window_rule_valid = (
        rpm_count_valid
        and torque_available
        and fuel_pair_distinct
        and fuel_time_valid
        and fuel_delta_valid
        and fuel_boundary_valid
    )

    rpm_range_valid = (
        avg_rpm is not None
        and range_is_valid(
            avg_rpm,
            quality_rules.get(
                "rpm",
                {},
            ),
        )
    )

    load_range_valid = (
        engine_load is not None
        and range_is_valid(
            engine_load,
            quality_rules.get(
                "engine_load",
                {},
            ),
        )
    )

    torque_range_valid = (
        engine_torque is not None
        and range_is_valid(
            engine_torque,
            quality_rules.get(
                "engine_torque",
                {},
            ),
        )
    )

    core_quality_flag = (
        rpm_range_valid
        and load_range_valid
        and torque_range_valid
    )

    return {
        "vehicle": vehicle_label,
        "anchor_time_utc": anchor_time,
        "window_start_utc": window_start,
        "window_end_utc": window_end,

        "avg_rpm": avg_rpm,
        "rpm_observation_count": (
            rpm_count
        ),

        "engine_load": engine_load,
        "engine_torque": engine_torque,
        "torque_distance_sec": (
            torque_distance_sec
        ),

        "rpm_load": rpm_load,

        "fuel_start": fuel_start,
        "fuel_end": fuel_end,
        "fuel_start_time_utc": (
            fuel_start_time
        ),
        "fuel_end_time_utc": (
            fuel_end_time
        ),

        "fuel_start_boundary_gap_sec": (
            start_boundary_distance_sec
        ),
        "fuel_end_boundary_gap_sec": (
            end_boundary_distance_sec
        ),

        "actual_fuel_interval_sec": (
            actual_fuel_interval_sec
        ),
        "actual_fuel_used": (
            actual_fuel_used
        ),
        "derived_fuel_rate": (
            derived_fuel_rate
        ),

        "rpm_count_valid": (
            rpm_count_valid
        ),
        "torque_available": (
            torque_available
        ),
        "fuel_boundary_valid": (
            fuel_boundary_valid
        ),
        "fuel_pair_distinct": (
            fuel_pair_distinct
        ),
        "fuel_time_valid": (
            fuel_time_valid
        ),
        "fuel_delta_valid": (
            fuel_delta_valid
        ),

        "rpm_range_valid": (
            rpm_range_valid
        ),
        "load_range_valid": (
            load_range_valid
        ),
        "torque_range_valid": (
            torque_range_valid
        ),

        "window_rule_valid": (
            window_rule_valid
        ),
        "core_quality_flag": (
            core_quality_flag
        ),
    }


def build_vehicle_dataset(
    vehicle_label,
    vehicle_df,
    rules,
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

    if torque_df.empty:
        logger.warning(
            "%s | skipped: Engine Torque unavailable",
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
            build_anchor_row(
                vehicle_label=vehicle_label,
                anchor_time=anchor.datetime_utc,
                engine_load=anchor.value,
                rpm_df=rpm_df,
                torque_df=torque_df,
                fuel_df=fuel_df,
                rules=rules,
            )
        )

    return pd.DataFrame(
        rows
    )


def build_vehicle_summary(
    dataset,
):
    if dataset.empty:
        return pd.DataFrame()

    return (
        dataset
        .groupby(
            "vehicle",
            as_index=False,
        )
        .agg(
            total_anchor_count=(
                "anchor_time_utc",
                "size",
            ),
            valid_window_count=(
                "window_rule_valid",
                "sum",
            ),
            quality_pass_count=(
                "core_quality_flag",
                "sum",
            ),
            median_rpm_count=(
                "rpm_observation_count",
                "median",
            ),
            median_actual_fuel_used=(
                "actual_fuel_used",
                "median",
            ),
        )
    )


def save_outputs(
    dataset,
    summary,
    dataset_version,
):
    dataset_name = (
        f"CACE_Dataset_v"
        f"{dataset_version}"
        f"_candidate_private.csv"
    )

    dataset_path = (
        OUTPUT_DIR
        / dataset_name
    )

    summary_path = (
        REPORT_DIR
        / "cace_v1_candidate_summary_private.csv"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved candidate dataset: %s",
        dataset_path.relative_to(
            PROJECT_ROOT
        ),
    )

    logger.info(
        "Saved dataset summary: %s",
        summary_path.relative_to(
            PROJECT_ROOT
        ),
    )


def main():
    logger.info(
        "Starting CACE V1 candidate dataset build"
    )

    rules = load_v1_rules()

    vehicles = load_vehicle_ids(
        VEHICLE_CONFIG
    )

    vehicle_labels = (
        build_vehicle_labels(
            vehicles
        )
    )

    vehicle_frames = []

    for vehicle in vehicles:
        vehicle_label = (
            vehicle_labels[
                vehicle
            ]
        )

        logger.info(
            "%s | building V1 observations",
            vehicle_label,
        )

        vehicle_df = (
            load_vehicle_data(
                INPUT_DIR,
                vehicle,
            )
        )

        if vehicle_df.empty:
            logger.warning(
                "%s | skipped: no processed CACE signals",
                vehicle_label,
            )
            continue

        vehicle_dataset = (
            build_vehicle_dataset(
                vehicle_label=vehicle_label,
                vehicle_df=vehicle_df,
                rules=rules,
            )
        )

        if vehicle_dataset.empty:
            continue

        vehicle_frames.append(
            vehicle_dataset
        )

        logger.info(
            "%s | anchors=%d | valid_windows=%d",
            vehicle_label,
            len(vehicle_dataset),
            int(
                vehicle_dataset[
                    "window_rule_valid"
                ].sum()
            ),
        )

    if not vehicle_frames:
        raise RuntimeError(
            "No CACE V1 observations were generated"
        )

    dataset = pd.concat(
        vehicle_frames,
        ignore_index=True,
    )

    summary = (
        build_vehicle_summary(
            dataset
        )
    )

    save_outputs(
        dataset=dataset,
        summary=summary,
        dataset_version=rules[
            "dataset_version"
        ],
    )

    logger.info(
        "CACE V1 candidate dataset build complete | "
        "vehicles=%d | anchors=%d | valid_windows=%d",
        dataset[
            "vehicle"
        ].nunique(),
        len(dataset),
        int(
            dataset[
                "window_rule_valid"
            ].sum()
        ),
    )


if __name__ == "__main__":
    main()
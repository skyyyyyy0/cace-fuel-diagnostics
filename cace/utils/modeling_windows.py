# Provides reusable data-loading, windowing, and signal-matching helpers used by
# the CACE modeling pipeline. These functions preserve the Phase 4 window rules
# without interpolation or forward filling so downstream datasets use the same
# validated observation logic.

from pathlib import Path

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "vehicle",
    "datetime_utc",
    "signal_key",
    "value",
}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file) or {}


def load_vehicle_ids(config_path: Path) -> list[str]:
    config = load_yaml(
        config_path
    )

    vehicles = config.get(
        "vehicles",
        [],
    )

    if not vehicles:
        raise ValueError(
            f"No vehicles found in {config_path.name}"
        )

    return [
        str(vehicle).strip()
        for vehicle in vehicles
        if str(vehicle).strip()
    ]


def build_vehicle_labels(
    vehicles: list[str],
) -> dict[str, str]:
    return {
        vehicle: f"VEHICLE_{index:02d}"
        for index, vehicle in enumerate(
            vehicles,
            start=1,
        )
    }


def find_vehicle_files(
    input_dir: Path,
    vehicle: str,
) -> list[Path]:
    vehicle_dir = (
        input_dir
        / f"vehicle_{vehicle}"
    )

    if not vehicle_dir.exists():
        return []

    return sorted(
        vehicle_dir.rglob(
            "statusdata_*.csv"
        )
    )


def read_signal_file(
    path: Path,
) -> pd.DataFrame:
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


def load_vehicle_data(
    input_dir: Path,
    vehicle: str,
) -> pd.DataFrame:
    files = find_vehicle_files(
        input_dir,
        vehicle,
    )

    if not files:
        return pd.DataFrame()

    frames = [
        read_signal_file(path)
        for path in files
    ]

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
        df.sort_values(
            "datetime_utc"
        )
        .reset_index(
            drop=True
        )
    )


def select_signal(
    df: pd.DataFrame,
    signal_key: str,
) -> pd.DataFrame:
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
    signal_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df

    return signal_df.loc[
        signal_df[
            "datetime_utc"
        ].between(
            start_time,
            end_time,
            inclusive="both",
        )
    ]


def find_nearest_observation(
    signal_df: pd.DataFrame,
    target_time: pd.Timestamp,
) -> dict | None:
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

    distance_sec = abs(
        (
            row["datetime_utc"]
            - target_time
        ).total_seconds()
    )

    return {
        "datetime_utc": row[
            "datetime_utc"
        ],
        "value": row[
            "value"
        ],
        "distance_sec": (
            distance_sec
        ),
    }


def range_is_valid(
    value: float,
    rule: dict,
) -> bool:
    if not rule:
        return True

    if (
        rule.get("type")
        != "range"
    ):
        return True

    minimum = rule.get(
        "min"
    )

    maximum = rule.get(
        "max"
    )

    if (
        minimum is not None
        and value < minimum
    ):
        return False

    if (
        maximum is not None
        and value > maximum
    ):
        return False

    return True
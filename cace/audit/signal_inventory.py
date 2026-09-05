# Reviews the GeoTab StatusData collected for each CACE vehicle and summarizes
# which core, operating, and DPF signals are actually available in the raw data.
# The reports are used to confirm signal coverage before building the CACE
# modeling dataset and can be re-run as new vehicle data is collected.

import logging
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CONFIG = PROJECT_ROOT / "config" / "vehicles.yaml"
SIGNAL_CONFIG = PROJECT_ROOT / "config" / "signals.yaml"

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

SIGNAL_INVENTORY_FILE = OUTPUT_DIR / "signal_inventory.csv"
CORE_REPORT_FILE = OUTPUT_DIR / "core_pid_availability.csv"
ADDITIONAL_REPORT_FILE = OUTPUT_DIR / "additional_pid_availability.csv"
DPF_REPORT_FILE = OUTPUT_DIR / "dpf_pid_availability.csv"

LOG_FILE = LOG_DIR / "signal_inventory.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
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


def load_signal_groups():
    config = load_yaml(SIGNAL_CONFIG)

    required_groups = [
        "core",
        "additional",
        "dpf",
    ]

    missing_groups = [
        group
        for group in required_groups
        if group not in config
    ]

    if missing_groups:
        raise ValueError(
            "Missing signal groups in signals.yaml: "
            + ", ".join(missing_groups)
        )

    return config


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

    # Overlapping incremental extractions can contain the same StatusData row.
    vehicle_df = vehicle_df.drop_duplicates(
        subset=[
            "vehicle",
            "datetime_utc",
            "diagnostic_id",
            "value",
        ]
    )

    return vehicle_df.reset_index(drop=True)


def build_full_inventory(vehicle, vehicle_df):
    if vehicle_df.empty:
        return pd.DataFrame()

    inventory_df = (
        vehicle_df
        .groupby(
            [
                "diagnostic_id",
                "diagnostic_name",
            ],
            dropna=False,
        )
        .agg(
            record_count=(
                "diagnostic_name",
                "size",
            ),
            first_datetime_utc=(
                "datetime_utc",
                "min",
            ),
            last_datetime_utc=(
                "datetime_utc",
                "max",
            ),
            unique_value_count=(
                "value",
                "nunique",
            ),
            min_value=(
                "value",
                "min",
            ),
            max_value=(
                "value",
                "max",
            ),
            avg_value=(
                "value",
                "mean",
            ),
            unit=(
                "unit",
                "first",
            ),
        )
        .reset_index()
    )

    inventory_df.insert(
        0,
        "vehicle",
        vehicle,
    )

    return inventory_df.sort_values(
        "record_count",
        ascending=False,
    )


def match_signal(vehicle_df, aliases):
    if vehicle_df.empty:
        return pd.DataFrame()

    aliases = {
        str(alias).strip().lower()
        for alias in aliases
    }

    diagnostic_names = (
        vehicle_df["diagnostic_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return vehicle_df[
        diagnostic_names.isin(aliases)
    ].copy()


def build_group_report(
    vehicle,
    vehicle_df,
    group_name,
    signal_group,
):
    rows = []

    for signal_key, settings in signal_group.items():
        aliases = settings.get(
            "aliases",
            [],
        )

        display_name = settings.get(
            "display_name",
            signal_key,
        )

        matched_df = match_signal(
            vehicle_df,
            aliases,
        )

        matched_diagnostics = sorted(
            matched_df["diagnostic_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rows.append({
            "vehicle": vehicle,
            "signal_group": group_name,
            "signal_key": signal_key,
            "signal_name": display_name,
            "available": not matched_df.empty,
            "record_count": len(matched_df),
            "matched_diagnostics": " | ".join(
                matched_diagnostics
            ),
            "first_datetime_utc": (
                matched_df["datetime_utc"].min()
                if not matched_df.empty
                else pd.NaT
            ),
            "last_datetime_utc": (
                matched_df["datetime_utc"].max()
                if not matched_df.empty
                else pd.NaT
            ),
            "unit": (
                matched_df["unit"]
                .dropna()
                .iloc[0]
                if (
                    not matched_df.empty
                    and not matched_df["unit"].dropna().empty
                )
                else None
            ),
        })

    return pd.DataFrame(rows)


def save_report(df, path):
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved report: %s",
        path,
    )


def print_group_summary(df, title):
    if df.empty:
        return

    summary = df.pivot(
        index="vehicle",
        columns="signal_name",
        values="available",
    )

    summary = summary.map(
        lambda value: (
            "Yes"
            if pd.notna(value) and bool(value)
            else "No"
            if pd.notna(value)
            else ""
        )
    )

    print()
    print(title)
    print(summary.to_string())


def main():
    vehicles = load_target_vehicles()
    signal_groups = load_signal_groups()

    inventory_frames = []
    core_frames = []
    additional_frames = []
    dpf_frames = []

    for vehicle in vehicles:
        logger.info(
            "%s | starting signal inventory",
            vehicle,
        )

        vehicle_df = load_vehicle_data(vehicle)

        if vehicle_df.empty:
            logger.warning(
                "%s | no raw StatusData found",
                vehicle,
            )
        else:
            logger.info(
                "%s | loaded %d unique StatusData rows",
                vehicle,
                len(vehicle_df),
            )

        inventory_frames.append(
            build_full_inventory(
                vehicle,
                vehicle_df,
            )
        )

        core_frames.append(
            build_group_report(
                vehicle=vehicle,
                vehicle_df=vehicle_df,
                group_name="core",
                signal_group=signal_groups["core"],
            )
        )

        additional_frames.append(
            build_group_report(
                vehicle=vehicle,
                vehicle_df=vehicle_df,
                group_name="additional",
                signal_group=signal_groups["additional"],
            )
        )

        dpf_frames.append(
            build_group_report(
                vehicle=vehicle,
                vehicle_df=vehicle_df,
                group_name="dpf",
                signal_group=signal_groups["dpf"],
            )
        )

    inventory_df = pd.concat(
        inventory_frames,
        ignore_index=True,
    )

    core_df = pd.concat(
        core_frames,
        ignore_index=True,
    )

    additional_df = pd.concat(
        additional_frames,
        ignore_index=True,
    )

    dpf_df = pd.concat(
        dpf_frames,
        ignore_index=True,
    )

    save_report(
        inventory_df,
        SIGNAL_INVENTORY_FILE,
    )

    save_report(
        core_df,
        CORE_REPORT_FILE,
    )

    save_report(
        additional_df,
        ADDITIONAL_REPORT_FILE,
    )

    save_report(
        dpf_df,
        DPF_REPORT_FILE,
    )

    print_group_summary(
        core_df,
        "Core PID Availability",
    )

    print_group_summary(
        additional_df,
        "Additional PID Availability",
    )

    print_group_summary(
        dpf_df,
        "DPF PID Availability",
    )

    logger.info(
        "Signal inventory complete for %d vehicles",
        len(vehicles),
    )


if __name__ == "__main__":
    main()
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import mygeotab
import pandas as pd
import yaml
from dotenv import load_dotenv

# Extracts new GeoTab StatusData for the vehicles configured in the CACE project.
# The script uses a checkpoint for each vehicle so daily runs only collect new data.
# A small overlap is included between runs, and duplicate rows are removed before saving.
# Raw files are stored locally by vehicle and extraction date before the AWS upload step.


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"
VEHICLE_CONFIG = PROJECT_ROOT / "config" / "vehicles.yaml"
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"

MAPPING_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "vehicle_mapping"
    / "vehicle_mapping_private.csv"
)

DIAGNOSTICS_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "signal_inventory"
    / "all_diagnostics.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "checkpoints"
    / "ingestion_checkpoint.json"
)

RAW_BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "geotab"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

LOG_FILE = LOG_DIR / "extract_incremental_statusdata.log"

RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
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
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_geotab_credentials():
    if not ENV_FILE.exists():
        raise FileNotFoundError(f".env file not found: {ENV_FILE}")

    load_dotenv(ENV_FILE)

    username = os.getenv("GEOTAB_USERNAME")
    password = os.getenv("GEOTAB_PASSWORD")
    database = os.getenv("GEOTAB_DATABASE")

    missing = [
        name
        for name, value in {
            "GEOTAB_USERNAME": username,
            "GEOTAB_PASSWORD": password,
            "GEOTAB_DATABASE": database,
        }.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    return username, password, database


def connect_to_geotab():
    username, password, database = load_geotab_credentials()

    api = mygeotab.API(
        username=username,
        password=password,
        database=database,
    )

    api.authenticate()

    logger.info("Connected to GeoTab database: %s", database)

    return api


def load_target_vehicles():
    config = load_yaml(VEHICLE_CONFIG)
    vehicles = config.get("vehicles", [])

    if not vehicles:
        raise ValueError("No vehicles found in config/vehicles.yaml")

    return [str(vehicle).strip() for vehicle in vehicles]


def load_pipeline_settings():
    config = load_yaml(PIPELINE_CONFIG)

    timezone_name = config.get(
        "timezone",
        "America/New_York",
    )

    initial_start = config.get("initial_start_local")

    if not initial_start:
        raise ValueError(
            "initial_start_local is required in config/pipeline.yaml"
        )

    return {
        "timezone": ZoneInfo(timezone_name),
        "initial_start_local": datetime.fromisoformat(initial_start),
        "chunk_hours": int(config.get("chunk_hours", 6)),
        "overlap_minutes": int(config.get("overlap_minutes", 10)),
        "safe_delay_minutes": int(config.get("safe_delay_minutes", 5)),
        "sleep_seconds": float(config.get("sleep_seconds", 0.3)),
    }


def load_checkpoints():
    if not CHECKPOINT_FILE.exists():
        return {}

    if CHECKPOINT_FILE.stat().st_size == 0:
        return {}

    with CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoints(checkpoints):
    with CHECKPOINT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            checkpoints,
            file,
            indent=2,
            sort_keys=True,
        )


def load_reference_data():
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Vehicle mapping not found: {MAPPING_FILE}"
        )

    if not DIAGNOSTICS_FILE.exists():
        raise FileNotFoundError(
            f"Diagnostic inventory not found: {DIAGNOSTICS_FILE}"
        )

    mapping_df = pd.read_csv(MAPPING_FILE)
    diagnostics_df = pd.read_csv(DIAGNOSTICS_FILE)

    diagnostic_lookup = (
        diagnostics_df
        .drop_duplicates("diagnostic_id")
        .set_index("diagnostic_id")["diagnostic_name"]
        .to_dict()
    )

    unit_lookup = (
        diagnostics_df
        .drop_duplicates("diagnostic_id")
        .set_index("diagnostic_id")["unit_of_measure"]
        .to_dict()
    )

    return mapping_df, diagnostic_lookup, unit_lookup


def get_vehicle_mapping(mapping_df, vehicle):
    matched = mapping_df[
        (mapping_df["target_vehicle"].astype(str) == str(vehicle))
        & (mapping_df["matched"] == True)
    ]

    if matched.empty:
        raise ValueError(
            f"No valid GeoTab mapping found for vehicle {vehicle}"
        )

    if len(matched) > 1:
        raise ValueError(
            f"Multiple GeoTab devices found for vehicle {vehicle}"
        )

    row = matched.iloc[0]

    return row["device_id"], row["device_name"]


def to_geotab_datetime(dt):
    return dt.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )


def get_extraction_window(
    vehicle,
    checkpoints,
    settings,
):
    local_tz = settings["timezone"]

    now_utc = datetime.now(timezone.utc)
    extraction_end = now_utc - timedelta(
        minutes=settings["safe_delay_minutes"]
    )

    last_checkpoint = checkpoints.get(str(vehicle))

    if last_checkpoint:
        checkpoint_dt = datetime.fromisoformat(
            last_checkpoint.replace("Z", "+00:00")
        )

        extraction_start = checkpoint_dt - timedelta(
            minutes=settings["overlap_minutes"]
        )
    else:
        initial_start_local = settings["initial_start_local"]

        if initial_start_local.tzinfo is None:
            initial_start_local = initial_start_local.replace(
                tzinfo=local_tz
            )

        extraction_start = initial_start_local.astimezone(
            timezone.utc
        )

    return extraction_start, extraction_end


def parse_status_data(
    status_data,
    vehicle,
    device_id,
    device_name,
    diagnostic_lookup,
    unit_lookup,
):
    rows = []

    for item in status_data:
        diagnostic = item.get("diagnostic", {})

        diagnostic_id = (
            diagnostic.get("id")
            if isinstance(diagnostic, dict)
            else diagnostic
        )

        rows.append({
            "vehicle": vehicle,
            "device_id": device_id,
            "device_name": device_name,
            "datetime_raw": item.get("dateTime"),
            "diagnostic_id": diagnostic_id,
            "diagnostic_name": diagnostic_lookup.get(
                diagnostic_id
            ),
            "value": item.get("data"),
            "unit": unit_lookup.get(diagnostic_id),
        })

    return rows


def extract_vehicle_statusdata(
    api,
    vehicle,
    device_id,
    device_name,
    start_utc,
    end_utc,
    settings,
    diagnostic_lookup,
    unit_lookup,
):
    all_rows = []
    current_start = start_utc
    chunk_no = 1

    while current_start < end_utc:
        current_end = min(
            current_start
            + timedelta(hours=settings["chunk_hours"]),
            end_utc,
        )

        from_date = to_geotab_datetime(current_start)
        to_date = to_geotab_datetime(current_end)

        logger.info(
            "%s | chunk %d | %s -> %s",
            vehicle,
            chunk_no,
            from_date,
            to_date,
        )

        try:
            status_data = api.get(
                "StatusData",
                search={
                    "deviceSearch": {
                        "id": device_id,
                    },
                    "fromDate": from_date,
                    "toDate": to_date,
                },
            )

            logger.info(
                "%s | chunk %d | %d rows",
                vehicle,
                chunk_no,
                len(status_data),
            )

            all_rows.extend(
                parse_status_data(
                    status_data=status_data,
                    vehicle=vehicle,
                    device_id=device_id,
                    device_name=device_name,
                    diagnostic_lookup=diagnostic_lookup,
                    unit_lookup=unit_lookup,
                )
            )

            if len(status_data) >= 50000:
                logger.warning(
                    "%s | chunk %d returned 50,000 rows. "
                    "Consider reducing chunk_hours.",
                    vehicle,
                    chunk_no,
                )

        except Exception:
            logger.exception(
                "%s | chunk %d failed",
                vehicle,
                chunk_no,
            )
            raise

        current_start = current_end
        chunk_no += 1

        time.sleep(settings["sleep_seconds"])

    return pd.DataFrame(all_rows)


def clean_statusdata(raw_df, local_tz):
    if raw_df.empty:
        return raw_df

    raw_df = raw_df.copy()

    raw_df["datetime_utc"] = pd.to_datetime(
        raw_df["datetime_raw"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    failed_datetime_rows = raw_df["datetime_utc"].isna().sum()

    if failed_datetime_rows:
        logger.warning(
            "%d rows could not be parsed as timestamps",
            failed_datetime_rows,
        )

    raw_df = raw_df[
        raw_df["datetime_utc"].notna()
    ].copy()

    raw_df["datetime_local"] = (
        raw_df["datetime_utc"]
        .dt.tz_convert(local_tz)
    )

    raw_df["value"] = pd.to_numeric(
        raw_df["value"],
        errors="coerce",
    )

    before_dedup = len(raw_df)

    raw_df = raw_df.drop_duplicates(
        subset=[
            "vehicle",
            "datetime_utc",
            "diagnostic_id",
            "value",
        ]
    )

    removed = before_dedup - len(raw_df)

    if removed:
        logger.info(
            "Removed %d duplicate rows",
            removed,
        )

    return (
        raw_df
        .sort_values(
            ["datetime_utc", "diagnostic_name"]
        )
        .reset_index(drop=True)
    )


def keep_new_rows(raw_df, last_checkpoint):
    if raw_df.empty or not last_checkpoint:
        return raw_df

    checkpoint_dt = pd.Timestamp(last_checkpoint)

    if checkpoint_dt.tzinfo is None:
        checkpoint_dt = checkpoint_dt.tz_localize("UTC")

    return raw_df[
        raw_df["datetime_utc"] > checkpoint_dt
    ].copy()


def save_raw_data(raw_df, vehicle, local_tz):
    if raw_df.empty:
        return None

    run_time = datetime.now(timezone.utc)
    local_run_time = run_time.astimezone(local_tz)

    output_dir = (
        RAW_BASE_DIR
        / f"vehicle_{vehicle}"
        / f"year={local_run_time:%Y}"
        / f"month={local_run_time:%m}"
        / f"day={local_run_time:%d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"statusdata_{run_time:%Y%m%dT%H%M%SZ}.csv"
    )

    raw_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "%s | saved %d rows -> %s",
        vehicle,
        len(raw_df),
        output_file,
    )

    return output_file


def main():
    vehicles = load_target_vehicles()
    settings = load_pipeline_settings()
    checkpoints = load_checkpoints()

    mapping_df, diagnostic_lookup, unit_lookup = (
        load_reference_data()
    )

    api = connect_to_geotab()

    logger.info(
        "Starting incremental StatusData extraction for %d vehicles",
        len(vehicles),
    )

    for vehicle in vehicles:
        device_id, device_name = get_vehicle_mapping(
            mapping_df,
            vehicle,
        )

        start_utc, end_utc = get_extraction_window(
            vehicle,
            checkpoints,
            settings,
        )

        if start_utc >= end_utc:
            logger.info(
                "%s | no extraction window available",
                vehicle,
            )
            continue

        logger.info(
            "%s | extraction window: %s -> %s",
            vehicle,
            start_utc.isoformat(),
            end_utc.isoformat(),
        )

        raw_df = extract_vehicle_statusdata(
            api=api,
            vehicle=vehicle,
            device_id=device_id,
            device_name=device_name,
            start_utc=start_utc,
            end_utc=end_utc,
            settings=settings,
            diagnostic_lookup=diagnostic_lookup,
            unit_lookup=unit_lookup,
        )

        raw_df = clean_statusdata(
            raw_df,
            settings["timezone"],
        )

        last_checkpoint = checkpoints.get(str(vehicle))

        new_df = keep_new_rows(
            raw_df,
            last_checkpoint,
        )

        if new_df.empty:
            logger.info(
                "%s | no new rows found",
                vehicle,
            )
            continue

        save_raw_data(
            new_df,
            vehicle,
            settings["timezone"],
        )

        latest_timestamp = (
            new_df["datetime_utc"]
            .max()
            .isoformat()
        )

        checkpoints[str(vehicle)] = latest_timestamp

        # Save after each vehicle so a later failure does not lose completed progress.
        save_checkpoints(checkpoints)

        logger.info(
            "%s | checkpoint updated to %s",
            vehicle,
            latest_timestamp,
        )

    logger.info("Incremental extraction complete.")


if __name__ == "__main__":
    main()
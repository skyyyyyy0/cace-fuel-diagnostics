import logging
import os
from pathlib import Path

import mygeotab
import pandas as pd
import yaml
from dotenv import load_dotenv

# Validates the target vehicles in GeoTab and creates the vehicle-to-device mapping
# required to initialize the CACE data pipeline. This script does not collect StatusData.
# Re-run it when vehicles are added, removed, or their GeoTab devices change.


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CONFIG = PROJECT_ROOT / "config" / "vehicles.yaml"
ENV_FILE = PROJECT_ROOT / ".env"

OUTPUT_DIR = PROJECT_ROOT / "metadata" / "vehicle_mapping"
LOG_DIR = PROJECT_ROOT / "geotab_pipeline" / "logs"

OUTPUT_FILE = OUTPUT_DIR / "vehicle_mapping_private.csv"
LOG_FILE = LOG_DIR / "validate_vehicles.log"

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


def load_target_vehicles():
    if not VEHICLE_CONFIG.exists():
        raise FileNotFoundError(
            f"Vehicle config not found: {VEHICLE_CONFIG}"
        )

    with VEHICLE_CONFIG.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    vehicles = config.get("vehicles", [])

    if not vehicles:
        raise ValueError(
            "No vehicles found in config/vehicles.yaml"
        )

    vehicles = [str(vehicle).strip() for vehicle in vehicles]

    if len(vehicles) != len(set(vehicles)):
        raise ValueError(
            "Duplicate vehicles found in config/vehicles.yaml"
        )

    return vehicles


def load_geotab_credentials():
    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f".env file not found: {ENV_FILE}"
        )

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


def is_physical_device(device):
    serial_number = str(
        device.get("serialNumber", "")
    ).strip()

    return (
        bool(serial_number)
        and serial_number != "000-000-0000"
    )


def build_device_lookup(devices):
    lookup = {}

    for device in devices:
        if not is_physical_device(device):
            continue

        device_name = str(
            device.get("name", "")
        ).strip()

        if not device_name:
            continue

        lookup.setdefault(device_name, []).append(device)

    return lookup


def build_vehicle_mapping(target_vehicles, devices):
    device_lookup = build_device_lookup(devices)
    rows = []

    for vehicle in target_vehicles:
        matches = device_lookup.get(vehicle, [])

        if not matches:
            logger.warning("Vehicle not found: %s", vehicle)

            rows.append({
                "target_vehicle": vehicle,
                "matched": False,
                "match_count": 0,
                "device_id": None,
                "device_name": None,
                "serial_number": None,
                "vin": None,
            })

            continue

        if len(matches) > 1:
            logger.warning(
                "Multiple devices matched vehicle %s: %d",
                vehicle,
                len(matches),
            )

        for device in matches:
            rows.append({
                "target_vehicle": vehicle,
                "matched": True,
                "match_count": len(matches),
                "device_id": device.get("id"),
                "device_name": device.get("name"),
                "serial_number": device.get("serialNumber"),
                "vin": device.get("vehicleIdentificationNumber"),
            })

        logger.info(
            "Matched vehicle %s -> %d device(s)",
            vehicle,
            len(matches),
        )

    return pd.DataFrame(rows)


def validate_mapping(mapping_df):
    if mapping_df.empty:
        raise ValueError("Vehicle mapping is empty.")

    unmatched = mapping_df.loc[
        ~mapping_df["matched"],
        "target_vehicle",
    ].tolist()

    if unmatched:
        logger.warning(
            "Unmatched vehicles: %s",
            ", ".join(unmatched),
        )

    matched = mapping_df[mapping_df["matched"]].copy()

    duplicate_device_ids = matched.loc[
        matched["device_id"].duplicated(keep=False),
        "device_id",
    ].dropna().unique()

    if len(duplicate_device_ids) > 0:
        logger.warning(
            "Duplicate device IDs found: %s",
            duplicate_device_ids.tolist(),
        )


def save_mapping(mapping_df):
    mapping_df = mapping_df.sort_values(
        ["target_vehicle", "device_name"],
        na_position="last",
    ).reset_index(drop=True)

    mapping_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Saved vehicle mapping: %s", OUTPUT_FILE)


def main():
    target_vehicles = load_target_vehicles()

    logger.info(
        "Validating %d configured vehicles",
        len(target_vehicles),
    )

    api = connect_to_geotab()
    devices = api.get("Device")

    logger.info(
        "Retrieved %d devices from GeoTab",
        len(devices),
    )

    mapping_df = build_vehicle_mapping(
        target_vehicles,
        devices,
    )

    validate_mapping(mapping_df)
    save_mapping(mapping_df)

    matched_count = mapping_df.loc[
        mapping_df["matched"],
        "target_vehicle",
    ].nunique()

    logger.info(
        "Validation complete: %d/%d vehicles matched",
        matched_count,
        len(target_vehicles),
    )


if __name__ == "__main__":
    main()
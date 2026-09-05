import logging
import os
from pathlib import Path

import mygeotab
import pandas as pd
from dotenv import load_dotenv


# Pulls the GeoTab diagnostic catalog used by the CACE pipeline to identify
# available PIDs, units, sources, and controllers. This is reference metadata,
# not vehicle StatusData. Re-run this script when new diagnostics or feature
# flags become available.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "metadata"
    / "signal_inventory"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

OUTPUT_FILE = OUTPUT_DIR / "all_diagnostics.csv"
LOG_FILE = LOG_DIR / "extract_diagnostics.log"

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


def get_nested_value(value, key):
    if isinstance(value, dict):
        return value.get(key)

    return None


def parse_diagnostics(diagnostics):
    rows = []

    for diagnostic in diagnostics:
        source = diagnostic.get("source")
        controller = diagnostic.get("controller")

        rows.append({
            "diagnostic_id": diagnostic.get("id"),
            "diagnostic_name": diagnostic.get("name"),
            "source_id": (
                get_nested_value(source, "id")
                if isinstance(source, dict)
                else source
            ),
            "source_name": get_nested_value(source, "name"),
            "unit_of_measure": diagnostic.get("unitOfMeasure"),
            "diagnostic_code": diagnostic.get("code"),
            "controller_id": (
                get_nested_value(controller, "id")
                if isinstance(controller, dict)
                else controller
            ),
            "controller_name": get_nested_value(
                controller,
                "name",
            ),
            "raw_json": str(diagnostic),
        })

    diagnostics_df = pd.DataFrame(rows)

    if diagnostics_df.empty:
        return diagnostics_df

    diagnostics_df = (
        diagnostics_df
        .sort_values(
            by="diagnostic_name",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return diagnostics_df


def save_diagnostics(diagnostics_df):
    diagnostics_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved diagnostic inventory: %s",
        OUTPUT_FILE,
    )


def main():
    api = connect_to_geotab()

    logger.info("Extracting GeoTab diagnostic catalog...")

    diagnostics = api.get("Diagnostic")

    logger.info(
        "Retrieved %d diagnostics",
        len(diagnostics),
    )

    diagnostics_df = parse_diagnostics(diagnostics)

    if diagnostics_df.empty:
        logger.warning(
            "No diagnostics were returned from GeoTab."
        )
        return

    save_diagnostics(diagnostics_df)

    logger.info(
        "Diagnostic extraction complete: %d records",
        len(diagnostics_df),
    )

    print()
    print(diagnostics_df.head(20))
    print()
    print("Columns:")
    print(list(diagnostics_df.columns))


if __name__ == "__main__":
    main()
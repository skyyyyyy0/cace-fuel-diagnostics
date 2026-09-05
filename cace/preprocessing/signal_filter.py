# Filters raw GeoTab StatusData to the signals used by the CACE pipeline.
# Signal names are matched through config/signals.yaml so the filtering logic
# can be updated through configuration without changing this script. Raw data
# is never modified; filtered records are written separately for downstream use.

import logging
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_FILE = PROJECT_ROOT / "config" / "signals.yaml"

LOG_DIR = PROJECT_ROOT / "geotab_pipeline" / "logs"
LOG_FILE = LOG_DIR / "signal_filter.log"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
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


def normalize_name(value):
    """Normalize diagnostic names before matching them to the signal registry."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().lower().split())


def load_signal_registry():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Signal configuration not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(
            f"Signal configuration is empty: {CONFIG_FILE}"
        )

    registry = {}

    for signal_group, signals in config.items():
        if not isinstance(signals, dict):
            continue

        for signal_key, signal_info in signals.items():
            display_name = signal_info.get("display_name")
            aliases = signal_info.get("aliases", [])

            if not display_name:
                logger.warning(
                    "%s.%s has no display_name and will be skipped",
                    signal_group,
                    signal_key,
                )
                continue

            names = [display_name, *aliases]

            for name in names:
                normalized = normalize_name(name)

                if not normalized:
                    continue

                existing = registry.get(normalized)

                if existing and existing["signal_key"] != signal_key:
                    raise ValueError(
                        f"Signal alias '{name}' is assigned to more than "
                        f"one signal key."
                    )

                registry[normalized] = {
                    "signal_group": signal_group,
                    "signal_key": signal_key,
                    "signal_name": display_name,
                }

    if not registry:
        raise ValueError(
            "No valid signals were loaded from signals.yaml"
        )

    return registry


def find_raw_files():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}"
        )

    return sorted(RAW_DIR.rglob("*.csv"))


def read_raw_file(path):
    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read raw file: {path}"
        ) from exc


def find_column(columns, candidates):
    normalized_columns = {
        normalize_name(column).replace(" ", "_"): column
        for column in columns
    }

    for candidate in candidates:
        normalized_candidate = (
            normalize_name(candidate)
            .replace(" ", "_")
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def identify_columns(df):
    diagnostic_column = find_column(
        df.columns,
        [
            "diagnostic_name",
            "diagnostic",
            "signal_name",
        ],
    )

    datetime_column = find_column(
        df.columns,
        [
            "datetime_utc",
            "datetime",
            "date_time",
            "timestamp",
        ],
    )

    value_column = find_column(
        df.columns,
        [
            "value",
            "data",
        ],
    )

    missing = []

    if diagnostic_column is None:
        missing.append("diagnostic name")

    if datetime_column is None:
        missing.append("datetime")

    if value_column is None:
        missing.append("value")

    if missing:
        raise ValueError(
            "Required column(s) not found: "
            + ", ".join(missing)
        )

    return {
        "diagnostic": diagnostic_column,
        "datetime": datetime_column,
        "value": value_column,
    }


def filter_signals(df, registry, source_file):
    if df.empty:
        return pd.DataFrame()

    columns = identify_columns(df)

    working_df = df.copy()

    working_df["_normalized_diagnostic"] = (
        working_df[columns["diagnostic"]]
        .map(normalize_name)
    )

    matched = working_df[
        working_df["_normalized_diagnostic"].isin(registry)
    ].copy()

    if matched.empty:
        return pd.DataFrame()

    matched["signal_group"] = matched[
        "_normalized_diagnostic"
    ].map(
        lambda name: registry[name]["signal_group"]
    )

    matched["signal_key"] = matched[
        "_normalized_diagnostic"
    ].map(
        lambda name: registry[name]["signal_key"]
    )

    matched["signal_name"] = matched[
        "_normalized_diagnostic"
    ].map(
        lambda name: registry[name]["signal_name"]
    )

    matched["datetime_utc"] = pd.to_datetime(
        matched[columns["datetime"]],
        utc=True,
        errors="coerce",
    )

    invalid_timestamp_count = matched["datetime_utc"].isna().sum()

    if invalid_timestamp_count > 0:
        logger.warning(
            "%s | dropping %d rows with invalid timestamps",
            source_file.name,
            invalid_timestamp_count,
        )

    matched = matched[
        matched["datetime_utc"].notna()
    ].copy()

    matched["value"] = pd.to_numeric(
        matched[columns["value"]],
        errors="coerce",
    )

    matched["source_file"] = source_file.name

    vehicle_column = find_column(
        matched.columns,
        [
            "vehicle",
            "vehicle_id",
            "device",
            "device_id",
        ],
    )

    if vehicle_column:
        matched["vehicle"] = (
            matched[vehicle_column]
            .astype(str)
            .str.strip()
        )
    else:
        matched["vehicle"] = None

    output_columns = [
        "vehicle",
        "datetime_utc",
        "signal_group",
        "signal_key",
        "signal_name",
        "value",
        "source_file",
    ]

    return matched[output_columns]


def remove_duplicates(df):
    if df.empty:
        return df

    duplicate_columns = [
        "vehicle",
        "datetime_utc",
        "signal_key",
        "value",
    ]

    return (
        df.drop_duplicates(
            subset=duplicate_columns,
            keep="last",
        )
        .sort_values(
            ["vehicle", "datetime_utc", "signal_key"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_output_path(raw_file):
    relative_path = raw_file.relative_to(RAW_DIR)

    output_directory = (
        PROCESSED_DIR
        / "cace_signals"
        / relative_path.parent
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory / raw_file.name


def process_file(raw_file, registry):
    logger.info(
        "Processing raw file: %s",
        raw_file.relative_to(PROJECT_ROOT),
    )

    raw_df = read_raw_file(raw_file)

    if raw_df.empty:
        logger.warning(
            "Skipping empty raw file: %s",
            raw_file.name,
        )
        return 0

    filtered_df = filter_signals(
        raw_df,
        registry,
        raw_file,
    )

    if filtered_df.empty:
        logger.warning(
            "No CACE signals found: %s",
            raw_file.name,
        )
        return 0

    filtered_df = remove_duplicates(
        filtered_df
    )

    output_file = build_output_path(
        raw_file
    )

    filtered_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved %d CACE records: %s",
        len(filtered_df),
        output_file.relative_to(PROJECT_ROOT),
    )

    return len(filtered_df)


def main():
    logger.info("Starting CACE signal filtering")

    registry = load_signal_registry()

    logger.info(
        "Loaded %d configured signal names and aliases",
        len(registry),
    )

    raw_files = find_raw_files()

    if not raw_files:
        logger.warning(
            "No raw CSV files found under %s",
            RAW_DIR,
        )
        return

    logger.info(
        "Found %d raw file(s)",
        len(raw_files),
    )

    processed_files = 0
    total_records = 0

    for raw_file in raw_files:
        try:
            record_count = process_file(
                raw_file,
                registry,
            )

            if record_count > 0:
                processed_files += 1
                total_records += record_count

        except Exception:
            logger.exception(
                "Failed to process: %s",
                raw_file,
            )

    logger.info(
        "Signal filtering complete | files=%d | records=%d",
        processed_files,
        total_records,
    )


if __name__ == "__main__":
    main()
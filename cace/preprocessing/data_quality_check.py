# Runs repeatable data quality checks on the filtered signals used by CACE.
# The checks cover timestamps, missing values, duplicates, configured signal
# ranges, and cumulative fuel counter behavior. Source records are never removed;
# potential issues are flagged and written to separate QC reports for review.

import logging
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIGNAL_CONFIG = PROJECT_ROOT / "config" / "signals.yaml"
QUALITY_CONFIG = PROJECT_ROOT / "config" / "quality_rules.yaml"

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cace_signals"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "data_quality_summary_private.csv"
)

ISSUE_FILE = (
    OUTPUT_DIR
    / "data_quality_issues_private.csv"
)

FUEL_EVENT_FILE = (
    OUTPUT_DIR
    / "fuel_counter_events_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "data_quality_check.log"
)

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


def load_signal_registry():
    config = load_yaml(SIGNAL_CONFIG)
    registry = {}

    for signal_group, signals in config.items():
        if not isinstance(signals, dict):
            continue

        for signal_key, settings in signals.items():
            registry[signal_key] = {
                "signal_group": signal_group,
                "signal_name": settings.get(
                    "display_name",
                    signal_key,
                ),
            }

    return registry


def load_quality_rules():
    config = load_yaml(QUALITY_CONFIG)
    rules = config.get("rules", {})

    if not rules:
        raise ValueError(
            "No quality rules found in quality_rules.yaml"
        )

    return rules


def find_input_files():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Filtered data directory not found: {INPUT_DIR}"
        )

    files = sorted(
        INPUT_DIR.rglob("statusdata_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No filtered StatusData files found under {INPUT_DIR}"
        )

    logger.info(
        "Found %d filtered data file(s)",
        len(files),
    )

    return files


def load_filtered_data():
    files = find_input_files()
    frames = []

    required_columns = [
        "vehicle",
        "datetime_utc",
        "signal_group",
        "signal_key",
        "signal_name",
        "value",
    ]

    for path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=required_columns,
                low_memory=False,
            )

            df["source_file"] = path.name
            frames.append(df)

        except Exception as exc:
            logger.warning(
                "Failed to read %s | %s",
                path.name,
                exc,
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def prepare_data(df):
    df = df.copy()

    df["vehicle"] = (
        df["vehicle"]
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

    return df


def build_basic_issues(df):
    issues = []

    invalid_timestamp_df = df[
        df["datetime_utc"].isna()
    ]

    for _, row in invalid_timestamp_df.iterrows():
        issues.append({
            "vehicle": row["vehicle"],
            "datetime_utc": None,
            "signal_key": row["signal_key"],
            "signal_name": row["signal_name"],
            "value": row["value"],
            "issue_type": "invalid_timestamp",
            "rule_min": None,
            "rule_max": None,
            "source_file": row["source_file"],
        })

    missing_value_df = df[
        df["value"].isna()
    ]

    for _, row in missing_value_df.iterrows():
        issues.append({
            "vehicle": row["vehicle"],
            "datetime_utc": row["datetime_utc"],
            "signal_key": row["signal_key"],
            "signal_name": row["signal_name"],
            "value": None,
            "issue_type": "missing_value",
            "rule_min": None,
            "rule_max": None,
            "source_file": row["source_file"],
        })

    duplicate_mask = df.duplicated(
        subset=[
            "vehicle",
            "datetime_utc",
            "signal_key",
            "value",
        ],
        keep=False,
    )

    duplicate_df = df[
        duplicate_mask
    ]

    for _, row in duplicate_df.iterrows():
        issues.append({
            "vehicle": row["vehicle"],
            "datetime_utc": row["datetime_utc"],
            "signal_key": row["signal_key"],
            "signal_name": row["signal_name"],
            "value": row["value"],
            "issue_type": "duplicate_record",
            "rule_min": None,
            "rule_max": None,
            "source_file": row["source_file"],
        })

    return issues


def check_range_rules(df, quality_rules):
    issues = []

    for signal_key, rule in quality_rules.items():
        if rule.get("enabled", True) is False:
            continue

        if rule.get("type") != "range":
            continue

        signal_df = df[
            df["signal_key"] == signal_key
        ].copy()

        if signal_df.empty:
            continue

        minimum = rule.get("min")
        maximum = rule.get("max")

        mask = pd.Series(
            False,
            index=signal_df.index,
        )

        if minimum is not None:
            mask |= signal_df["value"] < minimum

        if maximum is not None:
            mask |= signal_df["value"] > maximum

        outliers = signal_df[
            mask
        ]

        for _, row in outliers.iterrows():
            issues.append({
                "vehicle": row["vehicle"],
                "datetime_utc": row["datetime_utc"],
                "signal_key": row["signal_key"],
                "signal_name": row["signal_name"],
                "value": row["value"],
                "issue_type": "range_outlier",
                "rule_min": minimum,
                "rule_max": maximum,
                "source_file": row["source_file"],
            })

    return issues


def check_allowed_values(df, quality_rules):
    issues = []

    for signal_key, rule in quality_rules.items():
        if rule.get("enabled", True) is False:
            continue

        if rule.get("type") != "allowed_values":
            continue

        allowed_values = rule.get(
            "values",
            [],
        )

        signal_df = df[
            df["signal_key"] == signal_key
        ]

        invalid_df = signal_df[
            ~signal_df["value"].isin(
                allowed_values
            )
        ]

        for _, row in invalid_df.iterrows():
            issues.append({
                "vehicle": row["vehicle"],
                "datetime_utc": row["datetime_utc"],
                "signal_key": row["signal_key"],
                "signal_name": row["signal_name"],
                "value": row["value"],
                "issue_type": "invalid_category",
                "rule_min": None,
                "rule_max": None,
                "source_file": row["source_file"],
            })

    return issues


def check_total_fuel_counter(df):
    fuel_df = df[
        df["signal_key"] == "total_fuel_used"
    ].copy()

    if fuel_df.empty:
        return pd.DataFrame()

    fuel_df = (
        fuel_df
        .dropna(
            subset=[
                "datetime_utc",
                "value",
            ]
        )
        .drop_duplicates(
            subset=[
                "vehicle",
                "datetime_utc",
                "value",
            ]
        )
        .sort_values(
            [
                "vehicle",
                "datetime_utc",
            ]
        )
    )

    fuel_df["previous_value"] = (
        fuel_df
        .groupby("vehicle")["value"]
        .shift(1)
    )

    fuel_df["fuel_delta"] = (
        fuel_df["value"]
        - fuel_df["previous_value"]
    )

    events = fuel_df[
        fuel_df["fuel_delta"] < 0
    ].copy()

    if events.empty:
        return pd.DataFrame(
            columns=[
                "vehicle",
                "datetime_utc",
                "previous_value",
                "value",
                "fuel_delta",
                "event_type",
                "source_file",
            ]
        )

    events["event_type"] = (
        "fuel_counter_decrease"
    )

    return events[
        [
            "vehicle",
            "datetime_utc",
            "previous_value",
            "value",
            "fuel_delta",
            "event_type",
            "source_file",
        ]
    ]


def build_signal_summary(
    df,
    signal_registry,
    issue_df,
):
    rows = []

    vehicles = sorted(
        df["vehicle"]
        .dropna()
        .astype(str)
        .unique()
    )

    for vehicle in vehicles:
        vehicle_df = df[
            df["vehicle"] == vehicle
        ]

        for signal_key, signal_info in signal_registry.items():
            signal_df = vehicle_df[
                vehicle_df["signal_key"]
                == signal_key
            ]

            signal_issues = issue_df[
                (issue_df["vehicle"] == vehicle)
                & (
                    issue_df["signal_key"]
                    == signal_key
                )
            ]

            range_outliers = signal_issues[
                signal_issues["issue_type"]
                == "range_outlier"
            ]

            rows.append({
                "vehicle": vehicle,
                "signal_group": (
                    signal_info["signal_group"]
                ),
                "signal_key": signal_key,
                "signal_name": (
                    signal_info["signal_name"]
                ),
                "available": not signal_df.empty,
                "record_count": len(signal_df),
                "missing_value_count": (
                    signal_df["value"]
                    .isna()
                    .sum()
                ),
                "outlier_count": len(
                    range_outliers
                ),
                "outlier_pct": (
                    round(
                        len(range_outliers)
                        / len(signal_df)
                        * 100,
                        4,
                    )
                    if len(signal_df)
                    else 0
                ),
                "first_datetime_utc": (
                    signal_df["datetime_utc"].min()
                    if not signal_df.empty
                    else pd.NaT
                ),
                "last_datetime_utc": (
                    signal_df["datetime_utc"].max()
                    if not signal_df.empty
                    else pd.NaT
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
        path.relative_to(PROJECT_ROOT),
    )


def main():
    logger.info(
        "Starting CACE data quality check"
    )

    signal_registry = load_signal_registry()
    quality_rules = load_quality_rules()

    df = load_filtered_data()

    if df.empty:
        logger.warning(
            "No filtered CACE data found"
        )
        return

    df = prepare_data(df)

    logger.info(
        "Loaded %d filtered CACE records",
        len(df),
    )

    issues = build_basic_issues(df)

    issues.extend(
        check_range_rules(
            df,
            quality_rules,
        )
    )

    issues.extend(
        check_allowed_values(
            df,
            quality_rules,
        )
    )

    issue_columns = [
        "vehicle",
        "datetime_utc",
        "signal_key",
        "signal_name",
        "value",
        "issue_type",
        "rule_min",
        "rule_max",
        "source_file",
    ]

    issue_df = pd.DataFrame(
        issues,
        columns=issue_columns,
    )

    fuel_event_df = (
        check_total_fuel_counter(df)
    )

    summary_df = build_signal_summary(
        df,
        signal_registry,
        issue_df,
    )

    save_report(
        summary_df,
        SUMMARY_FILE,
    )

    save_report(
        issue_df,
        ISSUE_FILE,
    )

    save_report(
        fuel_event_df,
        FUEL_EVENT_FILE,
    )

    logger.info(
        "Data quality check complete | "
        "vehicles=%d | records=%d | issues=%d | fuel_events=%d",
        df["vehicle"].nunique(),
        len(df),
        len(issue_df),
        len(fuel_event_df),
    )


if __name__ == "__main__":
    main()
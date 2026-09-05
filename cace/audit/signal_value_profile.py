# Profiles the value distribution of CACE signals before outlier rules are defined.
# The script summarizes each signal by vehicle and across the available fleet using
# percentiles and basic statistics. It does not remove or modify any observations.
# The output is used to review realistic signal ranges and define QC thresholds.

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

VEHICLE_PROFILE_FILE = (
    OUTPUT_DIR
    / "signal_value_profile_by_vehicle_private.csv"
)

FLEET_PROFILE_FILE = (
    OUTPUT_DIR
    / "signal_value_profile_fleet_private.csv"
)

LOG_FILE = (
    LOG_DIR
    / "signal_value_profile.log"
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


REQUIRED_COLUMNS = {
    "vehicle",
    "datetime_utc",
    "signal_group",
    "signal_key",
    "signal_name",
    "value",
}


def find_input_files():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Filtered signal directory not found: {INPUT_DIR}"
        )

    files = sorted(
        INPUT_DIR.rglob("statusdata_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No filtered StatusData files found under {INPUT_DIR}"
        )

    return files


def load_signal_data():
    files = find_input_files()

    frames = []

    for path in files:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception as exc:
            logger.warning(
                "Could not read %s | %s",
                path.name,
                exc,
            )
            continue

        missing_columns = (
            REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing_columns:
            logger.warning(
                "Skipping %s | missing columns: %s",
                path.name,
                sorted(missing_columns),
            )
            continue

        df = df[
            list(REQUIRED_COLUMNS)
        ].copy()

        df["source_file"] = path.name

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No valid filtered signal files were loaded."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    logger.info(
        "Loaded %d records from %d file(s)",
        len(combined),
        len(frames),
    )

    return combined


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

    invalid_timestamp_count = (
        df["datetime_utc"]
        .isna()
        .sum()
    )

    invalid_value_count = (
        df["value"]
        .isna()
        .sum()
    )

    if invalid_timestamp_count:
        logger.warning(
            "Found %d record(s) with invalid timestamps",
            invalid_timestamp_count,
        )

    if invalid_value_count:
        logger.warning(
            "Found %d record(s) with non-numeric or missing values",
            invalid_value_count,
        )

    return df


def summarize_group(group):
    values = (
        group["value"]
        .dropna()
    )

    if values.empty:
        return pd.Series({
            "record_count": 0,
            "missing_value_count": len(group),
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
            "std": None,
        })

    return pd.Series({
        "record_count": len(values),
        "missing_value_count": (
            group["value"]
            .isna()
            .sum()
        ),
        "min": values.min(),
        "p01": values.quantile(0.01),
        "p05": values.quantile(0.05),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "mean": values.mean(),
        "p75": values.quantile(0.75),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
        "max": values.max(),
        "std": values.std(),
    })


def build_vehicle_profile(df):
    profile = (
        df.groupby(
            [
                "vehicle",
                "signal_group",
                "signal_key",
                "signal_name",
            ],
            dropna=False,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )

    return profile.sort_values(
        [
            "vehicle",
            "signal_group",
            "signal_key",
        ]
    ).reset_index(drop=True)


def build_fleet_profile(df):
    profile = (
        df.groupby(
            [
                "signal_group",
                "signal_key",
                "signal_name",
            ],
            dropna=False,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )

    vehicle_counts = (
        df.groupby(
            "signal_key"
        )["vehicle"]
        .nunique()
        .rename("vehicle_count")
        .reset_index()
    )

    profile = profile.merge(
        vehicle_counts,
        on="signal_key",
        how="left",
    )

    columns = [
        "signal_group",
        "signal_key",
        "signal_name",
        "vehicle_count",
        "record_count",
        "missing_value_count",
        "min",
        "p01",
        "p05",
        "p25",
        "median",
        "mean",
        "p75",
        "p95",
        "p99",
        "max",
        "std",
    ]

    return (
        profile[columns]
        .sort_values(
            [
                "signal_group",
                "signal_key",
            ]
        )
        .reset_index(drop=True)
    )


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


def print_summary(profile):
    if profile.empty:
        return

    columns = [
        "signal_key",
        "vehicle_count",
        "record_count",
        "min",
        "median",
        "p95",
        "p99",
        "max",
    ]

    print()
    print("CACE Signal Value Profile")
    print(
        profile[columns]
        .to_string(index=False)
    )


def main():
    logger.info(
        "Starting CACE signal value profiling"
    )

    df = load_signal_data()
    df = prepare_data(df)

    vehicle_profile = build_vehicle_profile(
        df
    )

    fleet_profile = build_fleet_profile(
        df
    )

    save_report(
        vehicle_profile,
        VEHICLE_PROFILE_FILE,
    )

    save_report(
        fleet_profile,
        FLEET_PROFILE_FILE,
    )

    print_summary(
        fleet_profile
    )

    logger.info(
        "Signal value profiling complete | "
        "vehicles=%d | signals=%d | records=%d",
        df["vehicle"].nunique(),
        df["signal_key"].nunique(),
        len(df),
    )


if __name__ == "__main__":
    main()
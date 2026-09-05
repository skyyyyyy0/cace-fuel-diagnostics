# Compares signal availability across the current CACE vehicle fleet.
# The report shows which features are available on all vehicles, which are only
# available on part of the fleet, and which are missing completely. This helps
# define the common feature set that can be used in the initial CACE model.

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_audit"
)

LOG_DIR = (
    PROJECT_ROOT
    / "geotab_pipeline"
    / "logs"
)

CORE_FILE = AUDIT_DIR / "core_pid_availability.csv"
ADDITIONAL_FILE = AUDIT_DIR / "additional_pid_availability.csv"
DPF_FILE = AUDIT_DIR / "dpf_pid_availability.csv"

OUTPUT_FILE = (
    AUDIT_DIR
    / "vehicle_signal_coverage.csv"
)

LOG_FILE = (
    LOG_DIR
    / "vehicle_coverage.log"
)

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


def load_availability_report(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Availability report not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "vehicle",
        "signal_group",
        "signal_key",
        "signal_name",
        "available",
        "record_count",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def load_all_availability_reports():
    reports = [
        load_availability_report(CORE_FILE),
        load_availability_report(ADDITIONAL_FILE),
        load_availability_report(DPF_FILE),
    ]

    return pd.concat(
        reports,
        ignore_index=True,
    )


def normalize_available_column(df):
    df = df.copy()

    # CSV files may store boolean values as text, so normalize them before
    # calculating fleet coverage.
    if df["available"].dtype == object:
        df["available"] = (
            df["available"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
            })
        )

    if df["available"].isna().any():
        raise ValueError(
            "The available column contains values "
            "that could not be interpreted as True or False."
        )

    return df


def build_coverage_report(availability_df):
    vehicles = sorted(
        availability_df["vehicle"]
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []

    grouped = availability_df.groupby(
        [
            "signal_group",
            "signal_key",
            "signal_name",
        ],
        dropna=False,
    )

    for (
        signal_group,
        signal_key,
        signal_name,
    ), signal_df in grouped:

        row = {
            "signal_group": signal_group,
            "signal_key": signal_key,
            "signal_name": signal_name,
        }

        vehicles_available = 0
        total_records = 0

        for vehicle in vehicles:
            vehicle_df = signal_df[
                signal_df["vehicle"]
                .astype(str)
                == vehicle
            ]

            if vehicle_df.empty:
                available = False
                record_count = 0
            else:
                available = bool(
                    vehicle_df.iloc[0]["available"]
                )

                record_count = int(
                    vehicle_df.iloc[0]["record_count"]
                )

            row[f"vehicle_{vehicle}"] = available
            row[f"records_{vehicle}"] = record_count

            if available:
                vehicles_available += 1

            total_records += record_count

        coverage_pct = (
            vehicles_available
            / len(vehicles)
            * 100
        )

        if vehicles_available == len(vehicles):
            coverage_status = "Common"

        elif vehicles_available == 0:
            coverage_status = "Unavailable"

        else:
            coverage_status = "Partial"

        row["vehicles_available"] = vehicles_available
        row["total_vehicles"] = len(vehicles)
        row["coverage_pct"] = round(
            coverage_pct,
            1,
        )
        row["coverage_status"] = coverage_status
        row["total_record_count"] = total_records

        rows.append(row)

    return pd.DataFrame(rows)


def save_report(coverage_df):
    coverage_df = coverage_df.sort_values(
        [
            "signal_group",
            "coverage_pct",
            "signal_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    coverage_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved vehicle coverage report: %s",
        OUTPUT_FILE,
    )

    return coverage_df


def print_summary(coverage_df):
    if coverage_df.empty:
        return

    summary_columns = [
        "signal_group",
        "signal_name",
        "vehicles_available",
        "total_vehicles",
        "coverage_pct",
        "coverage_status",
    ]

    print()
    print("Vehicle Signal Coverage")
    print(
        coverage_df[
            summary_columns
        ].to_string(index=False)
    )


def main():
    logger.info(
        "Starting vehicle signal coverage analysis"
    )

    availability_df = (
        load_all_availability_reports()
    )

    availability_df = (
        normalize_available_column(
            availability_df
        )
    )

    coverage_df = build_coverage_report(
        availability_df
    )

    coverage_df = save_report(
        coverage_df
    )

    print_summary(
        coverage_df
    )

    logger.info(
        "Vehicle coverage analysis complete: "
        "%d signals reviewed",
        len(coverage_df),
    )


if __name__ == "__main__":
    main()
# Calculates anonymized vehicle-level fuel deviation from CACE V1 LOVO predictions.
# The results use only out-of-sample vehicle predictions and contain aggregated
# window-level values without timestamps or raw telemetry.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shap_global_importance import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    find_artifact,
)


LOVO_PREDICTIONS_FILE = "CACE_v1.0_lovo_predictions_private.csv"

REQUIRED_COLUMNS = [
    "vehicle",
    "actual_fuel_used",
    "physics_expected_fuel",
    "cace_expected_fuel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing the private LOVO predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated interpretation outputs.",
    )
    return parser.parse_args()


def load_lovo_predictions(
    artifacts_dir: Path,
) -> pd.DataFrame:
    prediction_path = find_artifact(
        artifacts_dir,
        LOVO_PREDICTIONS_FILE,
    )
    return pd.read_csv(prediction_path)


def validate_lovo_predictions(
    predictions: pd.DataFrame,
) -> None:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"LOVO predictions are missing columns: {missing_columns}"
        )

    if predictions.empty:
        raise ValueError(
            "The LOVO prediction dataset is empty."
        )

    if predictions["vehicle"].isna().any():
        raise ValueError(
            "LOVO predictions contain missing vehicle identifiers."
        )

    vehicle_labels = predictions["vehicle"].astype("string")
    valid_vehicle_labels = vehicle_labels.str.fullmatch(
        r"VEHICLE_\d{2}"
    )

    if not valid_vehicle_labels.all():
        invalid_labels = sorted(
            vehicle_labels[~valid_vehicle_labels]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Vehicle identifiers must already be anonymized using "
            f"the VEHICLE_XX format. Invalid values: {invalid_labels}"
        )

    fuel_columns = [
        "actual_fuel_used",
        "physics_expected_fuel",
        "cace_expected_fuel",
    ]

    if predictions[fuel_columns].isna().any().any():
        raise ValueError(
            "LOVO predictions contain missing fuel values."
        )

    fuel_values = predictions[fuel_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(fuel_values).all():
        raise ValueError(
            "LOVO predictions contain non-finite fuel values."
        )

    if "anchor_time_utc" in predictions.columns:
        duplicate_rows = predictions.duplicated(
            subset=["vehicle", "anchor_time_utc"]
        )

        if duplicate_rows.any():
            raise ValueError(
                "Duplicate vehicle and anchor-time rows were found."
            )


def calculate_deviation_percent(
    actual_fuel: float,
    expected_fuel: float,
) -> float:
    if expected_fuel <= 0:
        raise ValueError(
            "Expected fuel must be greater than zero "
            "for percentage deviation."
        )

    return (
        (actual_fuel - expected_fuel)
        / expected_fuel
        * 100
    )


def build_vehicle_deviation_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.copy()

    frame["vehicle"] = frame["vehicle"].astype("string")
    frame["physics_window_deviation_l"] = (
        frame["actual_fuel_used"]
        - frame["physics_expected_fuel"]
    )
    frame["cace_window_deviation_l"] = (
        frame["actual_fuel_used"]
        - frame["cace_expected_fuel"]
    )

    rows = []

    for vehicle, group in frame.groupby(
        "vehicle",
        sort=True,
    ):
        actual_sum = float(
            group["actual_fuel_used"].sum()
        )
        physics_expected_sum = float(
            group["physics_expected_fuel"].sum()
        )
        cace_expected_sum = float(
            group["cace_expected_fuel"].sum()
        )

        physics_deviation = (
            actual_sum - physics_expected_sum
        )
        cace_deviation = (
            actual_sum - cace_expected_sum
        )

        rows.append(
            {
                "vehicle": str(vehicle),
                "lovo_observations": len(group),
                "aggregate_actual_window_fuel_l": actual_sum,
                "aggregate_physics_expected_window_fuel_l": (
                    physics_expected_sum
                ),
                "aggregate_cace_expected_window_fuel_l": (
                    cace_expected_sum
                ),
                "physics_window_deviation_l": physics_deviation,
                "cace_window_deviation_l": cace_deviation,
                "physics_deviation_percent": (
                    calculate_deviation_percent(
                        actual_sum,
                        physics_expected_sum,
                    )
                ),
                "cace_deviation_percent": (
                    calculate_deviation_percent(
                        actual_sum,
                        cace_expected_sum,
                    )
                ),
                "positive_cace_deviation_window_rate_percent": (
                    float(
                        (
                            group["cace_window_deviation_l"] > 0
                        ).mean()
                        * 100
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("vehicle")
        .reset_index(drop=True)
    )


def save_vehicle_deviation_summary(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        table_dir / "vehicle_fuel_deviation_public.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    return output_path


def main() -> None:
    args = parse_args()

    predictions = load_lovo_predictions(
        args.artifacts_dir.resolve()
    )
    validate_lovo_predictions(predictions)

    summary = build_vehicle_deviation_summary(
        predictions
    )

    output_path = save_vehicle_deviation_summary(
        summary,
        args.output_dir.resolve(),
    )

    print(
        summary[
            [
                "vehicle",
                "lovo_observations",
                "aggregate_actual_window_fuel_l",
                "aggregate_cace_expected_window_fuel_l",
                "cace_window_deviation_l",
                "cace_deviation_percent",
                "positive_cace_deviation_window_rate_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved table: {output_path}")
    print(
        "Note: Aggregated window fuel is a diagnostic measure, "
        "not total fuel waste or realized savings."
    )


if __name__ == "__main__":
    main()
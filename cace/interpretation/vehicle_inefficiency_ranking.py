# Ranks vehicles for review using out-of-sample CACE fuel deviation.
# The ranking uses aggregate LOVO fuel deviation percent as the primary metric.
# Positive deviation is treated as a diagnostic review signal, not confirmed waste.


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_global_importance import DEFAULT_OUTPUT_DIR


DEFAULT_INPUT_FILE = (
    DEFAULT_OUTPUT_DIR
    / "tables"
    / "vehicle_fuel_deviation_public.csv"
)

REQUIRED_COLUMNS = [
    "vehicle",
    "lovo_observations",
    "aggregate_actual_window_fuel_l",
    "aggregate_cace_expected_window_fuel_l",
    "cace_window_deviation_l",
    "cace_deviation_percent",
    "positive_cace_deviation_window_rate_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Vehicle-level fuel deviation table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for interpretation outputs.",
    )
    return parser.parse_args()


def load_vehicle_deviation(
    input_file: Path,
) -> pd.DataFrame:
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Vehicle deviation table not found: {input_file}"
        )

    return pd.read_csv(input_file)


def validate_vehicle_deviation(
    summary: pd.DataFrame,
) -> None:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in summary.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Vehicle deviation table is missing columns: {missing_columns}"
        )

    if summary.empty:
        raise ValueError(
            "The vehicle deviation table is empty."
        )

    if summary["vehicle"].isna().any():
        raise ValueError(
            "The vehicle deviation table contains missing vehicle labels."
        )

    vehicle_labels = summary["vehicle"].astype("string")
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
            "Vehicle labels must use the VEHICLE_XX format. "
            f"Invalid values: {invalid_labels}"
        )

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != "vehicle"
    ]

    if summary[numeric_columns].isna().any().any():
        raise ValueError(
            "The vehicle deviation table contains missing numeric values."
        )

    values = summary[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "The vehicle deviation table contains non-finite values."
        )

    if summary["vehicle"].duplicated().any():
        raise ValueError(
            "The vehicle deviation table contains duplicate vehicles."
        )


def assign_review_status(
    deviation_percent: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                deviation_percent > 0,
                deviation_percent < 0,
            ],
            [
                "Actual fuel above CACE expected",
                "Actual fuel below CACE expected",
            ],
            default="Actual fuel matches CACE expected",
        ),
        index=deviation_percent.index,
    )


def build_vehicle_ranking(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ranking = (
        summary.copy()
        .sort_values(
            [
                "cace_deviation_percent",
                "positive_cace_deviation_window_rate_percent",
            ],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    ranking["review_status"] = assign_review_status(
        ranking["cace_deviation_percent"]
    )

    ranking["review_priority_rank"] = pd.Series(
        pd.NA,
        index=ranking.index,
        dtype="Int64",
    )

    positive_positions = ranking.index[
        ranking["cace_deviation_percent"] > 0
    ]

    ranking.loc[
        positive_positions,
        "review_priority_rank",
    ] = range(1, len(positive_positions) + 1)

    first_columns = [
        "review_priority_rank",
        "vehicle",
        "review_status",
        "cace_deviation_percent",
        "positive_cace_deviation_window_rate_percent",
        "cace_window_deviation_l",
        "lovo_observations",
    ]

    remaining_columns = [
        column
        for column in ranking.columns
        if column not in first_columns
    ]

    return ranking[
        first_columns + remaining_columns
    ]


def save_vehicle_ranking(
    ranking: pd.DataFrame,
    output_dir: Path,
) -> Path:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        table_dir / "vehicle_inefficiency_ranking_public.csv"
    )

    ranking.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    return output_path


def save_ranking_plot(
    ranking: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        figure_dir / "vehicle_inefficiency_ranking.png"
    )

    plot_data = ranking.sort_values(
        "cace_deviation_percent",
        ascending=True,
    )

    colors = np.where(
        plot_data["cace_deviation_percent"] > 0,
        "#B45309",
        "#2563EB",
    )

    fig, axis = plt.subplots(figsize=(9, 5))

    bars = axis.barh(
        plot_data["vehicle"],
        plot_data["cace_deviation_percent"],
        color=colors,
    )

    axis.axvline(
        0,
        color="#64748B",
        linewidth=1,
    )

    axis.set_title(
        "CACE V1 — LOVO Vehicle Review Priority",
        fontweight="bold",
    )
    axis.set_xlabel(
        "Aggregate window fuel deviation (%)"
    )
    axis.set_ylabel("Vehicle")
    axis.grid(
        axis="x",
        color="#E2E8F0",
        linewidth=0.8,
    )
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)

    axis.bar_label(
        bars,
        labels=[
            f"{value:+.2f}%"
            for value in plot_data["cace_deviation_percent"]
        ],
        padding=4,
        fontsize=10,
    )

    minimum_value = min(
        plot_data["cace_deviation_percent"].min(),
        0,
    )
    maximum_value = max(
        plot_data["cace_deviation_percent"].max(),
        0,
    )
    value_range = maximum_value - minimum_value
    padding = max(value_range * 0.15, 1)

    axis.set_xlim(
        minimum_value - padding,
        maximum_value + padding,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Positive values indicate actual fuel above CACE expected. "
            "Results are diagnostic and do not represent confirmed fuel waste."
        ),
        ha="center",
        fontsize=9,
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()

    summary = load_vehicle_deviation(
        args.input_file.resolve()
    )
    validate_vehicle_deviation(summary)

    ranking = build_vehicle_ranking(summary)

    output_dir = args.output_dir.resolve()

    table_path = save_vehicle_ranking(
        ranking,
        output_dir,
    )
    figure_path = save_ranking_plot(
        ranking,
        output_dir,
    )

    print(
        ranking[
            [
                "review_priority_rank",
                "vehicle",
                "cace_deviation_percent",
                "positive_cace_deviation_window_rate_percent",
                "review_status",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved table: {table_path}")
    print(f"Saved figure: {figure_path}")
    print(
        "Note: Ranking is a diagnostic review priority, "
        "not a confirmed measure of inefficiency or fuel waste."
    )


if __name__ == "__main__":
    main()
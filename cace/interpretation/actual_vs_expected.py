# Compares actual fuel with Physics and CACE expected fuel on the final test set.
# The script exports only aggregated performance metrics and decile-level results.
# Vehicle identifiers, timestamps, and row-level predictions are not exported.


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from shap_global_importance import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    INPUT_FILES,
    find_artifact,
)


MODEL_COLORS = {
    "Actual Fuel": "#0F172A",
    "Physics Baseline": "#64748B",
    "CACE V1": "#0F766E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing the private final-test predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated interpretation outputs.",
    )
    return parser.parse_args()


def load_final_predictions(
    artifacts_dir: Path,
) -> pd.DataFrame:
    prediction_path = find_artifact(
        artifacts_dir,
        INPUT_FILES["final_predictions"],
    )
    return pd.read_csv(prediction_path)


def validate_predictions(
    predictions: pd.DataFrame,
) -> None:
    required_columns = [
        "actual_fuel_used",
        "physics_expected_fuel",
        "cace_expected_fuel",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Final-test predictions are missing columns: {missing_columns}"
        )

    if predictions.empty:
        raise ValueError(
            "The final-test prediction dataset is empty."
        )

    if predictions[required_columns].isna().any().any():
        raise ValueError(
            "Final-test predictions contain missing fuel values."
        )

    values = predictions[required_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Final-test predictions contain non-finite values."
        )

    if "split" in predictions.columns:
        split_values = set(
            predictions["split"]
            .astype("string")
            .str.lower()
            .dropna()
        )

        if split_values != {"test"}:
            raise ValueError(
                f"Expected only untouched test rows, found: {split_values}"
            )


def calculate_metrics(
    actual: pd.Series,
    expected: pd.Series,
) -> dict[str, float]:
    residual = actual - expected

    return {
        "mae_l_per_window": float(
            mean_absolute_error(actual, expected)
        ),
        "rmse_l_per_window": float(
            mean_squared_error(actual, expected) ** 0.5
        ),
        "r2": float(
            r2_score(actual, expected)
        ),
        "mean_residual_l_per_window": float(
            residual.mean()
        ),
    }


def build_metric_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    actual = predictions["actual_fuel_used"]

    physics_metrics = calculate_metrics(
        actual,
        predictions["physics_expected_fuel"],
    )
    cace_metrics = calculate_metrics(
        actual,
        predictions["cace_expected_fuel"],
    )

    physics_mae = physics_metrics["mae_l_per_window"]
    physics_rmse = physics_metrics["rmse_l_per_window"]

    rows = [
        {
            "model": "Physics Baseline",
            "test_observations": len(predictions),
            "mean_actual_fuel_l_per_window": float(actual.mean()),
            "mean_expected_fuel_l_per_window": float(
                predictions["physics_expected_fuel"].mean()
            ),
            **physics_metrics,
            "mae_improvement_percent": 0.0,
            "rmse_improvement_percent": 0.0,
        },
        {
            "model": "CACE V1",
            "test_observations": len(predictions),
            "mean_actual_fuel_l_per_window": float(actual.mean()),
            "mean_expected_fuel_l_per_window": float(
                predictions["cace_expected_fuel"].mean()
            ),
            **cace_metrics,
            "mae_improvement_percent": (
                (physics_mae - cace_metrics["mae_l_per_window"])
                / physics_mae
                * 100
            ),
            "rmse_improvement_percent": (
                (physics_rmse - cace_metrics["rmse_l_per_window"])
                / physics_rmse
                * 100
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_decile_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions[
        [
            "actual_fuel_used",
            "physics_expected_fuel",
            "cace_expected_fuel",
        ]
    ].copy()

    unique_actual_values = frame["actual_fuel_used"].nunique()
    decile_count = min(10, unique_actual_values)

    if decile_count < 2:
        raise ValueError(
            "At least two unique actual-fuel values are required."
        )

    frame["actual_fuel_decile"] = pd.qcut(
        frame["actual_fuel_used"],
        q=decile_count,
        labels=False,
        duplicates="drop",
    )

    summary = (
        frame.groupby(
            "actual_fuel_decile",
            observed=True,
        )
        .agg(
            observations=("actual_fuel_used", "size"),
            actual_fuel_min_l=("actual_fuel_used", "min"),
            actual_fuel_max_l=("actual_fuel_used", "max"),
            mean_actual_fuel_l=("actual_fuel_used", "mean"),
            mean_physics_expected_fuel_l=(
                "physics_expected_fuel",
                "mean",
            ),
            mean_cace_expected_fuel_l=(
                "cace_expected_fuel",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["actual_fuel_decile"] = (
        summary["actual_fuel_decile"].astype(int) + 1
    )

    return summary


def save_tables(
    metric_summary: pd.DataFrame,
    decile_summary: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    metric_path = (
        table_dir / "actual_vs_expected_metrics_public.csv"
    )
    decile_path = (
        table_dir / "actual_vs_expected_deciles_public.csv"
    )

    metric_summary.to_csv(
        metric_path,
        index=False,
        float_format="%.8f",
    )
    decile_summary.to_csv(
        decile_path,
        index=False,
        float_format="%.8f",
    )

    return metric_path, decile_path


def save_comparison_plot(
    metric_summary: pd.DataFrame,
    decile_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        figure_dir / "actual_vs_expected_comparison.png"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2),
    )

    decile_axis = axes[0]
    metric_axis = axes[1]

    decile_axis.plot(
        decile_summary["actual_fuel_decile"],
        decile_summary["mean_actual_fuel_l"],
        color=MODEL_COLORS["Actual Fuel"],
        marker="o",
        linewidth=2.2,
        label="Actual Fuel",
    )
    decile_axis.plot(
        decile_summary["actual_fuel_decile"],
        decile_summary["mean_physics_expected_fuel_l"],
        color=MODEL_COLORS["Physics Baseline"],
        marker="o",
        linewidth=2,
        label="Physics Baseline",
    )
    decile_axis.plot(
        decile_summary["actual_fuel_decile"],
        decile_summary["mean_cace_expected_fuel_l"],
        color=MODEL_COLORS["CACE V1"],
        marker="o",
        linewidth=2,
        label="CACE V1",
    )

    decile_axis.set_title(
        "Actual vs Expected Fuel by Actual-Fuel Decile",
        fontweight="bold",
    )
    decile_axis.set_xlabel("Actual-fuel decile")
    decile_axis.set_ylabel("Mean fuel (liters per window)")
    decile_axis.set_xticks(
        decile_summary["actual_fuel_decile"]
    )
    decile_axis.grid(
        color="#E2E8F0",
        linewidth=0.8,
    )
    decile_axis.set_axisbelow(True)
    decile_axis.legend(frameon=False)

    metric_names = ["MAE", "RMSE"]
    physics_row = metric_summary.loc[
        metric_summary["model"] == "Physics Baseline"
    ].iloc[0]
    cace_row = metric_summary.loc[
        metric_summary["model"] == "CACE V1"
    ].iloc[0]

    physics_values = [
        physics_row["mae_l_per_window"],
        physics_row["rmse_l_per_window"],
    ]
    cace_values = [
        cace_row["mae_l_per_window"],
        cace_row["rmse_l_per_window"],
    ]

    positions = np.arange(len(metric_names))
    width = 0.34

    physics_bars = metric_axis.bar(
        positions - width / 2,
        physics_values,
        width,
        color=MODEL_COLORS["Physics Baseline"],
        label="Physics Baseline",
    )
    cace_bars = metric_axis.bar(
        positions + width / 2,
        cace_values,
        width,
        color=MODEL_COLORS["CACE V1"],
        label="CACE V1",
    )

    metric_axis.set_title(
        "Final-Test Error Comparison",
        fontweight="bold",
    )
    metric_axis.set_ylabel("Liters per window")
    metric_axis.set_xticks(
        positions,
        metric_names,
    )
    metric_axis.grid(
        axis="y",
        color="#E2E8F0",
        linewidth=0.8,
    )
    metric_axis.set_axisbelow(True)
    metric_axis.legend(frameon=False)

    for bars in [physics_bars, cace_bars]:
        for bar in bars:
            metric_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "CACE V1 — Untouched Final-Test Comparison",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
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

    predictions = load_final_predictions(
        args.artifacts_dir.resolve()
    )
    validate_predictions(predictions)

    metric_summary = build_metric_summary(predictions)
    decile_summary = build_decile_summary(predictions)

    output_dir = args.output_dir.resolve()

    metric_path, decile_path = save_tables(
        metric_summary,
        decile_summary,
        output_dir,
    )
    figure_path = save_comparison_plot(
        metric_summary,
        decile_summary,
        output_dir,
    )

    print(
        metric_summary[
            [
                "model",
                "test_observations",
                "mae_l_per_window",
                "rmse_l_per_window",
                "r2",
                "mae_improvement_percent",
                "rmse_improvement_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved metric table: {metric_path}")
    print(f"Saved decile table: {decile_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
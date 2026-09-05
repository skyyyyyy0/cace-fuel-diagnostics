# Analyzes CACE fuel deviation across out-of-sample operating-condition bands.
# LOVO predictions provide the primary vehicle-level evidence, while the final test
# is used as a supporting chronological check. Only aggregated results are exported.
# Only aggregated results are exported.
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_global_importance import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    find_artifact,
)


LOVO_PREDICTIONS_FILE = "CACE_v1.0_lovo_predictions_private.csv"
FINAL_PREDICTIONS_FILE = "CACE_v1.0_final_test_predictions_private.csv"

LOVO_SCOPE = "LOVO out-of-sample"
FINAL_SCOPE = "Untouched final test"

FEATURE_CONFIG = {
    "avg_rpm": {
        "label": "Average RPM",
        "unit": "rpm",
        "missing_indicator": None,
    },
    "engine_load": {
        "label": "Engine Load",
        "unit": "percent",
        "missing_indicator": None,
    },
    "engine_torque": {
        "label": "Engine Torque",
        "unit": "percent",
        "missing_indicator": None,
    },
    "avg_vehicle_speed": {
        "label": "Average Vehicle Speed",
        "unit": "km/h",
        "missing_indicator": "avg_vehicle_speed_missing",
    },
    "avg_coolant_temperature": {
        "label": "Average Coolant Temperature",
        "unit": "deg C",
        "missing_indicator": "avg_coolant_temperature_missing",
    },
}

BAND_LABELS = {
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing private CACE prediction artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated interpretation outputs.",
    )
    return parser.parse_args()


def load_predictions(
    artifacts_dir: Path,
    filename: str,
) -> pd.DataFrame:
    prediction_path = find_artifact(
        artifacts_dir,
        filename,
    )
    return pd.read_csv(prediction_path)


def validate_predictions(
    predictions: pd.DataFrame,
    scope: str,
) -> None:
    required_columns = [
        "actual_fuel_used",
        "cace_expected_fuel",
        *FEATURE_CONFIG.keys(),
    ]

    required_columns.extend(
        config["missing_indicator"]
        for config in FEATURE_CONFIG.values()
        if config["missing_indicator"] is not None
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{scope} is missing columns: {missing_columns}"
        )

    if predictions.empty:
        raise ValueError(
            f"{scope} prediction data is empty."
        )

    numeric_data = predictions[required_columns]

    if numeric_data.isna().any().any():
        raise ValueError(
            f"{scope} contains missing values after preprocessing."
        )

    if not np.isfinite(
        numeric_data.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{scope} contains non-finite values."
        )


def filter_observed_feature(
    predictions: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    missing_indicator = FEATURE_CONFIG[feature][
        "missing_indicator"
    ]

    if missing_indicator is None:
        return predictions.copy()

    return predictions.loc[
        predictions[missing_indicator] == 0
    ].copy()


def build_feature_bands(
    lovo_predictions: pd.DataFrame,
) -> dict[str, dict]:
    band_definitions = {}

    for feature in FEATURE_CONFIG:
        reference = filter_observed_feature(
            lovo_predictions,
            feature,
        )

        quantiles = reference[feature].quantile(
            [0, 1 / 3, 2 / 3, 1]
        )
        finite_edges = np.unique(
            quantiles.to_numpy(dtype=float)
        )

        band_count = len(finite_edges) - 1

        if band_count not in BAND_LABELS:
            raise ValueError(
                f"Unable to create operating-condition bands for {feature}."
            )

        cut_edges = finite_edges.copy()
        cut_edges[0] = -np.inf
        cut_edges[-1] = np.inf

        band_definitions[feature] = {
            "cut_edges": cut_edges,
            "labels": BAND_LABELS[band_count],
        }

    return band_definitions


def summarize_feature_bands(
    predictions: pd.DataFrame,
    scope: str,
    band_definitions: dict[str, dict],
) -> pd.DataFrame:
    rows = []

    for feature, config in FEATURE_CONFIG.items():
        frame = filter_observed_feature(
            predictions,
            feature,
        )

        frame["cace_residual_l"] = (
            frame["actual_fuel_used"]
            - frame["cace_expected_fuel"]
        )

        band_definition = band_definitions[feature]

        frame["condition_band"] = pd.cut(
            frame[feature],
            bins=band_definition["cut_edges"],
            labels=band_definition["labels"],
            include_lowest=True,
        )

        for condition_band, group in frame.groupby(
            "condition_band",
            observed=True,
        ):
            residual = group["cace_residual_l"]

            rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "feature_label": config["label"],
                    "unit": config["unit"],
                    "condition_band": str(condition_band),
                    "observations": len(group),
                    "feature_min": float(group[feature].min()),
                    "feature_max": float(group[feature].max()),
                    "feature_mean": float(group[feature].mean()),
                    "mean_cace_residual_l_per_window": float(
                        residual.mean()
                    ),
                    "median_cace_residual_l_per_window": float(
                        residual.median()
                    ),
                    "positive_deviation_rate_percent": float(
                        (residual > 0).mean() * 100
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_review_signals(
    condition_summary: pd.DataFrame,
) -> pd.DataFrame:
    join_columns = [
        "feature",
        "feature_label",
        "unit",
        "condition_band",
    ]

    value_columns = [
        "observations",
        "feature_min",
        "feature_max",
        "feature_mean",
        "mean_cace_residual_l_per_window",
        "median_cace_residual_l_per_window",
        "positive_deviation_rate_percent",
    ]

    lovo_summary = condition_summary.loc[
        condition_summary["scope"] == LOVO_SCOPE,
        join_columns + value_columns,
    ].rename(
        columns={
            column: f"{column}_lovo"
            for column in value_columns
        }
    )

    final_summary = condition_summary.loc[
        condition_summary["scope"] == FINAL_SCOPE,
        join_columns + value_columns,
    ].rename(
        columns={
            column: f"{column}_final_test"
            for column in value_columns
        }
    )

    combined = lovo_summary.merge(
        final_summary,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    review_signals = combined.loc[
        (
            combined[
                "mean_cace_residual_l_per_window_lovo"
            ] > 0
        )
        & (
            combined[
                "positive_deviation_rate_percent_lovo"
            ] >= 50
        )
        & (
            combined[
                "mean_cace_residual_l_per_window_final_test"
            ] > 0
        )
        & (
            combined[
                "positive_deviation_rate_percent_final_test"
            ] >= 50
        )
    ].copy()

    review_signals[
        "minimum_cross_scope_mean_residual_l"
    ] = review_signals[
        [
            "mean_cace_residual_l_per_window_lovo",
            "mean_cace_residual_l_per_window_final_test",
        ]
    ].min(axis=1)

    review_signals = review_signals.sort_values(
        "minimum_cross_scope_mean_residual_l",
        ascending=False,
    ).reset_index(drop=True)

    review_signals.insert(
        0,
        "review_signal_rank",
        np.arange(1, len(review_signals) + 1),
    )

    return review_signals


def save_tables(
    condition_summary: pd.DataFrame,
    review_signals: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        table_dir / "operating_condition_summary_public.csv"
    )
    signal_path = (
        table_dir / "operating_condition_review_signals_public.csv"
    )

    condition_summary.to_csv(
        summary_path,
        index=False,
        float_format="%.8f",
    )
    review_signals.to_csv(
        signal_path,
        index=False,
        float_format="%.8f",
    )

    return summary_path, signal_path


def save_condition_plot(
    condition_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        figure_dir / "operating_condition_residuals.png"
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharey=True,
    )

    axes = axes.flatten()
    band_order = ["Low", "Medium", "High"]

    for axis, (feature, config) in zip(
        axes,
        FEATURE_CONFIG.items(),
    ):
        feature_summary = condition_summary.loc[
            condition_summary["feature"] == feature
        ].copy()

        available_bands = [
            band
            for band in band_order
            if band in feature_summary["condition_band"].unique()
        ]

        positions = np.arange(len(available_bands))
        width = 0.35

        lovo_values = (
            feature_summary.loc[
                feature_summary["scope"] == LOVO_SCOPE
            ]
            .set_index("condition_band")
            .loc[
                available_bands,
                "mean_cace_residual_l_per_window",
            ]
            .to_numpy()
        )

        final_values = (
            feature_summary.loc[
                feature_summary["scope"] == FINAL_SCOPE
            ]
            .set_index("condition_band")
            .loc[
                available_bands,
                "mean_cace_residual_l_per_window",
            ]
            .to_numpy()
        )

        axis.bar(
            positions - width / 2,
            lovo_values,
            width,
            color="#0F766E",
            label="LOVO out-of-sample",
        )
        axis.bar(
            positions + width / 2,
            final_values,
            width,
            color="#B45309",
            label="Final test",
        )

        axis.axhline(
            0,
            color="#64748B",
            linewidth=1,
        )

        axis.set_title(
            config["label"],
            fontweight="bold",
        )
        axis.set_xticks(
            positions,
            available_bands,
        )
        axis.set_xlabel("Condition band")
        axis.grid(
            axis="y",
            color="#E2E8F0",
            linewidth=0.8,
        )
        axis.set_axisbelow(True)

    for unused_axis in axes[len(FEATURE_CONFIG):]:
        unused_axis.axis("off")

    axes[0].set_ylabel(
        "Mean residual (liters per window)"
    )
    axes[3].set_ylabel(
        "Mean residual (liters per window)"
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )

    fig.suptitle(
        "CACE V1 — Fuel Deviation by Operating Condition",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.02,
        (
            "Residual = Actual Fuel − CACE Expected Fuel. "
            "Positive values indicate actual fuel above expected."
        ),
        ha="center",
        fontsize=10,
        color="#475569",
    )

    fig.tight_layout(
        rect=[0, 0.05, 1, 0.92]
    )
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
    artifacts_dir = args.artifacts_dir.resolve()

    lovo_predictions = load_predictions(
        artifacts_dir,
        LOVO_PREDICTIONS_FILE,
    )
    final_predictions = load_predictions(
        artifacts_dir,
        FINAL_PREDICTIONS_FILE,
    )

    validate_predictions(
        lovo_predictions,
        LOVO_SCOPE,
    )
    validate_predictions(
        final_predictions,
        FINAL_SCOPE,
    )

    band_definitions = build_feature_bands(
        lovo_predictions
    )

    condition_summary = pd.concat(
        [
            summarize_feature_bands(
                lovo_predictions,
                LOVO_SCOPE,
                band_definitions,
            ),
            summarize_feature_bands(
                final_predictions,
                FINAL_SCOPE,
                band_definitions,
            ),
        ],
        ignore_index=True,
    )

    review_signals = build_review_signals(
        condition_summary
    )

    output_dir = args.output_dir.resolve()

    summary_path, signal_path = save_tables(
        condition_summary,
        review_signals,
        output_dir,
    )
    figure_path = save_condition_plot(
        condition_summary,
        output_dir,
    )

    if review_signals.empty:
        print(
            "No operating-condition band met the review-signal criteria."
        )
    else:
        print(
            review_signals[
                [
                    "review_signal_rank",
                    "feature_label",
                    "condition_band",
                    "mean_cace_residual_l_per_window_lovo",
                    "positive_deviation_rate_percent_lovo",
                    "mean_cace_residual_l_per_window_final_test",
                    "positive_deviation_rate_percent_final_test",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print(f"\nSaved condition table: {summary_path}")
    print(f"Saved review-signal table: {signal_path}")
    print(f"Saved figure: {figure_path}")
    print(
        "Note: Operating-condition signals are diagnostic associations, "
        "not causal evidence of inefficiency."
    )


if __name__ == "__main__":
    main()
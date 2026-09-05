# Analyzes SHAP direction and dependence patterns for the CACE V1 residual model.
# The analysis uses only the untouched final-test data and exports aggregated tables
# and figures without vehicle identifiers, timestamps, or row-level SHAP values.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

from shap_global_importance import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_LABELS,
    build_global_importance,
    calculate_shap_values,
    fit_residual_model,
    load_inputs,
    validate_inputs,
    verify_model_reproduction,
)


DIRECTION_THRESHOLD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing private Phase 8 and Phase 9 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for aggregated interpretation outputs.",
    )
    return parser.parse_args()


def describe_direction(correlation: float) -> str:
    if pd.isna(correlation):
        return "Not estimable"

    if correlation >= DIRECTION_THRESHOLD:
        return "Higher values tend to increase the ML correction"

    if correlation <= -DIRECTION_THRESHOLD:
        return "Higher values tend to decrease the ML correction"

    return "No clear monotonic direction"


def build_direction_summary(
    expanded_test: pd.DataFrame,
    features: list[str],
    shap_values: np.ndarray,
    global_importance: pd.DataFrame,
) -> pd.DataFrame:
    importance_ranks = global_importance.set_index(
        "feature"
    )["importance_rank"]

    rows = []

    for feature_index, feature in enumerate(features):
        feature_values = expanded_test[feature].to_numpy(dtype=float)
        feature_shap = shap_values[:, feature_index]

        if np.unique(feature_values).size > 1:
            correlation, p_value = spearmanr(
                feature_values,
                feature_shap,
            )
        else:
            correlation = np.nan
            p_value = np.nan

        rows.append(
            {
                "importance_rank": int(importance_ranks.loc[feature]),
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "test_observations": len(feature_values),
                "feature_min": float(feature_values.min()),
                "feature_max": float(feature_values.max()),
                "mean_shap_l_per_window": float(feature_shap.mean()),
                "spearman_feature_vs_shap": float(correlation),
                "spearman_p_value": float(p_value),
                "direction": describe_direction(correlation),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("importance_rank")
        .reset_index(drop=True)
    )


def save_direction_table(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        table_dir / "shap_direction_summary_public.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    return output_path


def save_direction_plot(
    expanded_test: pd.DataFrame,
    features: list[str],
    shap_values: np.ndarray,
    global_importance: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        figure_dir / "shap_direction_summary.png"
    )

    ordered_features = (
        global_importance["feature"]
        .tolist()[::-1]
    )

    random_generator = np.random.default_rng(42)

    fig, axis = plt.subplots(figsize=(10, 5.8))

    for y_position, feature in enumerate(ordered_features):
        feature_index = features.index(feature)
        feature_values = expanded_test[feature].to_numpy(dtype=float)
        feature_shap = shap_values[:, feature_index]

        value_min = feature_values.min()
        value_max = feature_values.max()
        value_range = value_max - value_min

        if value_range == 0:
            normalized_values = np.zeros_like(feature_values)
        else:
            normalized_values = (
                feature_values - value_min
            ) / value_range

        jitter = random_generator.normal(
            loc=0,
            scale=0.07,
            size=len(feature_values),
        )

        axis.scatter(
            feature_shap,
            y_position + jitter,
            c=normalized_values,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=28,
            alpha=0.75,
            edgecolors="none",
        )

    axis.axvline(
        0,
        color="#64748B",
        linewidth=1,
    )

    axis.set_yticks(
        range(len(ordered_features)),
        [
            FEATURE_LABELS.get(feature, feature)
            for feature in ordered_features
        ],
    )

    axis.set_title(
        "CACE V1 ML Correction — SHAP Direction",
        fontweight="bold",
    )
    axis.set_xlabel(
        "SHAP value (change to ML correction, liters per window)"
    )
    axis.grid(
        axis="x",
        color="#E2E8F0",
        linewidth=0.8,
    )
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)

    color_scale = plt.cm.ScalarMappable(
        norm=Normalize(vmin=0, vmax=1),
        cmap="coolwarm",
    )
    color_scale.set_array([])

    colorbar = fig.colorbar(
        color_scale,
        ax=axis,
        pad=0.02,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Relative feature value")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return output_path


def build_dependence_trend(
    feature_values: np.ndarray,
    feature_shap: np.ndarray,
    requested_bins: int = 8,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature_value": feature_values,
            "shap_value": feature_shap,
        }
    )

    unique_count = frame["feature_value"].nunique()
    bin_count = min(requested_bins, unique_count)

    if bin_count < 2:
        return pd.DataFrame()

    frame["feature_bin"] = pd.qcut(
        frame["feature_value"],
        q=bin_count,
        duplicates="drop",
    )

    return (
        frame.groupby(
            "feature_bin",
            observed=True,
        )
        .agg(
            mean_feature_value=("feature_value", "mean"),
            mean_shap_value=("shap_value", "mean"),
        )
        .reset_index(drop=True)
    )


def save_dependence_plot(
    expanded_test: pd.DataFrame,
    features: list[str],
    shap_values: np.ndarray,
    global_importance: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        figure_dir / "shap_dependence_top_features.png"
    )

    continuous_features = [
        feature
        for feature in global_importance["feature"]
        if not feature.endswith("_missing")
    ][:3]

    fig, axes = plt.subplots(
        1,
        len(continuous_features),
        figsize=(15, 4.8),
    )

    axes = np.atleast_1d(axes)

    for axis, feature in zip(axes, continuous_features):
        feature_index = features.index(feature)
        feature_values = expanded_test[feature].to_numpy(dtype=float)
        feature_shap = shap_values[:, feature_index]

        axis.scatter(
            feature_values,
            feature_shap,
            color="#0F766E",
            s=28,
            alpha=0.60,
            edgecolors="none",
        )

        trend = build_dependence_trend(
            feature_values,
            feature_shap,
        )

        if not trend.empty:
            axis.plot(
                trend["mean_feature_value"],
                trend["mean_shap_value"],
                color="#B45309",
                marker="o",
                linewidth=2,
                markersize=4,
                label="Binned mean",
            )

        axis.axhline(
            0,
            color="#64748B",
            linewidth=1,
        )

        axis.set_title(
            FEATURE_LABELS.get(feature, feature),
            fontweight="bold",
        )
        axis.set_xlabel("Feature value")
        axis.grid(
            color="#E2E8F0",
            linewidth=0.8,
        )
        axis.set_axisbelow(True)
        axis.legend(frameon=False)

    axes[0].set_ylabel(
        "SHAP value (liters per window)"
    )

    fig.suptitle(
        "CACE V1 — SHAP Dependence Patterns",
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

    inputs = load_inputs(
        args.artifacts_dir.resolve()
    )
    features = validate_inputs(inputs)

    expanded_train = inputs["expanded_train"]
    physics_train = inputs["physics_train"]
    expanded_test = inputs["expanded_test"]
    final_predictions = inputs["final_predictions"]
    model_metrics = inputs["model_metrics"]

    assert isinstance(expanded_train, pd.DataFrame)
    assert isinstance(physics_train, pd.DataFrame)
    assert isinstance(expanded_test, pd.DataFrame)
    assert isinstance(final_predictions, pd.DataFrame)
    assert isinstance(model_metrics, dict)

    model = fit_residual_model(
        expanded_train,
        physics_train,
        features,
        model_metrics,
    )

    model_difference = verify_model_reproduction(
        model,
        expanded_test,
        final_predictions,
        features,
    )

    shap_values, shap_difference = calculate_shap_values(
        model,
        expanded_test,
        final_predictions,
        features,
    )

    global_importance = build_global_importance(
        features,
        shap_values,
    )

    direction_summary = build_direction_summary(
        expanded_test,
        features,
        shap_values,
        global_importance,
    )

    output_dir = args.output_dir.resolve()

    table_path = save_direction_table(
        direction_summary,
        output_dir,
    )
    direction_plot_path = save_direction_plot(
        expanded_test,
        features,
        shap_values,
        global_importance,
        output_dir,
    )
    dependence_plot_path = save_dependence_plot(
        expanded_test,
        features,
        shap_values,
        global_importance,
        output_dir,
    )

    print(
        direction_summary[
            [
                "importance_rank",
                "feature_label",
                "spearman_feature_vs_shap",
                "direction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nModel reproduction max difference: "
        f"{model_difference:.3e}"
    )
    print(
        f"SHAP additivity max difference: "
        f"{shap_difference:.3e}"
    )
    print(f"Saved table: {table_path}")
    print(f"Saved direction figure: {direction_plot_path}")
    print(f"Saved dependence figure: {dependence_plot_path}")


if __name__ == "__main__":
    main()
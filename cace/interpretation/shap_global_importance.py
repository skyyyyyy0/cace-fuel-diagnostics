# Calculates global SHAP importance for the validated CACE V1 residual model.
# The script reconstructs the selected model, verifies it against the saved final-test
# predictions, and exports only aggregated feature-level results.


from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "interpretation"

INPUT_FILES = {
    "expanded_train": "CACE_ML_Expanded_train_v1.0_private.csv",
    "physics_train": "physics_train_v1.0_private.csv",
    "expanded_test": "CACE_ML_Expanded_test_v1.0_private.csv",
    "final_predictions": "CACE_v1.0_final_test_predictions_private.csv",
    "preprocessing": "expanded_ml_preprocessing_v1.0_private.json",
    "model_metrics": "random_forest_expanded_validation_metrics_v1.0_private.json",
}

FEATURE_LABELS = {
    "engine_torque": "Engine Torque",
    "avg_vehicle_speed": "Average Vehicle Speed",
    "avg_coolant_temperature": "Average Coolant Temperature",
    "avg_vehicle_speed_missing": "Speed Missing Indicator",
    "avg_coolant_temperature_missing": "Coolant Temperature Missing Indicator",
}

MODEL_TOLERANCE = 1e-10


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


def find_artifact(artifacts_dir: Path, filename: str) -> Path:
    matches = sorted(
        path for path in artifacts_dir.rglob(filename) if path.is_file()
    )

    if not matches:
        raise FileNotFoundError(f"Required artifact not found: {filename}")

    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise ValueError(f"Multiple artifacts found for {filename}: {locations}")

    return matches[0]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_inputs(artifacts_dir: Path) -> dict[str, pd.DataFrame | dict]:
    if not artifacts_dir.is_dir():
        raise NotADirectoryError(
            f"Artifacts directory not found: {artifacts_dir}"
        )

    inputs: dict[str, pd.DataFrame | dict] = {}

    for name, filename in INPUT_FILES.items():
        path = find_artifact(artifacts_dir, filename)
        inputs[name] = (
            read_json(path)
            if path.suffix == ".json"
            else pd.read_csv(path)
        )

    return inputs


def validate_row_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> None:
    keys = ["vehicle", "anchor_time_utc"]

    missing_left = [column for column in keys if column not in left.columns]
    missing_right = [column for column in keys if column not in right.columns]

    if missing_left or missing_right:
        raise ValueError(
            f"Alignment keys are missing: {left_name}={missing_left}, "
            f"{right_name}={missing_right}"
        )

    left_keys = left[keys].astype("string").reset_index(drop=True)
    right_keys = right[keys].astype("string").reset_index(drop=True)

    if not left_keys.equals(right_keys):
        raise ValueError(
            f"Rows do not align between {left_name} and {right_name}."
        )


def validate_inputs(
    inputs: dict[str, pd.DataFrame | dict],
) -> list[str]:
    expanded_train = inputs["expanded_train"]
    physics_train = inputs["physics_train"]
    expanded_test = inputs["expanded_test"]
    final_predictions = inputs["final_predictions"]
    preprocessing = inputs["preprocessing"]
    model_metrics = inputs["model_metrics"]

    if not all(
        isinstance(frame, pd.DataFrame)
        for frame in [
            expanded_train,
            physics_train,
            expanded_test,
            final_predictions,
        ]
    ):
        raise TypeError(
            "Expected tabular artifacts were not loaded as data frames."
        )

    if not isinstance(preprocessing, dict):
        raise TypeError(
            "Preprocessing metadata was not loaded as a JSON object."
        )

    if not isinstance(model_metrics, dict):
        raise TypeError(
            "Model metadata was not loaded as a JSON object."
        )

    features = preprocessing.get("ml_features")
    metric_features = model_metrics.get("features")

    if not isinstance(features, list) or not features:
        raise ValueError(
            "The preprocessing artifact does not define ml_features."
        )

    if features != metric_features:
        raise ValueError(
            "Feature order differs between preprocessing and model metadata."
        )

    for name, frame in [
        ("expanded_train", expanded_train),
        ("expanded_test", expanded_test),
    ]:
        missing = [
            feature for feature in features if feature not in frame.columns
        ]

        if missing:
            raise ValueError(f"{name} is missing model features: {missing}")

        if frame[features].isna().any().any():
            raise ValueError(
                f"{name} contains missing values after preprocessing."
            )

    if "physics_residual" not in physics_train.columns:
        raise ValueError(
            "physics_train is missing physics_residual."
        )

    if "predicted_residual" not in final_predictions.columns:
        raise ValueError(
            "final_predictions is missing predicted_residual."
        )

    validate_row_alignment(
        expanded_train,
        physics_train,
        "expanded_train",
        "physics_train",
    )
    validate_row_alignment(
        expanded_test,
        final_predictions,
        "expanded_test",
        "final_predictions",
    )

    return features


def fit_residual_model(
    expanded_train: pd.DataFrame,
    physics_train: pd.DataFrame,
    features: list[str],
    model_metrics: dict,
) -> RandomForestRegressor:
    parameters = model_metrics.get("parameters")

    if not isinstance(parameters, dict):
        raise ValueError(
            "Model metadata does not contain Random Forest parameters."
        )

    model_parameters = parameters.copy()
    model_parameters["n_jobs"] = -1

    model = RandomForestRegressor(**model_parameters)
    model.fit(
        expanded_train[features],
        physics_train["physics_residual"],
    )

    return model


def verify_model_reproduction(
    model: RandomForestRegressor,
    expanded_test: pd.DataFrame,
    final_predictions: pd.DataFrame,
    features: list[str],
) -> float:
    reproduced = model.predict(expanded_test[features])
    saved = final_predictions["predicted_residual"].to_numpy(dtype=float)

    maximum_difference = float(
        np.abs(reproduced - saved).max()
    )

    if maximum_difference > MODEL_TOLERANCE:
        raise ValueError(
            "Reconstructed model does not match the saved final-test "
            f"predictions: max absolute difference={maximum_difference:.12g}"
        )

    return maximum_difference


def calculate_shap_values(
    model: RandomForestRegressor,
    expanded_test: pd.DataFrame,
    final_predictions: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float]:
    explanation = shap.TreeExplainer(model)(
        expanded_test[features],
        check_additivity=True,
    )

    shap_values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    ).reshape(-1)

    reconstructed = base_values + shap_values.sum(axis=1)
    saved = final_predictions["predicted_residual"].to_numpy(dtype=float)

    maximum_difference = float(
        np.abs(reconstructed - saved).max()
    )

    if maximum_difference > MODEL_TOLERANCE:
        raise ValueError(
            "SHAP values do not reconstruct the saved model output: "
            f"max absolute difference={maximum_difference:.12g}"
        )

    return shap_values, maximum_difference


def build_global_importance(
    features: list[str],
    shap_values: np.ndarray,
) -> pd.DataFrame:
    mean_absolute_shap = np.abs(shap_values).mean(axis=0)
    total_importance = float(mean_absolute_shap.sum())

    if total_importance <= 0:
        raise ValueError(
            "Global SHAP importance is zero for every feature."
        )

    summary = pd.DataFrame(
        {
            "feature": features,
            "feature_label": [
                FEATURE_LABELS.get(feature, feature)
                for feature in features
            ],
            "mean_absolute_shap_l_per_window": mean_absolute_shap,
            "importance_percent": (
                mean_absolute_shap / total_importance * 100
            ),
        }
    ).sort_values(
        "mean_absolute_shap_l_per_window",
        ascending=False,
    )

    summary.insert(
        0,
        "importance_rank",
        np.arange(1, len(summary) + 1),
    )

    return summary.reset_index(drop=True)


def save_importance_table(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        table_dir / "shap_global_importance_public.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    return output_path


def save_importance_plot(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        figure_dir / "shap_global_importance.png"
    )

    plot_data = summary.sort_values(
        "mean_absolute_shap_l_per_window"
    )

    fig, axis = plt.subplots(figsize=(9, 5))

    bars = axis.barh(
        plot_data["feature_label"],
        plot_data["mean_absolute_shap_l_per_window"],
        color="#0F766E",
    )

    axis.set_title(
        "CACE V1 ML Correction — Global SHAP Importance",
        fontweight="bold",
    )
    axis.set_xlabel(
        "Mean |SHAP value| (liters per window)"
    )
    axis.grid(
        axis="x",
        color="#E2E8F0",
        linewidth=0.8,
    )
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)

    for bar, importance in zip(
        bars,
        plot_data["importance_percent"],
    ):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {importance:.1f}%",
            va="center",
            color="#334155",
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

    summary = build_global_importance(
        features,
        shap_values,
    )

    output_dir = args.output_dir.resolve()

    table_path = save_importance_table(
        summary,
        output_dir,
    )
    figure_path = save_importance_plot(
        summary,
        output_dir,
    )

    print(
        summary.to_string(
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
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
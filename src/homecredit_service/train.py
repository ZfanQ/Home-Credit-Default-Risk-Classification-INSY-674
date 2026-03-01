from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from homecredit_service.features import build_training_frame
from homecredit_service.modeling import save_bundle, train_lightgbm_model

logger = logging.getLogger(__name__)

SNAPSHOT_PACKAGES = (
    "fastapi",
    "uvicorn",
    "pydantic-settings",
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "joblib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Home Credit default risk LightGBM model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("homecreditdefaultriskdata"),
        help="Directory containing all raw Home Credit CSV files.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row count for faster experimentation.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default decision threshold (used directly when threshold optimization is disabled).",
    )
    parser.add_argument(
        "--optimize-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize threshold on validation data using configured FP/FN costs.",
    )
    parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=1.0,
        help="Relative business cost for false positives in threshold optimization.",
    )
    parser.add_argument(
        "--false-negative-cost",
        type=float,
        default=5.0,
        help="Relative business cost for false negatives in threshold optimization.",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=101,
        help="Number of thresholds evaluated in the optimization grid.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of stratified folds for CV AUC estimation (set 0 to disable).",
    )
    parser.add_argument(
        "--cv-max-estimators",
        type=int,
        default=600,
        help="Max trees per CV fold to cap compute cost.",
    )
    parser.add_argument(
        "--temporal-holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction for most-recent temporal holdout diagnostic evaluation.",
    )
    parser.add_argument(
        "--temporal-max-estimators",
        type=int,
        default=600,
        help="Max trees for temporal holdout diagnostic model.",
    )
    parser.add_argument(
        "--subgroup-min-size",
        type=int,
        default=500,
        help="Minimum rows required before reporting subgroup metrics.",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Evaluate on held-out test set and include test_auc in the report.",
    )
    parser.add_argument(
        "--force-test-eval",
        action="store_true",
        help="Allow recomputing test metrics even if an existing report already includes test_auc.",
    )
    return parser.parse_args()


def collect_dependency_snapshot() -> dict[str, object]:
    packages: dict[str, str] = {}
    for package_name in SNAPSHOT_PACKAGES:
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = "not-installed"

    return {
        "python_version": platform.python_version(),
        "packages": packages,
    }


def _as_float_list(raw_value: object) -> list[float] | None:
    if not isinstance(raw_value, list):
        return None
    values: list[float] = []
    for item in raw_value:
        if not isinstance(item, (int, float)):
            return None
        values.append(float(item))
    return values


def _as_2x2_matrix(raw_value: object) -> list[list[float]] | None:
    if not (isinstance(raw_value, list) and len(raw_value) == 2):
        return None
    parsed_rows: list[list[float]] = []
    for row in raw_value:
        if not (isinstance(row, list) and len(row) == 2):
            return None
        parsed_row: list[float] = []
        for value in row:
            if not isinstance(value, (int, float)):
                return None
            parsed_row.append(float(value))
        parsed_rows.append(parsed_row)
    return parsed_rows


def _as_dict_list(raw_value: object) -> list[dict[str, object]] | None:
    if not isinstance(raw_value, list):
        return None
    rows: list[dict[str, object]] = []
    for item in raw_value:
        if not isinstance(item, dict):
            return None
        parsed: dict[str, object] = {}
        for key, value in item.items():
            if not isinstance(key, str):
                return None
            parsed[key] = value
        rows.append(parsed)
    return rows


def write_learning_curve_csv(learning_curves: dict[str, object], output_path: Path) -> None:
    iterations = _as_float_list(learning_curves.get("iterations", []))
    train_auc = _as_float_list(learning_curves.get("train_auc", []))
    validation_auc = _as_float_list(learning_curves.get("validation_auc", []))
    if iterations is None or train_auc is None or validation_auc is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["iteration", "train_auc", "validation_auc"])
        for iteration, train_value, validation_value in zip(
            iterations, train_auc, validation_auc, strict=False
        ):
            writer.writerow([iteration, train_value, validation_value])


def write_confusion_matrix_csv(
    matrix_payload: dict[str, object], split_name: str, output_path: Path
) -> None:
    matrix = _as_2x2_matrix(matrix_payload.get("matrix", []))
    normalized_matrix = _as_2x2_matrix(matrix_payload.get("normalized_matrix", []))
    threshold = matrix_payload.get("threshold")
    if matrix is None:
        return
    has_normalized = normalized_matrix is not None
    matrix_int = [[int(value) for value in row] for row in matrix]
    normalized = normalized_matrix if normalized_matrix is not None else [[0.0, 0.0], [0.0, 0.0]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", split_name])
        writer.writerow(["threshold", threshold])
        writer.writerow([])
        writer.writerow(
            ["actual_class", "pred_0_count", "pred_1_count", "pred_0_rate", "pred_1_rate"]
        )
        writer.writerow(
            [
                "true_0",
                matrix_int[0][0],
                matrix_int[0][1],
                normalized[0][0] if has_normalized else "",
                normalized[0][1] if has_normalized else "",
            ]
        )
        writer.writerow(
            [
                "true_1",
                matrix_int[1][0],
                matrix_int[1][1],
                normalized[1][0] if has_normalized else "",
                normalized[1][1] if has_normalized else "",
            ]
        )


def write_learning_curve_plot(learning_curves: dict[str, object], output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skipping learning curve plot export.")
        return False

    iterations = _as_float_list(learning_curves.get("iterations", []))
    train_auc = _as_float_list(learning_curves.get("train_auc", []))
    validation_auc = _as_float_list(learning_curves.get("validation_auc", []))
    if iterations is None or train_auc is None or validation_auc is None or len(iterations) == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(iterations, train_auc, label="Train AUC", linewidth=2.0)
    axis.plot(iterations, validation_auc, label="Validation AUC", linewidth=2.0)

    best_iteration = learning_curves.get("best_iteration")
    if isinstance(best_iteration, int) and 1 <= best_iteration <= len(validation_auc):
        best_validation_auc = float(validation_auc[best_iteration - 1])
        axis.axvline(best_iteration, linestyle="--", alpha=0.5, color="#666666")
        axis.scatter(
            [best_iteration],
            [best_validation_auc],
            s=40,
            color="#d62728",
            label=f"Best Iteration ({best_iteration})",
            zorder=3,
        )

    axis.set_title("Train vs Validation AUC by Boosting Iteration")
    axis.set_xlabel("Boosting Iteration")
    axis.set_ylabel("AUC")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def write_confusion_matrix_plot(
    matrix_payload: dict[str, object], split_name: str, output_path: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skipping confusion matrix plot export.")
        return False

    matrix = _as_2x2_matrix(matrix_payload.get("matrix", []))
    normalized_matrix = _as_2x2_matrix(matrix_payload.get("normalized_matrix", []))
    threshold = matrix_payload.get("threshold")

    if matrix is None:
        return False

    valid_normalized = normalized_matrix is not None
    matrix_int = [[int(value) for value in row] for row in matrix]
    normalized = normalized_matrix if normalized_matrix is not None else [[0.0, 0.0], [0.0, 0.0]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix_int, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")

    axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], labels=["True 0", "True 1"])

    for row_idx in range(2):
        for col_idx in range(2):
            count = matrix_int[row_idx][col_idx]
            if valid_normalized:
                rate = normalized[row_idx][col_idx]
                label = f"{count}\n({rate:.3f})"
            else:
                label = str(count)
            axis.text(col_idx, row_idx, label, ha="center", va="center", color="black")

    axis.set_title(f"{split_name.title()} Confusion Matrix (threshold={threshold})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def write_calibration_csv(
    calibration_payload: dict[str, object], split_name: str, output_path: Path
) -> None:
    bins = _as_dict_list(calibration_payload.get("bins", []))
    if bins is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", split_name])
        writer.writerow(["brier_score", calibration_payload.get("brier_score")])
        writer.writerow(
            ["expected_calibration_error", calibration_payload.get("expected_calibration_error")]
        )
        writer.writerow([])
        writer.writerow(
            [
                "bin_index",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_predicted_probability",
                "observed_default_rate",
                "absolute_gap",
            ]
        )
        for row in bins:
            writer.writerow(
                [
                    row.get("bin_index"),
                    row.get("bin_lower"),
                    row.get("bin_upper"),
                    row.get("count"),
                    row.get("mean_predicted_probability"),
                    row.get("observed_default_rate"),
                    row.get("absolute_gap"),
                ]
            )


def write_calibration_plot(
    calibration_payload: dict[str, object], split_name: str, output_path: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skipping calibration plot export.")
        return False

    bins = _as_dict_list(calibration_payload.get("bins", []))
    if bins is None:
        return False

    x_points: list[float] = []
    y_points: list[float] = []
    for row in bins:
        predicted = row.get("mean_predicted_probability")
        observed = row.get("observed_default_rate")
        count = row.get("count")
        if not (
            isinstance(predicted, (int, float))
            and isinstance(observed, (int, float))
            and isinstance(count, int)
            and count > 0
        ):
            continue
        x_points.append(float(predicted))
        y_points.append(float(observed))

    if not x_points:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", label="Perfect calibration")
    axis.plot(x_points, y_points, marker="o", linewidth=2.0, color="#1f77b4", label="Model")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed default rate")
    axis.set_title(f"{split_name.title()} Calibration Curve")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def write_cost_sensitivity_csv(cost_payload: dict[str, object], output_path: Path) -> None:
    rows = _as_dict_list(cost_payload.get("rows", []))
    if rows is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "false_positive_cost",
                "false_negative_cost",
                "selected_threshold",
                "expected_cost_per_applicant",
                "recall",
                "specificity",
                "approval_rate",
                "default_rate_within_approved",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("false_positive_cost"),
                    row.get("false_negative_cost"),
                    row.get("selected_threshold"),
                    row.get("expected_cost_per_applicant"),
                    row.get("recall"),
                    row.get("specificity"),
                    row.get("approval_rate"),
                    row.get("default_rate_within_approved"),
                ]
            )


def write_cost_sensitivity_plot(cost_payload: dict[str, object], output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skipping cost-sensitivity plot export.")
        return False

    rows = _as_dict_list(cost_payload.get("rows", []))
    if rows is None:
        return False

    x_costs: list[float] = []
    y_thresholds: list[float] = []
    for row in rows:
        fn_cost = row.get("false_negative_cost")
        threshold = row.get("selected_threshold")
        if isinstance(fn_cost, (int, float)) and isinstance(threshold, (int, float)):
            x_costs.append(float(fn_cost))
            y_thresholds.append(float(threshold))

    if not x_costs:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x_costs, y_thresholds, marker="o", linewidth=2.0, color="#2ca02c")
    axis.set_xlabel("False-negative cost weight")
    axis.set_ylabel("Selected threshold")
    axis.set_title("Threshold Sensitivity to Cost Weights")
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def write_subgroup_performance_csv(
    subgroup_payload: dict[str, object], split_name: str, output_path: Path
) -> None:
    columns_payload = subgroup_payload.get("columns")
    if not isinstance(columns_payload, dict):
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "column",
                "group",
                "count",
                "positive_rate",
                "auc",
                "pr_auc",
                "recall",
                "specificity",
                "approval_rate",
                "default_rate_within_approved",
            ]
        )
        for raw_column_name, raw_payload in columns_payload.items():
            if not (isinstance(raw_column_name, str) and isinstance(raw_payload, dict)):
                continue
            payload: dict[str, object] = {}
            for key, value in raw_payload.items():
                if isinstance(key, str):
                    payload[key] = value
            groups = _as_dict_list(payload.get("groups", []))
            if groups is None:
                continue
            for row in groups:
                writer.writerow(
                    [
                        raw_column_name,
                        row.get("group"),
                        row.get("count"),
                        row.get("positive_rate"),
                        row.get("auc"),
                        row.get("pr_auc"),
                        row.get("recall"),
                        row.get("specificity"),
                        row.get("approval_rate"),
                        row.get("default_rate_within_approved"),
                    ]
                )
    logger.info("%s subgroup metrics saved to %s", split_name.title(), output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    logger.info(
        "Starting training with data_dir=%s artifact_dir=%s",
        args.data_dir,
        args.artifact_dir,
    )

    report_path = args.artifact_dir / "training_report.json"
    if args.final_eval and not args.force_test_eval and report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("test_auc") is not None:
            msg = (
                f"Refusing to recompute test metrics because {report_path} "
                "already contains test_auc. "
                "Use --force-test-eval to override."
            )
            raise RuntimeError(msg)

    training_frame = build_training_frame(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    logger.info(
        "Training data prepared rows=%d cols=%d",
        len(training_frame),
        training_frame.shape[1],
    )

    bundle = train_lightgbm_model(
        train_df=training_frame,
        random_state=args.random_state,
        valid_size=args.valid_size,
        test_size=args.test_size,
        prediction_threshold=args.threshold,
        evaluate_test=args.final_eval,
        optimize_threshold=args.optimize_threshold,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        threshold_grid_size=args.threshold_grid_size,
        cv_folds=args.cv_folds,
        cv_max_estimators=args.cv_max_estimators,
        temporal_holdout_fraction=args.temporal_holdout_fraction,
        temporal_max_estimators=args.temporal_max_estimators,
        subgroup_min_size=args.subgroup_min_size,
    )
    if args.final_eval:
        logger.info(
            "Training finished train_auc=%.6f validation_auc=%.6f test_auc=%.6f",
            bundle["metrics"]["train_auc"],
            bundle["metrics"]["validation_auc"],
            bundle["metrics"]["test_auc"],
        )
    else:
        logger.info(
            "Training finished train_auc=%.6f validation_auc=%.6f (test not evaluated)",
            bundle["metrics"]["train_auc"],
            bundle["metrics"]["validation_auc"],
        )

    artifact_path = args.artifact_dir / "model_bundle.joblib"
    save_bundle(bundle, artifact_path)
    logger.info("Model bundle saved to %s", artifact_path)

    report = {
        "artifact_path": str(artifact_path),
        "test_auc": bundle["metrics"].get("test_auc"),
        "train_auc": bundle["metrics"]["train_auc"],
        "validation_auc": bundle["metrics"]["validation_auc"],
        "pr_auc": {
            "train": bundle["metrics"].get("train_pr_auc"),
            "validation": bundle["metrics"].get("validation_pr_auc"),
            "test": bundle["metrics"].get("test_pr_auc"),
        },
        "scale_pos_weight": bundle["scale_pos_weight"],
        "imbalance_strategy": bundle.get("imbalance_strategy", "scale_pos_weight"),
        "focal_params": bundle.get("focal_params", {}),
        "candidate_metrics": bundle.get("candidate_metrics", {}),
        "feature_count": len(bundle["feature_columns"]),
        "train_rows": bundle["train_rows"],
        "valid_rows": bundle["valid_rows"],
        "test_rows": bundle["test_rows"],
        "trained_at_utc": bundle["trained_at_utc"],
        "dataset_size": bundle["dataset_summary"],
        "class_distribution": bundle["class_distribution"],
        "split_counts": bundle["split_counts"],
        "hyperparameters": bundle["hyperparameters"],
        "random_seed": bundle["random_seed"],
        "threshold_default": bundle.get("threshold_default"),
        "threshold_used": bundle["threshold_used"],
        "threshold_optimization": bundle.get("threshold_optimization", {}),
        "policy_simulation": bundle.get("policy_simulation", {}),
        "cost_sensitivity": bundle.get("cost_sensitivity", {}),
        "calibration": bundle.get("calibration", {}),
        "cross_validation": bundle.get("cross_validation", {}),
        "temporal_validation": bundle.get("temporal_validation", {}),
        "subgroup_performance": bundle.get("subgroup_performance", {}),
        "drift_summary": bundle.get("drift_summary", {}),
        "best_iteration": bundle.get("best_iteration"),
        "learning_curves": bundle.get("learning_curves", {}),
        "confusion_matrices": bundle.get("confusion_matrices", {}),
        "test_evaluated": bundle.get("test_evaluated", False),
        "roc_auc": {
            "train": bundle["metrics"]["train_auc"],
            "validation": bundle["metrics"]["validation_auc"],
            "test": bundle["metrics"].get("test_auc"),
        },
        "dependency_snapshot": collect_dependency_snapshot(),
        "report_generated_at_utc": datetime.now(UTC).isoformat(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Training report saved to %s", report_path)

    learning_curve_csv_path = args.artifact_dir / "learning_curves.csv"
    write_learning_curve_csv(bundle.get("learning_curves", {}), learning_curve_csv_path)
    logger.info("Learning curves saved to %s", learning_curve_csv_path)

    validation_matrix_csv_path = args.artifact_dir / "validation_confusion_matrix.csv"
    validation_matrix_payload = bundle.get("confusion_matrices", {}).get("validation", {})
    if isinstance(validation_matrix_payload, dict):
        write_confusion_matrix_csv(
            validation_matrix_payload,
            split_name="validation",
            output_path=validation_matrix_csv_path,
        )
        logger.info("Validation confusion matrix saved to %s", validation_matrix_csv_path)

    plots_dir = args.artifact_dir / "plots"
    learning_curve_plot_path = plots_dir / "train_validation_learning_curve.png"
    if write_learning_curve_plot(bundle.get("learning_curves", {}), learning_curve_plot_path):
        logger.info("Learning curve plot saved to %s", learning_curve_plot_path)

    for split_name in ("train", "validation", "test"):
        payload = bundle.get("confusion_matrices", {}).get(split_name, {})
        if not isinstance(payload, dict):
            continue
        matrix_plot_path = plots_dir / f"{split_name}_confusion_matrix.png"
        plot_written = write_confusion_matrix_plot(
            payload,
            split_name=split_name,
            output_path=matrix_plot_path,
        )
        if plot_written:
            logger.info(
                "%s confusion matrix plot saved to %s",
                split_name.title(),
                matrix_plot_path,
            )

    calibration_payload = bundle.get("calibration", {})
    if isinstance(calibration_payload, dict):
        for split_name in ("train", "validation", "test"):
            split_payload = calibration_payload.get(split_name, {})
            if not isinstance(split_payload, dict):
                continue
            calibration_csv_path = args.artifact_dir / f"{split_name}_calibration.csv"
            write_calibration_csv(
                split_payload,
                split_name=split_name,
                output_path=calibration_csv_path,
            )
            logger.info(
                "%s calibration table saved to %s",
                split_name.title(),
                calibration_csv_path,
            )
            calibration_plot_path = plots_dir / f"{split_name}_calibration_curve.png"
            if write_calibration_plot(
                split_payload, split_name=split_name, output_path=calibration_plot_path
            ):
                logger.info(
                    "%s calibration plot saved to %s",
                    split_name.title(),
                    calibration_plot_path,
                )

    cost_sensitivity_payload = bundle.get("cost_sensitivity", {})
    if isinstance(cost_sensitivity_payload, dict):
        cost_csv_path = args.artifact_dir / "cost_sensitivity.csv"
        write_cost_sensitivity_csv(cost_sensitivity_payload, output_path=cost_csv_path)
        logger.info("Cost sensitivity table saved to %s", cost_csv_path)
        cost_plot_path = plots_dir / "cost_sensitivity_threshold_curve.png"
        if write_cost_sensitivity_plot(cost_sensitivity_payload, output_path=cost_plot_path):
            logger.info("Cost sensitivity plot saved to %s", cost_plot_path)

    subgroup_payload = bundle.get("subgroup_performance", {})
    if isinstance(subgroup_payload, dict):
        for split_name in ("validation", "test"):
            split_payload = subgroup_payload.get(split_name, {})
            if not isinstance(split_payload, dict):
                continue
            csv_path = args.artifact_dir / f"{split_name}_subgroup_metrics.csv"
            write_subgroup_performance_csv(
                split_payload,
                split_name=split_name,
                output_path=csv_path,
            )

    drift_payload = bundle.get("drift_summary", {})
    if isinstance(drift_payload, dict):
        drift_path = args.artifact_dir / "drift_summary.json"
        drift_path.write_text(json.dumps(drift_payload, indent=2), encoding="utf-8")
        logger.info("Drift summary saved to %s", drift_path)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

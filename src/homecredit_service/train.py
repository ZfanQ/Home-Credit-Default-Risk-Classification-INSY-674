from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold metadata stored in the training report.",
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
        prediction_threshold=args.threshold,
    )
    logger.info(
        "Training finished train_auc=%.6f validation_auc=%.6f",
        bundle["metrics"]["train_auc"],
        bundle["metrics"]["validation_auc"],
    )

    artifact_path = args.artifact_dir / "model_bundle.joblib"
    save_bundle(bundle, artifact_path)
    logger.info("Model bundle saved to %s", artifact_path)

    report = {
        "artifact_path": str(artifact_path),
        "validation_auc": bundle["metrics"]["validation_auc"],
        "train_auc": bundle["metrics"]["train_auc"],
        "scale_pos_weight": bundle["scale_pos_weight"],
        "feature_count": len(bundle["feature_columns"]),
        "train_rows": bundle["train_rows"],
        "valid_rows": bundle["valid_rows"],
        "trained_at_utc": bundle["trained_at_utc"],
        "dataset_size": bundle["dataset_summary"],
        "class_distribution": bundle["class_distribution"],
        "split_counts": bundle["split_counts"],
        "hyperparameters": bundle["hyperparameters"],
        "random_seed": bundle["random_seed"],
        "threshold_used": bundle["threshold_used"],
        "roc_auc": {
            "train": bundle["metrics"]["train_auc"],
            "validation": bundle["metrics"]["validation_auc"],
        },
        "dependency_snapshot": collect_dependency_snapshot(),
        "report_generated_at_utc": datetime.now(UTC).isoformat(),
    }

    report_path = args.artifact_dir / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Training report saved to %s", report_path)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

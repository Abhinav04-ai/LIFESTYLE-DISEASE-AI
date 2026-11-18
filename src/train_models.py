import argparse
import json
import logging

import joblib
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from .config import DISEASES, EVALUATION_FILES, MODEL_FILES, MODEL_NAMES, MODEL_DIR
from .data_loader import get_features_and_target, load_dataset
from .evaluation import evaluate_model
from .preprocessing import build_pipelines

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def train_disease(disease: str, data_dir=None):
    """Train and persist one disease's models and held-out evaluation results."""
    spec = DISEASES[disease]
    frame = load_dataset(disease, data_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    x, y = get_features_and_target(frame, disease)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    pipelines = build_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {
        "disease": disease,
        "title": spec["title"],
        "features": spec["features"],
        "class_names": spec["classes"],
        "dataset_rows": int(len(frame)),
        "test_size": float(len(x_test) / len(frame)),
        "random_state": 42,
        "cv_scoring": "f1",
        "cv": {},
        "models": {},
        "x_test": x_test,
        "y_test": y_test,
    }

    for key, pipeline in pipelines.items():
        logging.info("Training %s / %s", spec["title"], MODEL_NAMES[key])
        pipeline.fit(x_train, y_train)
        results["models"][key] = evaluate_model(
            pipeline, x_test, y_test, spec["classes"]
        )
        scores = cross_val_score(
            build_pipelines()[key],
            x,
            y,
            cv=cv,
            scoring="f1",
        )
        results["cv"][key] = {
            "fold_scores": scores.tolist(),
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
        }
        joblib.dump(pipeline, MODEL_FILES[disease][key])

    joblib.dump(results, EVALUATION_FILES[disease])
    summary = {
        "dataset_rows": results["dataset_rows"],
        "models": {
            MODEL_NAMES[key]: {
                **results["models"][key]["metrics"],
                "five_fold_mean_f1": results["cv"][key]["mean_f1"],
                "five_fold_std_f1": results["cv"][key]["std_f1"],
            }
            for key in MODEL_NAMES
        },
    }
    (MODEL_DIR / f"{disease}_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logging.info("Saved model and evaluation files for %s.", spec["title"])
    return results


def train_all(data_dir=None):
    return {disease: train_disease(disease, data_dir) for disease in DISEASES}


def main():
    parser = argparse.ArgumentParser(
        description="Train the six lifestyle disease prediction models."
    )
    parser.add_argument(
        "--disease",
        choices=["all", *DISEASES],
        default="all",
        help="Train one disease or both (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        help="Optional directory containing the two source dataset files.",
    )
    args = parser.parse_args()
    diseases = list(DISEASES) if args.disease == "all" else [args.disease]
    for disease in diseases:
        train_disease(disease, args.data_dir)


if __name__ == "__main__":
    main()

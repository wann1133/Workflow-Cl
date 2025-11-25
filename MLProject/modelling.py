from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "namadataset_preprocessing/breast_cancer_preprocessed.csv"
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLflow Project entry point for Breast Cancer classification.")
    parser.add_argument("--data_path", type=Path, default=DATA_PATH, help="Path to preprocessed CSV.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Hold-out ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations for Logistic Regression.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength.")
    return parser.parse_args()


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    return df.drop(columns=["target"]), df["target"]


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("workflow-ci")
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="logreg-ci"):
        mlflow.log_params(
            {
                "data_path": str(data_path),
                "test_size": args.test_size,
                "random_state": args.random_state,
                "max_iter": args.max_iter,
                "C": args.C,
            }
        )
        model = LogisticRegression(
            max_iter=args.max_iter,
            C=args.C,
            penalty="l2",
            solver="lbfgs",
            multi_class="auto",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro"),
            "recall_macro": recall_score(y_test, preds, average="macro"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

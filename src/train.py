
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("linear-regression-demo")

# Load the dataset
dataset = load_diabetes(as_frame=True)
X = dataset.data[["bmi"]]
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Capture code provenance
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    commit = "unavailable"

# Train and track
with mlflow.start_run(run_name="bmi-linear-regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "test_r2": float(r2_score(y_test, predictions)),
    }

    mlflow.log_params({
        "model_type": "LinearRegression",
        "feature": "bmi",
        "test_size": 0.2,
        "random_state": 42,
    })

    mlflow.log_metrics(metrics)
    mlflow.set_tag("git_commit", commit)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="linear_regression_model",
        input_example=X_train.head(3),
    )

    print("Git commit:", commit)
    print("Intercept:", round(float(model.intercept_), 3))
    print("Coefficient:", round(float(model.coef_[0]), 3))

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print("MLflow run:", mlflow.active_run().info.run_id)

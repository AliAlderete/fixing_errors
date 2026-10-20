# %% [markdown]
# # Tips multivariable linear regression
# Exportable notebook-style Python script. Compares holdout, k-fold, and
# leave-one-out validation, then logs evaluation and provenance to MLflow.

# %%
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET = "tip"
NUMERIC_FEATURES = ["total_bill", "size"]
CATEGORICAL_FEATURES = ["sex", "smoker", "day", "time"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
	"""Preprocess numeric and categorical predictors and fit linear regression."""
	preprocessing = ColumnTransformer(
		[
			("numeric", "passthrough", NUMERIC_FEATURES),
			(
				"categorical",
				OneHotEncoder(handle_unknown="ignore"),
				CATEGORICAL_FEATURES,
			),
		]
	)
	return Pipeline(
		[("preprocessing", preprocessing), ("regressor", LinearRegression())]
	)


def provenance() -> dict:
	"""Return source hash and Git revision when available."""
	source = Path(__file__).resolve()
	try:
		revision = subprocess.run(
			["git", "rev-parse", "HEAD"],
			cwd=source.parent,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		revision = "unavailable"
	return {
		"source_file": source.name,
		"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
		"git_commit": revision,
	}


def regression_metrics(actual, predicted) -> dict:
	return {
		"r2": float(r2_score(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
		"mae": float(mean_absolute_error(actual, predicted)),
	}


def main() -> None:
	# Dataset and deterministic fingerprint for reproducibility.
	tips = sns.load_dataset("tips").dropna(subset=FEATURES + [TARGET]).copy()
	X, y = tips[FEATURES], tips[TARGET]
	data_hash = hashlib.sha256(
		pd.util.hash_pandas_object(tips, index=True).values.tobytes()
	).hexdigest()

	# Validation 1: single 80/20 train-test split.
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE
	)
	holdout_model = build_pipeline().fit(X_train, y_train)
	holdout = regression_metrics(y_test, holdout_model.predict(X_test))
	metrics = {f"holdout_{key}": value for key, value in holdout.items()}

	# Validation 2: shuffled five-fold cross-validation.
	scoring = {
		"r2": "r2",
		"rmse": "neg_root_mean_squared_error",
		"mae": "neg_mean_absolute_error",
	}
	kfold = cross_validate(
		build_pipeline(),
		X,
		y,
		cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
		scoring=scoring,
	)
	for name in scoring:
		values = kfold[f"test_{name}"]
		if name in ("rmse", "mae"):
			values = -values
		metrics[f"kfold_{name}_mean"] = float(values.mean())
		metrics[f"kfold_{name}_std"] = float(values.std())

	# Validation 3: leave-one-out cross-validation; aggregate out-of-fold errors.
	loo = cross_validate(
		build_pipeline(),
		X,
		y,
		cv=LeaveOneOut(),
		scoring={
			"rmse": "neg_root_mean_squared_error",
			"mae": "neg_mean_absolute_error",
		},
		n_jobs=-1,
	)
	metrics["loo_rmse_mean"] = float(-loo["test_rmse"].mean())
	metrics["loo_mae_mean"] = float(-loo["test_mae"].mean())

	code_info = provenance()
	mlflow.set_experiment("tips-multivariable-linear-regression")
	with mlflow.start_run() as run:
		mlflow.log_params(
			{
				"estimator": "LinearRegression",
				"target": TARGET,
				"features": ",".join(FEATURES),
				"numeric_features": ",".join(NUMERIC_FEATURES),
				"categorical_features": ",".join(CATEGORICAL_FEATURES),
				"holdout_test_size": 0.2,
				"random_state": RANDOM_STATE,
				"kfold_splits": 5,
				"validation_methods": "holdout,kfold,loo",
				"dataset_rows": len(tips),
			}
		)
		mlflow.log_metrics(metrics)
		mlflow.set_tags(
			{
				**code_info,
				"dataset": "seaborn.tips",
				"dataset_sha256": data_hash,
			}
		)
		mlflow.log_dict(
			{**code_info, "dataset": "seaborn.tips", "dataset_sha256": data_hash},
			"provenance.json",
		)

		# Refit on all rows for the tracked model artifact after validation.
		final_model = build_pipeline().fit(X, y)
		mlflow.sklearn.log_model(final_model, artifact_path="model")
		print(json.dumps({"run_id": run.info.run_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
	main()

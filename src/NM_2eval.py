"""Holdout and five-fold validation for the tips regression model"""

import json

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET = "tip"
NUMERIC_FEATURES = ["total_bill", "size"]
CATEGORICAL_FEATURES = ["sex", "smoker", "day", "time"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
	"""Preprocess the tips features and fit a linear regression model."""
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


def holdout_validation(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
	"""Fit on a train split and return test-set predictions as a vector."""
	X_train, X_test, y_train, _ = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE
	)
	model = build_pipeline().fit(X_train, y_train)
	return np.asarray(model.predict(X_test))


def kfold_validation(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
	"""Return out-of-fold predictions as a vector in the input row order."""
	cross_validation = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
	return np.asarray(cross_val_predict(build_pipeline(), X, y, cv=cross_validation))


def main() -> None:
	tips = sns.load_dataset("tips").dropna(subset=FEATURES + [TARGET])
	X, y = tips[FEATURES], tips[TARGET]
	outputs = {
		"holdout_predictions": holdout_validation(X, y).tolist(),
		"kfold_predictions": kfold_validation(X, y).tolist(),
	}
	print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
	main()
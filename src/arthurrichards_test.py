"""
Arthur Richards - GitHub Validation Exercise

Additional validation methods:
1. Repeated K-Fold Cross-Validation
2. Bootstrap Validation

Outputs returned as NumPy arrays.
"""

import numpy as np
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

from validation import build_pipeline, FEATURES, TARGET


def repeated_kfold_validation(X, y):
    """Return RMSE values for repeated 5-fold cross-validation."""

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    rmse_results = []

    for train_idx, test_idx in cv.split(X):
        model = build_pipeline()

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[test_idx])

        rmse = np.sqrt(
            mean_squared_error(y.iloc[test_idx], predictions)
        )

        rmse_results.append(rmse)

    return np.array(rmse_results)


def bootstrap_validation(X, y, n_iterations=100):
    """Return out-of-bag RMSE values from bootstrap samples."""

    rng = np.random.default_rng(42)
    rmse_results = []

    for _ in range(n_iterations):

        train_idx = rng.choice(
            len(X),
            size=len(X),
            replace=True
        )

        test_idx = np.setdiff1d(
            np.arange(len(X)),
            np.unique(train_idx)
        )

        if len(test_idx) == 0:
            continue

        model = clone(build_pipeline())

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[test_idx])

        rmse = np.sqrt(
            mean_squared_error(y.iloc[test_idx], predictions)
        )

        rmse_results.append(rmse)

    return np.array(rmse_results)


def main():

    tips = sns.load_dataset("tips").dropna(
        subset=FEATURES + [TARGET]
    )

    X = tips[FEATURES]
    y = tips[TARGET]

    repeated_results = repeated_kfold_validation(X, y)
    bootstrap_results = bootstrap_validation(X, y)

    print("\nREPEATED K-FOLD VALIDATION")
    print("RMSE array:", repeated_results)
    print("Mean RMSE:", repeated_results.mean())

    print("\nBOOTSTRAP VALIDATION")
    print("RMSE array:", bootstrap_results)
    print("Mean RMSE:", bootstrap_results.mean())

    assert isinstance(repeated_results, np.ndarray)
    assert isinstance(bootstrap_results, np.ndarray)

    print("\nBoth validation methods completed successfully.")


if __name__ == "__main__":
    main()
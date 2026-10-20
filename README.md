# linear-regression-mlflow

Run `python src/validation.py` to compare holdout, five-fold, and leave-one-out
validation for the tips regression model. The MLflow run records coefficient
means, standard deviations, and variances for each validation method, and
includes `coefficient_boxplot.png`, `validation_coefficients.csv`, and
`coefficient_stability.json` artifacts to inspect parameter stability.
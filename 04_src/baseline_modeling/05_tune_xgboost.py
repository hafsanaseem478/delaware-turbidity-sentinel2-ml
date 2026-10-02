from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import (
    PredefinedSplit,
    GridSearchCV,
)

from xgboost import XGBRegressor

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

OUTPUT = Path(
    "05_results/metrics/09_xgboost_tuning.json"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    selection = json.load(f)

features = selection["final_original_features"]

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

X = train[features]
y = train["MeasurementValue"]

cv = PredefinedSplit(
    test_fold=train["cv_fold"].values
)

print("=" * 80)
print("XGBOOST GRID SEARCH")
print("=" * 80)

print("\nFeatures:")

for feature in features:
    print(" -", feature)

print("\nTraining samples:", len(train))
print(
    "Held-out test samples:",
    (df["split"] == "test").sum()
)

print("\nTest set is NOT used.")

model = XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=1,
)

param_grid = {

    "n_estimators": [
        200,
        400,
    ],

    "learning_rate": [
        0.03,
        0.05,
    ],

    "max_depth": [
        2,
        3,
        4,
    ],

    "min_child_weight": [
        1,
        3,
    ],

    "subsample": [
        0.8,
        1.0,
    ],
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
    return_train_score=False,
)

search.fit(
    X,
    y
)

print("\n" + "=" * 80)
print("BEST XGBOOST")
print("=" * 80)

print("\nBest parameters:")

for key, value in search.best_params_.items():
    print(
        f"{key}: {value}"
    )

print(
    "\nBest mean CV RÂ²:",
    round(search.best_score_, 4)
)

best_index = search.best_index_

fold_scores = []

for i in range(5):

    score = search.cv_results_[
        f"split{i}_test_score"
    ][best_index]

    fold_scores.append(
        float(score)
    )

    print(
        f"Fold {i + 1} RÂ²: "
        f"{score:.4f}"
    )

result = {
    "features": features,
    "best_parameters": search.best_params_,
    "best_mean_cv_r2": float(
        search.best_score_
    ),
    "fold_r2": fold_scores,
}

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )

print("\nSaved:")
print(OUTPUT)

print(
    "\n281 test observations remain untouched."
)


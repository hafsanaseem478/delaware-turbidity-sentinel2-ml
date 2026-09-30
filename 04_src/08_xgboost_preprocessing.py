from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from xgboost import XGBRegressor


# ============================================================
# FILES
# ============================================================

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

OUTPUT = Path(
    "05_results/tables/08_xgboost_preprocessing.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:
    selection = json.load(f)


selected = selection["final_original_features"]

print("=" * 80)
print("XGBOOST PREPROCESSING COMPARISON")
print("=" * 80)

print("\nSelected original features:")
for x in selected:
    print(" -", x)


# ============================================================
# ADD CUBE-ROOT VERSION OF EVERY SELECTED PREDICTOR
#
# Santos used selected predictors + cube-root predictors.
# ============================================================

cube_data = {}

for feature in selected:

    new_name = f"{feature}_cbrt"

    # use existing column if already created
    if new_name in df.columns:
        cube_data[new_name] = df[new_name]
    else:
        cube_data[new_name] = np.cbrt(
            df[feature].astype(float)
        )

cube_df = pd.DataFrame(
    cube_data,
    index=df.index
)

# Only add columns that do not already exist
new_cols = [
    c for c in cube_df.columns
    if c not in df.columns
]

if new_cols:
    df = pd.concat(
        [df, cube_df[new_cols]],
        axis=1
    )


cube_features = [
    f"{x}_cbrt"
    for x in selected
]

ALL_FEATURES = (
    selected
    + cube_features
)

print("\nOriginal predictors:", len(selected))
print("Cube-root predictors:", len(cube_features))
print("Total model predictors:", len(ALL_FEATURES))


# ============================================================
# USE TRAINING DATA ONLY
# ============================================================

train = df[
    df["split"] == "train"
].copy()

X = train[ALL_FEATURES]
y = train["MeasurementValue"]

print("\nTraining samples:", len(train))
print("Held-out test samples:",
      (df["split"] == "test").sum())

print("\nTest set is NOT used in this script.")


# ============================================================
# XGBOOST
# Same fixed benchmark settings as feature selection.
# ============================================================

def make_xgb():

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )


# ============================================================
# PREPROCESSING OPTIONS
#
# Yeo-Johnson is fitted ONLY inside each CV training fold,
# preventing leakage.
# ============================================================

pipelines = {

    "none":
        Pipeline([
            ("model", make_xgb())
        ]),

    "standard":
        Pipeline([
            (
                "standard",
                StandardScaler()
            ),
            (
                "model",
                make_xgb()
            )
        ]),

    "yeo_johnson":
        Pipeline([
            (
                "yeo_johnson",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=False
                )
            ),
            (
                "model",
                make_xgb()
            )
        ]),

    "yeo_johnson_standard":
        Pipeline([
            (
                "yeo_johnson",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=False
                )
            ),
            (
                "standard",
                StandardScaler()
            ),
            (
                "model",
                make_xgb()
            )
        ]),
}


# ============================================================
# 5-FOLD CV
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\n" + "=" * 80)
print("5-FOLD TRAINING CV")
print("=" * 80)

for name, pipeline in pipelines.items():

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="r2",
        cv=cv,
        n_jobs=-1
    )

    mean_r2 = scores.mean()
    std_r2 = scores.std()

    results.append({
        "preprocessing": name,
        "mean_cv_r2": mean_r2,
        "std_cv_r2": std_r2,
    })

    print(
        f"{name:24s}"
        f" R² = {mean_r2:.4f}"
        f" ± {std_r2:.4f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_cv_r2",
    ascending=False
).reset_index(drop=True)

best = results_df.iloc[0]

print("\n" + "=" * 80)
print("BEST XGBOOST PREPROCESSING")
print("=" * 80)

print(
    "Method:",
    best["preprocessing"]
)

print(
    "Mean training CV R²:",
    round(best["mean_cv_r2"], 4)
)

print(
    "CV standard deviation:",
    round(best["std_cv_r2"], 4)
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\nIMPORTANT:")
print(
    "The 281-sample held-out test set "
    "has still NOT been evaluated."
)
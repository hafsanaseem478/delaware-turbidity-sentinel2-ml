from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

OUTPUT = Path(
    "05_results/tables/08_cube_root_comparison.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    selection = json.load(f)

selected = selection["final_original_features"]


# ============================================================
# TRAINING DATA ONLY
# ============================================================

train = df[df["split"] == "train"].copy()

y = train["MeasurementValue"]


# ============================================================
# CREATE CUBE-ROOT FEATURES
# ============================================================

cube_features = []

for feature in selected:

    name = f"{feature}_cbrt"

    if name not in train.columns:
        train[name] = np.cbrt(
            train[feature].astype(float)
        )

    cube_features.append(name)


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {

    # exactly what gave us ~0.7228
    "original_only":
        selected,

    # Santos-style addition
    "original_plus_cube_root":
        selected + cube_features,

    # useful diagnostic
    "cube_root_only":
        cube_features,
}


# ============================================================
# SAME XGBOOST AS BEFORE
# ============================================================

def make_model():

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


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# TEST THREE FEATURE CONFIGURATIONS
# ============================================================

results = []

print("=" * 80)
print("CUBE-ROOT FEATURE COMPARISON")
print("=" * 80)

print("\nTraining samples:", len(train))
print("Held-out test set: NOT USED\n")


for name, features in feature_sets.items():

    X = train[features]

    scores = cross_val_score(
        make_model(),
        X,
        y,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
    )

    results.append({
        "feature_set": name,
        "n_features": len(features),
        "mean_cv_r2": scores.mean(),
        "std_cv_r2": scores.std(),
    })

    print(
        f"{name:28s}"
        f" features={len(features):2d}"
        f" | R² = {scores.mean():.4f}"
        f" ± {scores.std():.4f}"
    )


results_df = (
    pd.DataFrame(results)
    .sort_values(
        "mean_cv_r2",
        ascending=False
    )
)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(results_df.to_string(index=False))


results_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
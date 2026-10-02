from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import PredefinedSplit, cross_val_score
from xgboost import XGBRegressor

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

OUTPUT = Path(
    "05_results/tables/08_locked_cube_root_comparison.csv"
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

y = train["MeasurementValue"].values

cv = PredefinedSplit(
    test_fold=train["cv_fold"].values
)

cube_features = []

for feature in features:

    name = f"{feature}_cbrt"

    train[name] = np.cbrt(
        train[feature].astype(float)
    )

    cube_features.append(name)

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

feature_sets = {

    "original_only":
        features,

    "original_plus_cube_root":
        features + cube_features,
}

results = []

print("=" * 80)
print("LOCKED CUBE-ROOT COMPARISON")
print("=" * 80)

print("\nTraining samples:", len(train))
print("Test set remains untouched.\n")

for name, cols in feature_sets.items():

    scores = cross_val_score(
        make_model(),
        train[cols],
        y,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
    )

    results.append({
        "feature_set": name,
        "n_features": len(cols),
        "mean_cv_r2": scores.mean(),
        "std_cv_r2": scores.std(),
    })

    print(
        f"{name:27s}"
        f" RÂ² = {scores.mean():.4f}"
        f" Â± {scores.std():.4f}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\nExpected original-only RÂ²:")
print(round(selection["best_band_cv_r2"], 4))

print("\nSaved:")
print(OUTPUT)


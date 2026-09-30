from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    PredefinedSplit,
    cross_val_score,
)

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
    "05_results/tables/08_feature_set_comparison.csv"
)


# ============================================================
# LOAD DATA
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


# ============================================================
# TRAINING DATA ONLY
# ============================================================

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

y = train["MeasurementValue"].values

cv = PredefinedSplit(
    test_fold=train["cv_fold"].values
)


print("=" * 80)
print("FEATURE SET COMPARISON")
print("=" * 80)

print("\nTraining samples:", len(train))
print(
    "Held-out test samples:",
    (df["split"] == "test").sum()
)

print("\nTest set is NOT used.")


# ============================================================
# SANTOS TRANSFERRED FEATURE SET
#
# Santos selected 13 indices:
#
# NDGI, GCI, AWEI, GNDVI, SWM, 3BSI,
# NDMI, EVI, BSI, SAVI, MSI, NDWI, NDSI
#
# GCI cannot be reproduced because B09 is unavailable.
#
# Santos then selected B02 + B06.
# ============================================================

SANTOS_FEATURES = [
    "NDGI",
    "AWEI",
    "GNDVI",
    "SWM",
    "3BSI",
    "NDMI",
    "EVI",
    "BSI",
    "SAVI",
    "MSI",
    "NDWI",
    "NDSI",
    "B02_rhow",
    "B06_rhow",
]


# ============================================================
# OUR DELAWARE / ACOLITE FEATURE SET
# Read directly from locked feature-selection result
# ============================================================

DELAWARE_FEATURES = (
    selection["final_original_features"]
)


# ============================================================
# CHECK COLUMNS
# ============================================================

for name, features in {
    "Santos": SANTOS_FEATURES,
    "Delaware": DELAWARE_FEATURES,
}.items():

    missing = [
        feature
        for feature in features
        if feature not in train.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}"
        )


# ============================================================
# SAME FIXED XGBOOST FOR BOTH
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


# ============================================================
# COMPARE
# ============================================================

feature_sets = {
    "santos_transferred": SANTOS_FEATURES,
    "delaware_selected": DELAWARE_FEATURES,
}

results = []

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
        "mean_cv_r2": float(scores.mean()),
        "std_cv_r2": float(scores.std()),
        "fold_1_r2": float(scores[0]),
        "fold_2_r2": float(scores[1]),
        "fold_3_r2": float(scores[2]),
        "fold_4_r2": float(scores[3]),
        "fold_5_r2": float(scores[4]),
    })

    print("\n" + "-" * 80)
    print(name.upper())
    print("-" * 80)

    print("Features:", len(features))

    for feature in features:
        print(" -", feature)

    print("\nFold R²:")

    for i, score in enumerate(
        scores,
        start=1
    ):
        print(
            f"Fold {i}: {score:.4f}"
        )

    print(
        "\nMean CV R²:",
        round(scores.mean(), 4)
    )

    print(
        "CV standard deviation:",
        round(scores.std(), 4)
    )


# ============================================================
# RESULTS
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values(
        "mean_cv_r2",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(
    results_df[
        [
            "feature_set",
            "n_features",
            "mean_cv_r2",
            "std_cv_r2",
        ]
    ].to_string(index=False)
)


# ============================================================
# CHECK OUR RESULT REPRODUCES 0.7182
# ============================================================

our_score = float(
    results_df.loc[
        results_df["feature_set"]
        == "delaware_selected",
        "mean_cv_r2"
    ].iloc[0]
)

expected = float(
    selection["best_band_cv_r2"]
)

print("\nExpected Delaware CV R²:")
print(round(expected, 4))

print("Reproduced Delaware CV R²:")
print(round(our_score, 4))

print(
    "Difference:",
    round(
        abs(
            expected
            - our_score
        ),
        8
    )
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

print(
    "\n281-sample test set remains untouched."
)
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    PredefinedSplit,
    cross_val_score,
)

from xgboost import XGBRegressor


# ============================================================
# FILES
# ============================================================

INPUT = Path(
    "01_data/processed/delaware_turbidity_features.csv"
)

SPLIT_OUTPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

INDEX_OUTPUT = Path(
    "05_results/tables/07_forward_index_selection.csv"
)

BAND_OUTPUT = Path(
    "05_results/tables/07_forward_band_selection.csv"
)

FEATURE_OUTPUT = Path(
    "05_results/metrics/07_selected_features.json"
)

INDEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

TARGET = "MeasurementValue"

INDEX_FEATURES = [
    "NDVI",
    "GNDVI",
    "EVI",
    "SAVI",
    "NDMI",
    "MSI",
    "NBRI",
    "BSI",
    "NDWI",
    "NDSI",
    "NDGI",
    "ARVI",
    "SIPI",
    "SWM",
    "AWEI",
    "3BSI",
]

BAND_FEATURES = [
    "B01_rhow",
    "B02_rhow",
    "B03_rhow",
    "B04_rhow",
    "B05_rhow",
    "B06_rhow",
    "B07_rhow",
    "B08_rhow",
    "B8A_rhow",
    "B11_rhow",
    "B12_rhow",
]


# ============================================================
# CHECK DATA
# ============================================================

required = INDEX_FEATURES + BAND_FEATURES + [TARGET]

df = df.dropna(
    subset=required
).copy()

# Stable row identifier
df = df.reset_index(drop=True)

df["sample_id"] = np.arange(len(df))


print("=" * 80)
print("LOCKED FEATURE SELECTION")
print("=" * 80)

print("\nTotal clean samples:")
print(len(df))


# ============================================================
# FIXED 80 / 20 TRAIN-TEST SPLIT
# ============================================================

train_ids, test_ids = train_test_split(
    df["sample_id"].values,
    test_size=0.20,
    random_state=42,
)

df["split"] = "train"

df.loc[
    df["sample_id"].isin(test_ids),
    "split"
] = "test"


# ============================================================
# IMPORTANT:
# Put training rows into a STABLE order
# before constructing CV folds.
# ============================================================

train_df = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

test_df = (
    df[df["split"] == "test"]
    .sort_values("sample_id")
    .copy()
)

print("\nTraining samples:")
print(len(train_df))

print("\nTest samples:")
print(len(test_df))


# ============================================================
# CREATE PERMANENT 5-FOLD ASSIGNMENTS
#
# cv_fold:
#   0-4 = training CV fold
#   -1  = held-out test sample
# ============================================================

df["cv_fold"] = -1

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for fold, (_, validation_positions) in enumerate(
    kf.split(train_df)
):

    validation_sample_ids = (
        train_df.iloc[validation_positions]["sample_id"]
    )

    df.loc[
        df["sample_id"].isin(validation_sample_ids),
        "cv_fold"
    ] = fold


# ============================================================
# RELOAD TRAIN DATA WITH SAVED FOLD LABELS
# ============================================================

train_df = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

print("\nTraining fold counts:")

print(
    train_df["cv_fold"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE SPLITS NOW
# ============================================================

df.to_csv(
    SPLIT_OUTPUT,
    index=False
)

print("\nSaved permanent split/folds:")
print(SPLIT_OUTPUT)


# ============================================================
# PREDEFINED CV
#
# Every later script can reproduce EXACTLY
# the same five folds.
# ============================================================

cv = PredefinedSplit(
    test_fold=train_df["cv_fold"].values
)

y_train = train_df[TARGET].values


# ============================================================
# XGBOOST BENCHMARK
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
# EVALUATION
# ============================================================

def evaluate_features(features):

    X = train_df[features].values

    scores = cross_val_score(
        make_model(),
        X,
        y_train,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
    )

    return (
        float(scores.mean()),
        float(scores.std()),
    )


# ============================================================
# FORWARD SELECTION
# ============================================================

def forward_selection(
    candidate_features,
    base_features=None,
    stage_name="FEATURE SELECTION",
):

    if base_features is None:
        base_features = []

    selected = []
    remaining = candidate_features.copy()

    history = []

    print("\n" + "=" * 80)
    print(stage_name)
    print("=" * 80)

    for step in range(
        1,
        len(candidate_features) + 1
    ):

        candidates = []

        for feature in remaining:

            features = (
                base_features
                + selected
                + [feature]
            )

            mean_r2, std_r2 = (
                evaluate_features(features)
            )

            candidates.append(
                (
                    feature,
                    mean_r2,
                    std_r2,
                )
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        best_feature, best_mean, best_std = (
            candidates[0]
        )

        selected.append(best_feature)
        remaining.remove(best_feature)

        history.append({
            "step": step,
            "added_feature": best_feature,
            "n_features":
                len(base_features)
                + len(selected),
            "mean_cv_r2": best_mean,
            "std_cv_r2": best_std,
        })

        print(
            f"Step {step:2d}: "
            f"{best_feature:10s} "
            f"| CV R² = "
            f"{best_mean:.4f} "
            f"± {best_std:.4f}"
        )

    history_df = pd.DataFrame(history)

    best_row = history_df[
        "mean_cv_r2"
    ].idxmax()

    best_n = int(
        history_df.loc[
            best_row,
            "step"
        ]
    )

    best_selected = selected[:best_n]

    best_score = float(
        history_df.loc[
            best_row,
            "mean_cv_r2"
        ]
    )

    return (
        best_selected,
        best_score,
        history_df,
    )


# ============================================================
# STAGE 1 — INDICES
# ============================================================

(
    selected_indices,
    best_index_r2,
    index_history,
) = forward_selection(
    INDEX_FEATURES,
    stage_name="STAGE 1 — INDEX SELECTION"
)

index_history.to_csv(
    INDEX_OUTPUT,
    index=False
)


print("\nBEST INDEX SET:")
print(selected_indices)

print(
    "Best index CV R²:",
    round(best_index_r2, 4)
)


# ============================================================
# STAGE 2 — BANDS
# ============================================================

(
    selected_bands,
    best_band_r2,
    band_history,
) = forward_selection(
    BAND_FEATURES,
    base_features=selected_indices,
    stage_name="STAGE 2 — BAND SELECTION"
)

band_history.to_csv(
    BAND_OUTPUT,
    index=False
)


final_features = (
    selected_indices
    + selected_bands
)


print("\n" + "=" * 80)
print("FINAL LOCKED FEATURES")
print("=" * 80)

for i, feature in enumerate(
    final_features,
    1
):
    print(
        f"{i:2d}. {feature}"
    )

print(
    "\nFinal training CV R²:",
    round(best_band_r2, 4)
)


# ============================================================
# SAVE
# ============================================================

output = {
    "n_total": int(len(df)),
    "n_train": int(len(train_df)),
    "n_test": int(len(test_df)),

    "cv_method":
        "fixed 5-fold PredefinedSplit",

    "selected_indices":
        selected_indices,

    "best_index_cv_r2":
        best_index_r2,

    "selected_bands":
        selected_bands,

    "best_band_cv_r2":
        best_band_r2,

    "final_original_features":
        final_features,

    "unavailable_index":
        "GCI (B09 unavailable)",
}

with open(
    FEATURE_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


print("\nSaved:")
print(INDEX_OUTPUT)
print(BAND_OUTPUT)
print(FEATURE_OUTPUT)

print("\n281-sample test set remains untouched.")
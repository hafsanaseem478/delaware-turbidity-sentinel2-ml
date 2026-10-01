from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from xgboost import XGBRegressor


# ============================================================
# FILES
# ============================================================

DATA_FILE = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

XGB_FILE = Path(
    "05_results/metrics/09_xgboost_tuning.json"
)

OUT_DIR = Path(
    "05_results/tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:
    feature_info = json.load(f)

with open(
    XGB_FILE,
    "r",
    encoding="utf-8"
) as f:
    xgb_info = json.load(f)


FEATURES = feature_info[
    "final_original_features"
]

TARGET = "MeasurementValue"


# ============================================================
# BASIC CHECKS
# ============================================================

print("=" * 90)
print("VALIDATION FORENSIC AUDIT")
print("=" * 90)

print("\nRows:", len(df))
print(
    "Train:",
    (df["split"] == "train").sum()
)
print(
    "Test:",
    (df["split"] == "test").sum()
)

print("\nFeatures:")

for f in FEATURES:
    print(" -", f)


required = (
    FEATURES
    + [
        TARGET,
        "sample_id",
        "scene",
        "site_no",
        "split",
    ]
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


if df[FEATURES + [TARGET]].isna().any().any():

    bad = (
        df[
            FEATURES + [TARGET]
        ]
        .isna()
        .sum()
    )

    print("\nMissing values:")
    print(
        bad[bad > 0]
    )

    raise ValueError(
        "Missing model values found."
    )


# ============================================================
# VERIFY ONE ROW PER STATION-SCENE
# ============================================================

duplicate_station_scene = (
    df.duplicated(
        subset=[
            "site_no",
            "scene",
        ],
        keep=False
    )
)


print("\n" + "=" * 90)
print("STATION-SCENE UNIQUENESS")
print("=" * 90)

print(
    "Duplicate station-scene rows:",
    int(
        duplicate_station_scene.sum()
    )
)

print(
    "Unique station-scene combinations:",
    df[
        ["site_no", "scene"]
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# ORIGINAL TRAIN/TEST SCENE OVERLAP
# ============================================================

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

test = (
    df[df["split"] == "test"]
    .sort_values("sample_id")
    .copy()
)


train_scenes = set(
    train["scene"].astype(str)
)

test_scenes = set(
    test["scene"].astype(str)
)

scene_overlap = (
    train_scenes
    & test_scenes
)


test_scene_seen = (
    test["scene"]
    .astype(str)
    .isin(train_scenes)
)


print("\n" + "=" * 90)
print("ORIGINAL TRAIN/TEST SCENE OVERLAP")
print("=" * 90)

print(
    "Unique training scenes:",
    len(train_scenes)
)

print(
    "Unique test scenes:",
    len(test_scenes)
)

print(
    "Scenes appearing in BOTH train and test:",
    len(scene_overlap)
)

print(
    "Test rows whose scene appears in training:",
    int(test_scene_seen.sum()),
    "of",
    len(test),
    f"({100 * test_scene_seen.mean():.1f}%)"
)


# ============================================================
# MODEL
# ============================================================

best = xgb_info[
    "best_parameters"
]


def make_xgb():

    return XGBRegressor(

        objective="reg:squarederror",

        learning_rate=
            best["learning_rate"],

        max_depth=
            best["max_depth"],

        min_child_weight=
            best["min_child_weight"],

        n_estimators=
            best["n_estimators"],

        subsample=
            best["subsample"],

        # These were fixed during tuning
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,

        random_state=42,
        n_jobs=-1,
    )


# ============================================================
# METRICS
# ============================================================

def calc_metrics(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted
        )
    )

    mae = mean_absolute_error(
        observed,
        predicted
    )

    mbe = np.mean(
        predicted - observed
    )

    return {
        "n":
            len(observed),

        "r2":
            r2_score(
                observed,
                predicted
            ),

        "rmse_fnu":
            rmse,

        "mae_fnu":
            mae,

        "mbe_fnu":
            mbe,
    }


# ============================================================
# A. REPRODUCE ORIGINAL FIXED TEST
# ============================================================

model = make_xgb()

model.fit(
    train[FEATURES],
    train[TARGET]
)

test_pred = model.predict(
    test[FEATURES]
)

original_metrics = calc_metrics(
    test[TARGET],
    test_pred
)


print("\n" + "=" * 90)
print("A. ORIGINAL FIXED 80/20 TEST")
print("=" * 90)

for key, value in original_metrics.items():

    if key == "n":
        print(
            f"{key}: {value}"
        )

    else:
        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# B. GROUPKFOLD BY EXACT SENTINEL-2 SCENE
#
# IMPORTANT:
# Every observation belonging to the same exact scene
# must remain completely inside either train or validation.
# ============================================================

print("\n" + "=" * 90)
print("B. 5-FOLD GROUPKFold BY EXACT SCENE")
print("=" * 90)


X_all = (
    df[FEATURES]
    .reset_index(drop=True)
)

y_all = (
    df[TARGET]
    .astype(float)
    .reset_index(drop=True)
)

scene_groups = (
    df["scene"]
    .astype(str)
    .reset_index(drop=True)
)


scene_cv = GroupKFold(
    n_splits=5
)


scene_fold_rows = []

scene_oof = np.full(
    len(df),
    np.nan
)


for fold_number, (
    train_idx,
    val_idx
) in enumerate(

    scene_cv.split(
        X_all,
        y_all,
        groups=scene_groups
    ),

    start=1
):

    model = make_xgb()

    model.fit(
        X_all.iloc[train_idx],
        y_all.iloc[train_idx]
    )

    pred = model.predict(
        X_all.iloc[val_idx]
    )

    scene_oof[
        val_idx
    ] = pred

    metrics = calc_metrics(
        y_all.iloc[val_idx],
        pred
    )

    val_values = (
        y_all.iloc[val_idx]
    )

    metrics.update({

        "fold":
            fold_number,

        "n_unique_scenes":
            scene_groups.iloc[
                val_idx
            ].nunique(),

        "target_mean":
            val_values.mean(),

        "target_std":
            val_values.std(),

        "target_max":
            val_values.max(),
    })

    scene_fold_rows.append(
        metrics
    )


scene_fold_df = pd.DataFrame(
    scene_fold_rows
)


scene_pooled = calc_metrics(
    y_all,
    scene_oof
)


print(
    "\nFold results:"
)

print(
    scene_fold_df[
        [
            "fold",
            "n",
            "n_unique_scenes",
            "target_mean",
            "target_std",
            "target_max",
            "r2",
            "rmse_fnu",
            "mae_fnu",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print(
    "\nMean fold R²:",
    round(
        scene_fold_df[
            "r2"
        ].mean(),
        4
    )
)

print(
    "Fold R² SD:",
    round(
        scene_fold_df[
            "r2"
        ].std(ddof=0),
        4
    )
)

print(
    "\nPOOLED out-of-fold metrics:"
)

for key, value in scene_pooled.items():

    if key == "n":
        print(
            f"{key}: {value}"
        )

    else:
        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# C. LEAVE-ONE-STATION-OUT
#
# This is a DIFFERENT question:
# can the model generalize to a station it has never seen?
# ============================================================

print("\n" + "=" * 90)
print("C. LEAVE-ONE-STATION-OUT")
print("=" * 90)


station_groups = (
    df["site_no"]
    .astype(str)
    .reset_index(drop=True)
)

station_names = (
    df["station_nm"]
    .astype(str)
    .reset_index(drop=True)
)


logo = LeaveOneGroupOut()


station_rows = []

station_oof = np.full(
    len(df),
    np.nan
)


for train_idx, val_idx in logo.split(
    X_all,
    y_all,
    groups=station_groups
):

    held_site = (
        station_groups
        .iloc[val_idx]
        .iloc[0]
    )

    held_name = (
        station_names
        .iloc[val_idx]
        .iloc[0]
    )

    model = make_xgb()

    model.fit(
        X_all.iloc[train_idx],
        y_all.iloc[train_idx]
    )

    pred = model.predict(
        X_all.iloc[val_idx]
    )

    station_oof[
        val_idx
    ] = pred

    metrics = calc_metrics(
        y_all.iloc[val_idx],
        pred
    )

    values = (
        y_all.iloc[val_idx]
    )

    metrics.update({

        "site_no":
            held_site,

        "station_nm":
            held_name,

        "target_mean":
            values.mean(),

        "target_std":
            values.std(),

        "target_max":
            values.max(),
    })

    station_rows.append(
        metrics
    )


station_df = pd.DataFrame(
    station_rows
)


station_pooled = calc_metrics(
    y_all,
    station_oof
)


print(
    station_df[
        [
            "site_no",
            "station_nm",
            "n",
            "target_mean",
            "target_std",
            "target_max",
            "r2",
            "rmse_fnu",
            "mae_fnu",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print(
    "\nMean station R²:",
    round(
        station_df[
            "r2"
        ].mean(),
        4
    )
)

print(
    "\nPOOLED LOSO metrics:"
)

for key, value in station_pooled.items():

    if key == "n":
        print(
            f"{key}: {value}"
        )

    else:
        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# D. OPTIONAL MORE CONSERVATIVE CHECK:
# GROUP BY ACQUISITION DATE
#
# Different Sentinel scenes/tiles acquired on the same day
# are kept together.
# ============================================================

date_audit_available = (
    "scene_datetime_UTC"
    in df.columns
)


if date_audit_available:

    dates = pd.to_datetime(
        df["scene_datetime_UTC"],
        errors="coerce",
        utc=True
    )

    if dates.notna().all():

        date_groups = (
            dates.dt.strftime(
                "%Y-%m-%d"
            )
        )

        print(
            "\n"
            + "=" * 90
        )

        print(
            "D. 5-FOLD GROUPKFold BY ACQUISITION DATE"
        )

        print(
            "=" * 90
        )

        date_cv = GroupKFold(
            n_splits=5
        )

        date_oof = np.full(
            len(df),
            np.nan
        )

        date_rows = []

        for fold_number, (
            train_idx,
            val_idx
        ) in enumerate(

            date_cv.split(
                X_all,
                y_all,
                groups=date_groups
            ),

            start=1
        ):

            model = make_xgb()

            model.fit(
                X_all.iloc[
                    train_idx
                ],
                y_all.iloc[
                    train_idx
                ]
            )

            pred = model.predict(
                X_all.iloc[
                    val_idx
                ]
            )

            date_oof[
                val_idx
            ] = pred

            metrics = calc_metrics(
                y_all.iloc[
                    val_idx
                ],
                pred
            )

            metrics[
                "fold"
            ] = fold_number

            metrics[
                "n_dates"
            ] = (
                date_groups
                .iloc[val_idx]
                .nunique()
            )

            date_rows.append(
                metrics
            )


        date_df = pd.DataFrame(
            date_rows
        )

        date_pooled = calc_metrics(
            y_all,
            date_oof
        )


        print(
            date_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

        print(
            "\nMean fold R²:",
            round(
                date_df[
                    "r2"
                ].mean(),
                4
            )
        )

        print(
            "\nPOOLED date-grouped metrics:"
        )

        for key, value in (
            date_pooled.items()
        ):

            if key == "n":
                print(
                    f"{key}: {value}"
                )

            else:
                print(
                    f"{key}: {value:.4f}"
                )

    else:

        date_audit_available = False


# ============================================================
# SUMMARY
# ============================================================

summary_rows = [
    {
        "validation":
            "Original fixed 80/20",

        **original_metrics,
    },

    {
        "validation":
            "5-fold GroupKFold by scene (pooled OOF)",

        **scene_pooled,
    },

    {
        "validation":
            "Leave-one-station-out (pooled OOF)",

        **station_pooled,
    },
]


if (
    date_audit_available
    and "date_pooled" in locals()
):

    summary_rows.append({
        "validation":
            "5-fold GroupKFold by acquisition date (pooled OOF)",

        **date_pooled,
    })


summary_df = pd.DataFrame(
    summary_rows
)


print("\n" + "=" * 90)
print("FINAL VALIDATION AUDIT SUMMARY")
print("=" * 90)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUT_DIR
    / "16_validation_audit_summary.csv",
    index=False
)

scene_fold_df.to_csv(
    OUT_DIR
    / "16_scene_grouped_folds.csv",
    index=False
)

station_df.to_csv(
    OUT_DIR
    / "16_loso_station_results.csv",
    index=False
)


if (
    date_audit_available
    and "date_df" in locals()
):

    date_df.to_csv(
        OUT_DIR
        / "16_date_grouped_folds.csv",
        index=False
    )


oof_df = df[
    [
        "sample_id",
        "site_no",
        "station_nm",
        "scene",
        TARGET,
    ]
].copy()

oof_df[
    "scene_grouped_prediction"
] = scene_oof

oof_df[
    "loso_prediction"
] = station_oof


if (
    date_audit_available
    and "date_oof" in locals()
):

    oof_df[
        "date_grouped_prediction"
    ] = date_oof


oof_df.to_csv(
    OUT_DIR
    / "16_validation_oof_predictions.csv",
    index=False
)


print("\nSaved audit files to:")
print(OUT_DIR)

print(
    "\nNO feature selection, hyperparameter tuning, "
    "or model modification was performed."
)
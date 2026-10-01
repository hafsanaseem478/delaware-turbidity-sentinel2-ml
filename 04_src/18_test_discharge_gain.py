from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from sklearn.model_selection import (
    GroupKFold,
)

from xgboost import XGBRegressor


# ============================================================
# FILES
# ============================================================

INPUT = Path(
    "01_data/processed/delaware_with_discharge.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

XGB_FILE = Path(
    "05_results/metrics/09_xgboost_tuning.json"
)

OUTPUT = Path(
    "05_results/tables/18_discharge_gain.csv"
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

    feature_info = json.load(f)

with open(
    XGB_FILE,
    "r",
    encoding="utf-8"
) as f:

    xgb_info = json.load(f)


SPECTRAL = (
    feature_info[
        "final_original_features"
    ]
)


# Keep a compact hydrology set.
# Avoid giving the model eight redundant Q variables.

HYDRO = [

    "log_Q_lag1",

    "log_Q_pre3",

    "log_Q_pre7",

    "dQ_1d_cfs",

    "Q_rising",
]


TARGET = "MeasurementValue"


# ============================================================
# SAME OBSERVATIONS FOR ALL THREE EXPERIMENTS
# ============================================================

required = (
    SPECTRAL
    + HYDRO
    + [
        TARGET,
        "scene_datetime_UTC",
    ]
)


model_df = (
    df
    .dropna(
        subset=required
    )
    .copy()
    .reset_index(drop=True)
)


model_df[
    "scene_datetime_UTC"
] = pd.to_datetime(
    model_df[
        "scene_datetime_UTC"
    ],
    utc=True,
    errors="raise"
)


model_df[
    "acquisition_date"
] = (

    model_df[
        "scene_datetime_UTC"
    ]

    .dt.strftime(
        "%Y-%m-%d"
    )
)


print("=" * 90)
print("DISCHARGE PERFORMANCE TEST")
print("=" * 90)

print(
    "\nRows with complete discharge:",
    len(model_df)
)

print(
    "Stations:",
    model_df["site_no"].nunique()
)

print("\nRows by station:")

print(
    model_df[
        "station_nm"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# FROZEN CURRENT XGBOOST SETTINGS
#
# Keep them fixed FIRST so any gain comes from discharge,
# not from extra tuning.
# ============================================================

best = xgb_info[
    "best_parameters"
]


def make_model():

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

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=1.0,

        random_state=42,

        n_jobs=-1,
    )


# ============================================================
# THREE EXPERIMENTS
# ============================================================

FEATURE_SETS = {

    "spectral_only":
        SPECTRAL,

    "discharge_only":
        HYDRO,

    "spectral_plus_discharge":
        SPECTRAL + HYDRO,
}


# ============================================================
# 5-FOLD ACQUISITION-DATE GROUPED CV
#
# Same rows and same folds for all experiments.
# ============================================================

cv = GroupKFold(
    n_splits=5
)

groups = (
    model_df[
        "acquisition_date"
    ]
)

y = (
    model_df[TARGET]
    .astype(float)
    .values
)


results = []


for experiment, features in (
    FEATURE_SETS.items()
):

    print("\n" + "=" * 90)

    print(
        experiment.upper()
    )

    print("=" * 90)


    oof = np.full(
        len(model_df),
        np.nan
    )

    fold_scores = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(

        cv.split(
            model_df,
            y,
            groups=groups
        ),

        start=1
    ):

        model = make_model()

        model.fit(

            model_df.iloc[
                train_idx
            ][features],

            y[
                train_idx
            ],
        )


        pred = model.predict(

            model_df.iloc[
                val_idx
            ][features]
        )


        oof[
            val_idx
        ] = pred


        fold_r2 = r2_score(
            y[val_idx],
            pred
        )

        fold_scores.append(
            fold_r2
        )


        print(
            f"Fold {fold} R²: "
            f"{fold_r2:.4f}"
        )


    # --------------------------------------------------------
    # POOLED OOF METRICS
    # --------------------------------------------------------

    pooled_r2 = r2_score(
        y,
        oof
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            oof
        )
    )

    mae = mean_absolute_error(
        y,
        oof
    )

    mbe = np.mean(
        oof - y
    )


    result = {

        "experiment":
            experiment,

        "n":
            len(model_df),

        "n_features":
            len(features),

        "mean_fold_r2":
            np.mean(
                fold_scores
            ),

        "pooled_oof_r2":
            pooled_r2,

        "rmse_fnu":
            rmse,

        "mae_fnu":
            mae,

        "mbe_fnu":
            mbe,
    }


    results.append(
        result
    )


    print(
        "\nPOOLED OOF R²:",
        round(
            pooled_r2,
            4
        )
    )

    print(
        "RMSE:",
        round(
            rmse,
            4
        )
    )

    print(
        "MAE:",
        round(
            mae,
            4
        )
    )


# ============================================================
# RESULTS
# ============================================================

results_df = (
    pd.DataFrame(
        results
    )
    .sort_values(
        "pooled_oof_r2",
        ascending=False
    )
)


print("\n" + "=" * 90)
print("FINAL DISCHARGE COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# GAIN
# ============================================================

scores = (
    results_df
    .set_index(
        "experiment"
    )[
        "pooled_oof_r2"
    ]
)


if (
    "spectral_only"
    in scores.index
    and
    "spectral_plus_discharge"
    in scores.index
):

    gain = (
        scores[
            "spectral_plus_discharge"
        ]
        -
        scores[
            "spectral_only"
        ]
    )

    print(
        "\nR² GAIN FROM DISCHARGE:",
        f"{gain:+.4f}"
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
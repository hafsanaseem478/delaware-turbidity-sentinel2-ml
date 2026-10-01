from pathlib import Path
import json
import math
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from sklearn.model_selection import (
    KFold,
    GridSearchCV,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
)

from xgboost import XGBRegressor


# ============================================================
# PURPOSE
# ============================================================
#
# Illinois XGBoost baseline following Santos et al. (2025)
# as closely as possible.
#
# INPUT:
#   - Random 80/20 train/test split created in script 20b
#   - Santos-style selected predictors
#   - Cube root of each selected predictor
#
# PREPROCESSING OPTIONS:
#   1. None
#   2. Standard scaling
#   3. Yeo-Johnson
#   4. Yeo-Johnson + Standard scaling
#
# MODEL:
#   XGBoost
#
# MODEL SELECTION:
#   GridSearchCV
#   5-fold CV
#   R² scoring
#
# FINAL:
#   Best training-CV configuration is evaluated ONCE
#   on the held-out 20% test set.
#
# IMPORTANT:
#   The Santos paper reports GridSearch with 5 CV folds,
#   but does NOT publish the exact XGBoost search grid.
#
#   Therefore this is a documented, computationally
#   practical grid for our reproduction/extension.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "01_data/processed/illinois_santos_primary.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/20b_illinois_santos_features.json"
)


# Summary of best result from each normalization
CV_SUMMARY_OUTPUT = Path(
    "05_results/tables/22_illinois_santos_xgb_gridsearch.csv"
)


# Full GridSearchCV results from every normalization
CV_FULL_OUTPUT = Path(
    "05_results/tables/22_illinois_santos_xgb_gridsearch_all.csv"
)


METRIC_OUTPUT = Path(
    "05_results/metrics/22_illinois_santos_xgb_test_metrics.json"
)


PRED_OUTPUT = Path(
    "05_results/tables/22_illinois_santos_xgb_test_predictions.csv"
)


MODEL_OUTPUT = Path(
    "05_results/metrics/22_illinois_santos_xgb_model.joblib"
)


CV_SUMMARY_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MeasurementValue"

RANDOM_STATE = 42


# ------------------------------------------------------------
# Parallel jobs
#
# XGBoost itself is kept at n_jobs=1.
# GridSearchCV handles parallelism.
#
# Limiting to 4 avoids Windows/process oversubscription.
# ------------------------------------------------------------

SEARCH_JOBS = min(
    4,
    os.cpu_count() or 1
)


print("=" * 100)
print("ILLINOIS — SANTOS-STYLE XGBOOST")
print("=" * 100)

print(
    "\nGridSearch parallel jobs:",
    SEARCH_JOBS
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Missing dataset:\n{DATA_FILE}"
    )


if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Missing feature metadata:\n{FEATURE_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(
        f
    )


FEATURES = feature_info[
    "final_model_features"
]


print(
    "\nTotal dataset rows:",
    len(df)
)

print(
    "Number of predictors:",
    len(FEATURES)
)


print(
    "\nPredictors:"
)


for i, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = (

    FEATURES

    +

    [
        TARGET,
        "split",
        "sample_id",
        "site_no",
    ]

)


missing_columns = [

    col

    for col in required_columns

    if col not in df.columns

]


if missing_columns:

    raise KeyError(

        "Missing required columns:\n"

        +

        "\n".join(
            missing_columns
        )

    )


# ============================================================
# TRAIN / TEST
# ============================================================

train = (

    df.loc[
        df["split"]
        .astype(str)
        .eq("train")
    ]

    .copy()

)


test = (

    df.loc[
        df["split"]
        .astype(str)
        .eq("test")
    ]

    .copy()

)


print(
    "\nTrain rows:",
    len(train)
)

print(
    "Test rows:",
    len(test)
)


if len(train) == 0:

    raise RuntimeError(
        "No training rows found."
    )


if len(test) == 0:

    raise RuntimeError(
        "No test rows found."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in FEATURES + [TARGET]:

    train[col] = pd.to_numeric(
        train[col],
        errors="coerce"
    )

    test[col] = pd.to_numeric(
        test[col],
        errors="coerce"
    )


# ============================================================
# MISSING / INVALID CHECK
# ============================================================

train_invalid = (

    train[
        FEATURES + [TARGET]
    ]

    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    .isna()

    .sum()

)


test_invalid = (

    test[
        FEATURES + [TARGET]
    ]

    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    .isna()

    .sum()

)


if train_invalid.sum() > 0:

    print(
        "\nInvalid TRAIN values:"
    )

    print(
        train_invalid[
            train_invalid > 0
        ].to_string()
    )

    raise RuntimeError(
        "Invalid values found in training data."
    )


if test_invalid.sum() > 0:

    print(
        "\nInvalid TEST values:"
    )

    print(
        test_invalid[
            test_invalid > 0
        ].to_string()
    )

    raise RuntimeError(
        "Invalid values found in test data."
    )


# ============================================================
# MATRICES
# ============================================================

X_train = (

    train[
        FEATURES
    ]

    .astype(float)

)


y_train = (

    train[
        TARGET
    ]

    .astype(float)

)


X_test = (

    test[
        FEATURES
    ]

    .astype(float)

)


y_test = (

    test[
        TARGET
    ]

    .astype(float)

)


# ============================================================
# SHOW TARGET DISTRIBUTION
# ============================================================

print(
    "\nTRAIN TARGET"
)

print(
    f" Mean   : {y_train.mean():.3f}"
)

print(
    f" SD     : {y_train.std():.3f}"
)

print(
    f" Median : {y_train.median():.3f}"
)

print(
    f" Max    : {y_train.max():.3f}"
)


print(
    "\nTEST TARGET"
)

print(
    f" Mean   : {y_test.mean():.3f}"
)

print(
    f" SD     : {y_test.std():.3f}"
)

print(
    f" Median : {y_test.median():.3f}"
)

print(
    f" Max    : {y_test.max():.3f}"
)


# ============================================================
# FIVE-FOLD CV
# ============================================================
#
# Santos reports 5-fold cross-validation during GridSearch.
#
# Exact randomization details are not reported.
# We use shuffled KFold with a fixed seed for reproducibility.
#
# ============================================================

cv = KFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE,

)


# ============================================================
# XGBOOST BASE MODEL
# ============================================================

def make_xgb():

    return XGBRegressor(

        objective=
            "reg:squarederror",

        random_state=
            RANDOM_STATE,

        n_jobs=
            1,

        verbosity=
            0,

    )


# ============================================================
# PREPROCESSING OPTIONS
# ============================================================
#
# Santos investigated:
#
# 1. No normalization
# 2. Standard normalization
# 3. Yeo-Johnson
# 4. Yeo-Johnson + standard normalization
#
# ============================================================

pipelines = {


    # --------------------------------------------------------
    # 1. NONE
    # --------------------------------------------------------

    "None":

        Pipeline(

            steps=[

                (
                    "model",
                    make_xgb()
                ),

            ]

        ),


    # --------------------------------------------------------
    # 2. STANDARD SCALER
    # --------------------------------------------------------

    "Standard":

        Pipeline(

            steps=[

                (
                    "scale",
                    StandardScaler()
                ),

                (
                    "model",
                    make_xgb()
                ),

            ]

        ),


    # --------------------------------------------------------
    # 3. YEO-JOHNSON
    # --------------------------------------------------------

    "YeoJohnson":

        Pipeline(

            steps=[

                (
                    "yeojohnson",
                    PowerTransformer(

                        method=
                            "yeo-johnson",

                        standardize=
                            False,

                    )
                ),

                (
                    "model",
                    make_xgb()
                ),

            ]

        ),


    # --------------------------------------------------------
    # 4. YEO-JOHNSON + STANDARDIZATION
    # --------------------------------------------------------

    "YeoJohnson_Standard":

        Pipeline(

            steps=[

                (
                    "yeojohnson",
                    PowerTransformer(

                        method=
                            "yeo-johnson",

                        standardize=
                            False,

                    )
                ),

                (
                    "scale",
                    StandardScaler()
                ),

                (
                    "model",
                    make_xgb()
                ),

            ]

        ),

}


# ============================================================
# PRACTICAL GRID
# ============================================================
#
# 3 x 2 x 3 x 2 x 2 x 2
#
# = 144 combinations
#
# × 5 folds
#
# = 720 fits per normalization option
#
# Four preprocessing options:
#
# = 2,880 total model fits
#
# This retains a proper GridSearchCV while avoiding the
# previous unnecessarily large 432-combination grid.
#
# ============================================================

PARAM_GRID = {

    "model__n_estimators": [

        100,
        200,
        400,

    ],


    "model__learning_rate": [

        0.03,
        0.05,

    ],


    "model__max_depth": [

        2,
        3,
        4,

    ],


    "model__min_child_weight": [

        1,
        3,

    ],


    "model__subsample": [

        0.8,
        1.0,

    ],


    "model__colsample_bytree": [

        0.8,
        1.0,

    ],


    # Fixed because this parameter was not central to our
    # previous tuning results and doubles the search size.

    "model__reg_alpha": [

        0.0,

    ],


    "model__reg_lambda": [

        1.0,

    ],

}


# ============================================================
# CHECK NUMBER OF COMBINATIONS
# ============================================================

n_combinations = 1


for values in PARAM_GRID.values():

    n_combinations *= len(
        values
    )


print(
    "\nGrid combinations per preprocessing option:",
    n_combinations
)


print(
    "Fits per preprocessing option:",
    n_combinations * 5
)


print(
    "Approximate total fits:",
    n_combinations
    * 5
    * len(pipelines)
)


# ============================================================
# STORAGE
# ============================================================

summary_rows = []

all_grid_results = []


best_search = None

best_name = None

best_cv = -np.inf


# ============================================================
# GRID SEARCH
# ============================================================

for normalization_name, pipeline in pipelines.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NORMALIZATION:",
        normalization_name
    )

    print(
        "=" * 100
    )


    start_time = time.time()


    search = GridSearchCV(

        estimator=
            pipeline,

        param_grid=
            PARAM_GRID,

        scoring=
            "r2",

        cv=
            cv,

        refit=
            True,

        n_jobs=
            SEARCH_JOBS,

        verbose=
            1,

        return_train_score=
            False,

        error_score=
            "raise",

    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        search.fit(
            X_train,
            y_train
        )


    elapsed_minutes = (

        time.time()
        -
        start_time

    ) / 60.0


    # ========================================================
    # BEST RESULT FOR THIS NORMALIZATION
    # ========================================================

    print(
        "\nCompleted in:",
        f"{elapsed_minutes:.2f} minutes"
    )


    print(
        "\nBest CV R²:",
        f"{search.best_score_:.6f}"
    )


    print(
        "\nBest parameters:"
    )


    for key, value in search.best_params_.items():

        print(
            f" {key}: {value}"
        )


    # ========================================================
    # SUMMARY ROW
    # ========================================================

    row = {

        "normalization":
            normalization_name,

        "best_cv_r2":
            float(
                search.best_score_
            ),

        "runtime_minutes":
            float(
                elapsed_minutes
            ),

    }


    for key, value in search.best_params_.items():

        clean_key = key.replace(
            "model__",
            ""
        )


        row[
            clean_key
        ] = value


    summary_rows.append(
        row
    )


    # ========================================================
    # SAVE SUMMARY IMMEDIATELY
    #
    # So if a later normalization is interrupted, completed
    # runs are not lost.
    # ========================================================

    current_summary = (

        pd.DataFrame(
            summary_rows
        )

        .sort_values(
            "best_cv_r2",
            ascending=False
        )

    )


    current_summary.to_csv(
        CV_SUMMARY_OUTPUT,
        index=False
    )


    # ========================================================
    # SAVE FULL GRID RESULTS
    # ========================================================

    temp_results = pd.DataFrame(
        search.cv_results_
    )


    temp_results.insert(
        0,
        "normalization",
        normalization_name
    )


    all_grid_results.append(
        temp_results
    )


    pd.concat(

        all_grid_results,

        ignore_index=True

    ).to_csv(

        CV_FULL_OUTPUT,

        index=False

    )


    # ========================================================
    # TRACK OVERALL BEST USING TRAINING CV ONLY
    # ========================================================

    if (
        search.best_score_
        >
        best_cv
    ):

        best_cv = float(
            search.best_score_
        )

        best_search = search

        best_name = normalization_name


# ============================================================
# FINAL CV SUMMARY
# ============================================================

summary = (

    pd.DataFrame(
        summary_rows
    )

    .sort_values(
        "best_cv_r2",
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


summary.to_csv(
    CV_SUMMARY_OUTPUT,
    index=False
)


print(
    "\n"
    + "=" * 100
)

print(
    "NORMALIZATION COMPARISON"
)

print(
    "=" * 100
)


print(

    summary[
        [
            "normalization",
            "best_cv_r2",
            "runtime_minutes",
        ]
    ]

    .round(
        6
    )

    .to_string(
        index=False
    )

)


# ============================================================
# BEST TRAINING-CV CONFIGURATION
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "BEST TRAINING-CV CONFIGURATION"
)

print(
    "=" * 100
)


print(
    "\nNormalization:",
    best_name
)


print(
    "Mean CV R²:",
    f"{best_cv:.6f}"
)


print(
    "\nBest parameters:"
)


for key, value in best_search.best_params_.items():

    print(
        f" {key}: {value}"
    )


# ============================================================
# FINAL MODEL
# ============================================================
#
# GridSearchCV already refitted the winning configuration on
# the entire 80% training dataset.
#
# Test data have not been used up to this point.
#
# ============================================================

best_model = (
    best_search.best_estimator_
)


# ============================================================
# OPEN HELD-OUT TEST SET ONCE
# ============================================================

print(
    "\nEvaluating held-out 20% test set..."
)


pred = best_model.predict(
    X_test
)


# ============================================================
# SANTOS METRICS
# ============================================================

r2 = r2_score(
    y_test,
    pred
)


rmse = math.sqrt(

    mean_squared_error(
        y_test,
        pred
    )

)


mae = mean_absolute_error(
    y_test,
    pred
)


mean_observed = float(
    y_test.mean()
)


if mean_observed != 0:

    nrmse = (

        rmse
        /
        mean_observed

    )


else:

    nrmse = np.nan


# ------------------------------------------------------------
# MBE convention used in Santos:
#
# predicted - observed
#
# Positive = overprediction
# Negative = underprediction
# ------------------------------------------------------------

mbe = float(

    np.mean(

        pred

        -

        y_test.to_numpy()

    )

)


# ============================================================
# HELD-OUT RESULT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "HELD-OUT 20% TEST RESULT"
)

print(
    "=" * 100
)


print(
    f"\nR²    = {r2:.4f}"
)

print(
    f"RMSE  = {rmse:.4f} FNU"
)

print(
    f"nRMSE = {nrmse:.4f}"
)

print(
    f"MAE   = {mae:.4f} FNU"
)

print(
    f"MBE   = {mbe:.4f} FNU"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_data = {

    "sample_id":
        test[
            "sample_id"
        ]
        .values,

    "site_no":
        test[
            "site_no"
        ]
        .values,

    "observed_fnu":
        y_test.values,

    "predicted_fnu":
        pred,

    "residual_pred_minus_obs":
        (
            pred
            -
            y_test.values
        ),

    "absolute_error":
        np.abs(
            pred
            -
            y_test.values
        ),

}


# Include station name if available

if "station_nm" in test.columns:

    prediction_data[
        "station_nm"
    ] = (

        test[
            "station_nm"
        ]
        .values

    )


# Include scene if available

if "scene" in test.columns:

    prediction_data[
        "scene"
    ] = (

        test[
            "scene"
        ]
        .values

    )


pred_df = pd.DataFrame(
    prediction_data
)


pred_df.to_csv(
    PRED_OUTPUT,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "basin":
        "Illinois",

    "methodology":
        "Santos-style primary experiment",

    "model":
        "XGBoost",

    "normalization":
        best_name,

    "training_cv_r2":
        float(
            best_cv
        ),

    "best_parameters":
        best_search.best_params_,

    "random_state":
        RANDOM_STATE,

    "cv":
        (
            "5-fold shuffled KFold "
            "within 80% training dataset"
        ),

    "n_grid_combinations_per_normalization":
        int(
            n_combinations
        ),

    "n_train":
        int(
            len(train)
        ),

    "n_test":
        int(
            len(test)
        ),

    "n_predictors":
        int(
            len(FEATURES)
        ),

    "predictors":
        FEATURES,

    "test_r2":
        float(
            r2
        ),

    "test_rmse_fnu":
        float(
            rmse
        ),

    "test_nrmse":
        float(
            nrmse
        ),

    "test_mae_fnu":
        float(
            mae
        ),

    "test_mbe_fnu":
        float(
            mbe
        ),

}


with open(
    METRIC_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_OUTPUT
)


# ============================================================
# FINAL FILE REPORT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)


print(
    CV_SUMMARY_OUTPUT
)

print(
    CV_FULL_OUTPUT
)

print(
    METRIC_OUTPUT
)

print(
    PRED_OUTPUT
)

print(
    MODEL_OUTPUT
)


print(
    "\nIllinois Santos-style XGBoost baseline complete."
)
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    cross_val_score,
)

from xgboost import XGBRegressor

INPUT_FILE = Path(
    "01_data/processed/delaware_santos_primary.csv"
)

FEATURE_DATA_OUTPUT = Path(
    "01_data/processed/delaware_santos_features.csv"
)

INDEX_ALL_OUTPUT = Path(
    "05_results/tables/31_delaware_santos_index_selection_all.csv"
)

INDEX_PATH_OUTPUT = Path(
    "05_results/tables/31_delaware_santos_index_path.csv"
)

BAND_ALL_OUTPUT = Path(
    "05_results/tables/31_delaware_santos_band_selection_all.csv"
)

BAND_PATH_OUTPUT = Path(
    "05_results/tables/31_delaware_santos_band_path.csv"
)

METADATA_OUTPUT = Path(
    "05_results/metrics/31_delaware_santos_features.json"
)

INDEX_ALL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "MeasurementValue"

RANDOM_STATE = 42

CV_JOBS = 4

INDEX_CANDIDATES = [

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

BAND_CANDIDATES = [

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

print("=" * 100)

print(
    "DELAWARE â€” SANTOS-STYLE FEATURE SELECTION"
)

print("=" * 100)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    dtype={
        "site_no": "string"
    }
)

if "split" not in df.columns:

    raise KeyError(
        "The input dataset does not contain the 'split' column."
    )

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
    "\nTotal rows:",
    len(df)
)

print(
    "Training rows:",
    len(train)
)

print(
    "Held-out test rows:",
    len(test)
)

if len(train) != 1121:

    print(
        "\nWARNING:"
        f" expected approximately 1121 training rows, found {len(train)}."
    )

if len(test) != 281:

    print(
        "\nWARNING:"
        f" expected approximately 281 test rows, found {len(test)}."
    )

print(
    "\nIMPORTANT:"
)

print(
    "Feature selection will use TRAINING rows only."
)

print(
    "The held-out TEST set will not be evaluated."
)

required_columns = (

    INDEX_CANDIDATES
    +
    BAND_CANDIDATES
    +
    [TARGET]

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

for col in required_columns:

    train[col] = pd.to_numeric(

        train[col],

        errors="coerce"

    )

train_check = (

    train[
        required_columns
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

)

invalid_counts = (

    train_check
    .isna()
    .sum()

)

if invalid_counts.sum() > 0:

    print(
        "\nInvalid training values:"
    )

    print(

        invalid_counts[
            invalid_counts > 0
        ]

    )

    raise RuntimeError(
        "Training data contain invalid predictor values."
    )

y_train = (

    train[TARGET]

    .astype(float)

)

print(
    "\nTraining target:"
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

cv = KFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE,

)

def make_model():

    return XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=
            200,

        learning_rate=
            0.05,

        max_depth=
            3,

        min_child_weight=
            1,

        subsample=
            0.8,

        colsample_bytree=
            0.8,

        reg_alpha=
            0.0,

        reg_lambda=
            1.0,

        random_state=
            RANDOM_STATE,

        n_jobs=
            1,

        verbosity=
            0,

    )

def evaluate_features(
    features
):

    X = (

        train[
            features
        ]

        .astype(float)

    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        scores = cross_val_score(

            make_model(),

            X,

            y_train,

            scoring=
                "r2",

            cv=
                cv,

            n_jobs=
                CV_JOBS,

        )

    mean_r2 = float(
        np.mean(scores)
    )

    sd_r2 = float(

        np.std(
            scores,
            ddof=1
        )

    )

    se_r2 = float(

        sd_r2
        /
        math.sqrt(
            len(scores)
        )

    )

    return {

        "mean_r2":
            mean_r2,

        "sd_r2":
            sd_r2,

        "se_r2":
            se_r2,

        "fold_scores":
            [
                float(x)
                for x in scores
            ],

    }

def forward_selection(

    base_features,

    candidates,

    stage_name,

):

    selected = list(
        base_features
    )

    remaining = list(
        candidates
    )

    all_rows = []

    path_rows = []

    added_ranking = []

    step = 1

    while len(remaining) > 0:

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"{stage_name} STEP {step}"
        )

        print(
            "-" * 100
        )

        step_results = []

        for candidate in remaining:

            features = (

                selected

                +

                [
                    candidate
                ]

            )

            result = evaluate_features(
                features
            )

            row = {

                "step":
                    step,

                "candidate":
                    candidate,

                "n_features":
                    len(features),

                "mean_cv_r2":
                    result[
                        "mean_r2"
                    ],

                "sd_cv_r2":
                    result[
                        "sd_r2"
                    ],

                "se_cv_r2":
                    result[
                        "se_r2"
                    ],

                "features":
                    "|".join(
                        features
                    ),

            }

            for fold_number, fold_score in enumerate(

                result[
                    "fold_scores"
                ],

                start=1,

            ):

                row[
                    f"fold_{fold_number}_r2"
                ] = fold_score

            all_rows.append(
                row
            )

            step_results.append(
                row
            )

            print(

                f"{candidate:<14} "

                f"RÂ²="
                f"{result['mean_r2']:.4f} "

                f"Â±SE "
                f"{result['se_r2']:.4f}"

            )

        best = max(

            step_results,

            key=lambda row:
                row[
                    "mean_cv_r2"
                ]

        )

        best_candidate = best[
            "candidate"
        ]

        selected.append(
            best_candidate
        )

        added_ranking.append(
            best_candidate
        )

        remaining.remove(
            best_candidate
        )

        path_row = {

            "step":
                step,

            "selected_this_step":
                best_candidate,

            "combination_mean_cv_r2":
                best[
                    "mean_cv_r2"
                ],

            "combination_sd_cv_r2":
                best[
                    "sd_cv_r2"
                ],

            "combination_se_cv_r2":
                best[
                    "se_cv_r2"
                ],

            "selected_features":
                "|".join(
                    selected
                ),

        }

        path_rows.append(
            path_row
        )

        print(
            "\nSELECTED THIS STEP:",
            best_candidate
        )

        print(
            "Combination RÂ²:",
            f"{best['mean_cv_r2']:.6f}"
        )

        step += 1

    return (

        pd.DataFrame(
            all_rows
        ),

        pd.DataFrame(
            path_rows
        ),

        added_ranking,

    )

print(
    "\n"
    + "=" * 100
)

print(
    "STAGE 1 â€” FORWARD SELECTION OF SANTOS INDICES"
)

print(
    "=" * 100
)

index_all, index_path, index_ranking = (

    forward_selection(

        base_features=[],

        candidates=
            INDEX_CANDIDATES,

        stage_name=
            "INDICES",

    )

)

best_index_row_index = (

    index_path[
        "combination_mean_cv_r2"
    ]

    .idxmax()

)

best_index_row = (

    index_path
    .loc[
        best_index_row_index
    ]

)

best_index_step = int(

    best_index_row[
        "step"
    ]

)

selected_indices = (

    index_ranking[
        :best_index_step
    ]

)

best_index_r2 = float(

    best_index_row[
        "combination_mean_cv_r2"
    ]

)

print(
    "\n"
    + "=" * 100
)

print(
    "BEST INDEX COMBINATION"
)

print(
    "=" * 100
)

print(
    "\nForward ranking:"
)

print(
    index_ranking
)

print(
    "\nSelected indices:"
)

for feature in selected_indices:

    print(
        " -",
        feature
    )

print(
    "\nBest index CV RÂ²:",
    f"{best_index_r2:.6f}"
)

print(
    "Best index step:",
    best_index_step
)

index_all.to_csv(

    INDEX_ALL_OUTPUT,

    index=False

)

index_path.to_csv(

    INDEX_PATH_OUTPUT,

    index=False

)

print(
    "\n"
    + "=" * 100
)

print(
    "STAGE 2 â€” FORWARD SELECTION OF SENTINEL-2 BANDS"
)

print(
    "=" * 100
)

band_all, band_path, band_ranking = (

    forward_selection(

        base_features=
            selected_indices,

        candidates=
            BAND_CANDIDATES,

        stage_name=
            "BANDS",

    )

)

best_band_row_index = (

    band_path[
        "combination_mean_cv_r2"
    ]

    .idxmax()

)

best_band_row = (

    band_path
    .loc[
        best_band_row_index
    ]

)

best_band_step = int(

    best_band_row[
        "step"
    ]

)

selected_bands = (

    band_ranking[
        :best_band_step
    ]

)

best_final_r2 = float(

    best_band_row[
        "combination_mean_cv_r2"
    ]

)

original_features = (

    selected_indices

    +

    selected_bands

)

print(
    "\n"
    + "=" * 100
)

print(
    "SANTOS-STYLE SELECTED FEATURES"
)

print(
    "=" * 100
)

print(
    "\nSelected indices:"
)

for feature in selected_indices:

    print(
        " -",
        feature
    )

print(
    "\nSelected bands:"
)

for feature in selected_bands:

    print(
        " -",
        feature
    )

print(
    "\nOriginal predictors:",
    len(
        original_features
    )
)

print(
    "Best final CV RÂ²:",
    f"{best_final_r2:.6f}"
)

cube_root_features = []

for feature in original_features:

    new_feature = (

        f"{feature}_cbrt"

    )

    numeric_values = pd.to_numeric(

        df[feature],

        errors="coerce"

    )

    df[
        new_feature
    ] = np.cbrt(
        numeric_values
    )

    cube_root_features.append(
        new_feature
    )

final_model_features = (

    original_features

    +

    cube_root_features

)

print(
    "\nCube-root predictors added:",
    len(
        cube_root_features
    )
)

print(
    "TOTAL model predictors:",
    len(
        final_model_features
    )
)

final_check = (

    df[
        final_model_features
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

)

final_invalid = (

    final_check
    .isna()
    .sum()

)

if final_invalid.sum() > 0:

    print(
        "\nInvalid values in final predictors:"
    )

    print(

        final_invalid[
            final_invalid > 0
        ]

    )

    raise RuntimeError(
        "Invalid values found in final model predictors."
    )

band_all.to_csv(

    BAND_ALL_OUTPUT,

    index=False

)

band_path.to_csv(

    BAND_PATH_OUTPUT,

    index=False

)

df.to_csv(

    FEATURE_DATA_OUTPUT,

    index=False

)

metadata = {

    "study_area":
        "Delaware River Basin",

    "method":
        (
            "Santos-style forward stepwise "
            "feature selection"
        ),

    "feature_selection_data":
        (
            "80% training data only"
        ),

    "heldout_test_used":
        False,

    "cv_method":
        (
            "5-fold shuffled KFold"
        ),

    "random_state":
        RANDOM_STATE,

    "benchmark_model":
        "XGBoost",

    "benchmark_parameters": {

        "n_estimators":
            200,

        "learning_rate":
            0.05,

        "max_depth":
            3,

        "min_child_weight":
            1,

        "subsample":
            0.8,

        "colsample_bytree":
            0.8,

        "reg_alpha":
            0.0,

        "reg_lambda":
            1.0,

    },

    "unavailable_index":
        "GCI",

    "unavailable_index_reason":
        (
            "Requires Sentinel-2 B09, "
            "which is unavailable."
        ),

    "available_indices":
        INDEX_CANDIDATES,

    "index_forward_ranking":
        index_ranking,

    "selected_indices":
        selected_indices,

    "best_index_step":
        best_index_step,

    "best_index_cv_r2":
        best_index_r2,

    "available_bands":
        BAND_CANDIDATES,

    "unavailable_bands": [

        "B09",
        "B10",

    ],

    "band_forward_ranking":
        band_ranking,

    "selected_bands":
        selected_bands,

    "best_band_step":
        best_band_step,

    "best_final_cv_r2":
        best_final_r2,

    "original_features":
        original_features,

    "cube_root_features":
        cube_root_features,

    "final_model_features":
        final_model_features,

    "n_original_features":
        len(
            original_features
        ),

    "n_final_model_features":
        len(
            final_model_features
        ),

}

with open(

    METADATA_OUTPUT,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metadata,

        file,

        indent=4

    )

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
    FEATURE_DATA_OUTPUT
)

print(
    INDEX_ALL_OUTPUT
)

print(
    INDEX_PATH_OUTPUT
)

print(
    BAND_ALL_OUTPUT
)

print(
    BAND_PATH_OUTPUT
)

print(
    METADATA_OUTPUT
)

print(
    "\nFeature selection complete."
)

print(
    "Held-out TEST set has NOT been evaluated."
)

print(
    "Next step: freeze these predictors and run the Santos model comparison."
)


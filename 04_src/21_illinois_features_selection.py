from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    PredefinedSplit,
    cross_val_score,
)

from xgboost import XGBRegressor


# ============================================================
# PURPOSE
# ============================================================
#
# Illinois spectral feature construction + forward selection.
#
# IMPORTANT:
#
# - Uses SAME Santos index definitions as Delaware.
# - Uses SAME XGBoost selection model as Delaware.
# - Uses ONLY Illinois TRAINING observations.
# - Uses the already locked acquisition-date CV folds.
# - Does NOT use the held-out Illinois test set.
#
# ============================================================


# ============================================================
# FILES
# ============================================================

INPUT = Path(
    "01_data/processed/illinois_train_test_split.csv"
)

FEATURE_OUTPUT = Path(
    "01_data/processed/illinois_santos_features.csv"
)

INDEX_LOG_OUTPUT = Path(
    "05_results/tables/21_illinois_index_selection.csv"
)

BAND_LOG_OUTPUT = Path(
    "05_results/tables/21_illinois_band_selection.csv"
)

FINAL_FEATURE_OUTPUT = Path(
    "05_results/metrics/21_illinois_selected_features.json"
)


FEATURE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_LOG_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_FEATURE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MeasurementValue"

RANDOM_STATE = 42

IMPROVEMENT_TOLERANCE = 1e-6


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={
        "site_no": "string"
    }
)


print("=" * 100)
print("ILLINOIS SANTOS FEATURES + FORWARD SELECTION")
print("=" * 100)


print(
    "\nInput rows:",
    len(df)
)


print(
    "Train rows:",
    int(
        (
            df["split"]
            == "train"
        ).sum()
    )
)


print(
    "Test rows:",
    int(
        (
            df["split"]
            == "test"
        ).sum()
    )
)


# ============================================================
# VERIFY LOCKED SPLIT
# ============================================================

required_columns = [

    TARGET,

    "split",

    "cv_fold",

    "sample_id",

]


for col in required_columns:

    if col not in df.columns:

        raise KeyError(
            f"Missing required column: {col}"
        )


train_original = df.loc[
    df["split"] == "train"
].copy()


test_original = df.loc[
    df["split"] == "test"
].copy()


if (
    train_original["cv_fold"]
    .lt(0)
    .any()
):

    raise RuntimeError(
        "Training observations contain cv_fold < 0."
    )


if (
    test_original["cv_fold"]
    .ne(-1)
    .any()
):

    raise RuntimeError(
        "Test observations should have cv_fold = -1."
    )


print(
    "\nLocked split integrity: PASSED"
)


# ============================================================
# REQUIRED REFLECTANCE BANDS
# ============================================================

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


for col in BAND_FEATURES:

    if col not in df.columns:

        raise KeyError(
            f"Missing reflectance band: {col}"
        )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in BAND_FEATURES + [TARGET]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# SHORT BAND NAMES
# ============================================================

B1 = df["B01_rhow"]
B2 = df["B02_rhow"]
B3 = df["B03_rhow"]
B4 = df["B04_rhow"]
B5 = df["B05_rhow"]
B6 = df["B06_rhow"]
B7 = df["B07_rhow"]
B8 = df["B08_rhow"]
B8A = df["B8A_rhow"]
B11 = df["B11_rhow"]
B12 = df["B12_rhow"]


# ============================================================
# SAFE DIVISION
#
# Same numerical protection used for Delaware.
# ============================================================

EPS = 1e-12


def safe_divide(
    numerator,
    denominator
):

    numerator = np.asarray(
        numerator,
        dtype=float
    )

    denominator = np.asarray(
        denominator,
        dtype=float
    )


    result = np.full(
        numerator.shape,
        np.nan,
        dtype=float
    )


    valid = (

        np.isfinite(
            numerator
        )

        &

        np.isfinite(
            denominator
        )

        &

        (
            np.abs(
                denominator
            )
            > EPS
        )

    )


    result[
        valid
    ] = (

        numerator[
            valid
        ]

        /

        denominator[
            valid
        ]

    )


    return result


# ============================================================
# BUILD THE 16 AVAILABLE SANTOS INDICES
#
# Santos evaluated 17 indices.
#
# GCI cannot be recreated because:
#
#     GCI = B09 / B03 - 1
#
# and the USGS ACOLITE product does not contain B09.
#
# ============================================================


# ------------------------------------------------------------
# 1. NDVI
# ------------------------------------------------------------

df["NDVI"] = safe_divide(

    B8 - B4,

    B8 + B4

)


# ------------------------------------------------------------
# 2. GNDVI
# ------------------------------------------------------------

df["GNDVI"] = safe_divide(

    B8 - B3,

    B8 + B3

)


# ------------------------------------------------------------
# 3. EVI
# ------------------------------------------------------------

df["EVI"] = safe_divide(

    2.5 * (
        B8 - B4
    ),

    (
        B8
        + 6.0 * B4
        - 7.5 * B2
        + 1.0
    )

)


# ------------------------------------------------------------
# 4. SAVI
#
# Santos:
#
# 1.428 * (B8-B4) / (B8+B4+0.428)
# ------------------------------------------------------------

df["SAVI"] = safe_divide(

    1.428 * (
        B8 - B4
    ),

    (
        B8
        + B4
        + 0.428
    )

)


# ------------------------------------------------------------
# 5. NDMI
# ------------------------------------------------------------

df["NDMI"] = safe_divide(

    B8 - B11,

    B8 + B11

)


# ------------------------------------------------------------
# 6. MSI
# ------------------------------------------------------------

df["MSI"] = safe_divide(

    B11,

    B8

)


# ------------------------------------------------------------
# 7. NBRI
# ------------------------------------------------------------

df["NBRI"] = safe_divide(

    B8 - B12,

    B8 + B12

)


# ------------------------------------------------------------
# 8. BSI
# ------------------------------------------------------------

df["BSI"] = safe_divide(

    (
        B11 + B4
    )
    -
    (
        B8 + B2
    ),

    (
        B11 + B4
    )
    +
    (
        B8 + B2
    )

)


# ------------------------------------------------------------
# 9. NDWI
# ------------------------------------------------------------

df["NDWI"] = safe_divide(

    B3 - B8,

    B3 + B8

)


# ------------------------------------------------------------
# 10. NDSI
# ------------------------------------------------------------

df["NDSI"] = safe_divide(

    B3 - B11,

    B3 + B11

)


# ------------------------------------------------------------
# 11. NDGI
# ------------------------------------------------------------

df["NDGI"] = safe_divide(

    B3 - B4,

    B3 + B4

)


# ------------------------------------------------------------
# 12. ARVI
#
# Implemented exactly as the Santos formulation used in
# our Delaware pipeline:
#
# (B8 - 2*B4 + B2) / (B8 + 2*B4)
# ------------------------------------------------------------

df["ARVI"] = safe_divide(

    (
        B8
        - 2.0 * B4
        + B2
    ),

    (
        B8
        + 2.0 * B4
    )

)


# ------------------------------------------------------------
# 13. SIPI
# ------------------------------------------------------------

df["SIPI"] = safe_divide(

    B8 - B2,

    B8 - B4

)


# ------------------------------------------------------------
# 14. SWM
# ------------------------------------------------------------

df["SWM"] = safe_divide(

    B2 + B3,

    B8 + B11

)


# ------------------------------------------------------------
# 15. AWEI
# ------------------------------------------------------------

df["AWEI"] = (

    4.0
    *
    (
        B3 - B11
    )

    -

    (
        0.25 * B8
        +
        2.75 * B12
    )

)


# ------------------------------------------------------------
# 16. 3BSI
#
# (1/B4 - 1/B5) * B6
# ------------------------------------------------------------

inverse_B4 = safe_divide(

    np.ones(
        len(df)
    ),

    B4

)


inverse_B5 = safe_divide(

    np.ones(
        len(df)
    ),

    B5

)


df["3BSI"] = (

    (
        inverse_B4
        -
        inverse_B5
    )

    * B6

)


# ============================================================
# INDEX LIST
# ============================================================

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


print(
    "\nSantos indices generated:",
    len(
        INDEX_FEATURES
    )
)


# ============================================================
# REPLACE INF
# ============================================================

df[
    INDEX_FEATURES
] = (

    df[
        INDEX_FEATURES
    ]

    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

)


# ============================================================
# INVALID FEATURE REPORT
# ============================================================

print(
    "\nMissing / invalid values by index:"
)


invalid_counts = (

    df[
        INDEX_FEATURES
    ]

    .isna()

    .sum()

)


print(
    invalid_counts.to_string()
)


# ============================================================
# USE SAME ROWS FOR ALL FEATURE CANDIDATES
#
# This avoids a different sample size for different indices.
# ============================================================

before_complete_case = len(
    df
)


complete_feature_mask = (

    df[
        INDEX_FEATURES
        +
        BAND_FEATURES
        +
        [TARGET]
    ]

    .notna()

    .all(
        axis=1
    )

)


df = (

    df.loc[
        complete_feature_mask
    ]

    .copy()

    .reset_index(
        drop=True
    )

)


print(
    "\nRows before complete-feature QC:",
    before_complete_case
)

print(
    "Rows after complete-feature QC:",
    len(df)
)

print(
    "Rows removed:",
    (
        before_complete_case
        -
        len(df)
    )
)


print(
    "\nRemaining train rows:",
    int(
        (
            df["split"]
            == "train"
        ).sum()
    )
)

print(
    "Remaining test rows:",
    int(
        (
            df["split"]
            == "test"
        ).sum()
    )
)


# ============================================================
# SAVE FEATURE DATASET
#
# Formula-based feature generation does not learn anything
# from the test set, so it is fine to calculate the indices
# for all rows.
# ============================================================

df.to_csv(
    FEATURE_OUTPUT,
    index=False
)


# ============================================================
# TRAINING DATA ONLY FROM THIS POINT
# ============================================================

train = (

    df.loc[
        df["split"]
        == "train"
    ]

    .copy()

    .reset_index(
        drop=True
    )

)


if len(
    train
) == 0:

    raise RuntimeError(
        "No Illinois training observations remain."
    )


# ============================================================
# PREDEFINED DATE-GROUPED CV
#
# These folds were created BEFORE feature selection.
# ============================================================

fold_values = (

    train[
        "cv_fold"
    ]

    .astype(int)

    .to_numpy()

)


unique_folds = sorted(

    np.unique(
        fold_values
    )

)


print(
    "\nTraining CV folds:",
    unique_folds
)


if unique_folds != [
    0,
    1,
    2,
    3,
    4
]:

    raise RuntimeError(
        f"Unexpected CV folds: {unique_folds}"
    )


cv = PredefinedSplit(

    test_fold=fold_values

)


# ============================================================
# FIXED XGBOOST FEATURE-SELECTION MODEL
#
# IDENTICAL MODEL FAMILY / SETTINGS TO DELAWARE SELECTION.
#
# No tuning here.
# ============================================================

def make_selection_model():

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


# ============================================================
# CV SCORING FUNCTION
# ============================================================

y_train = (

    train[
        TARGET
    ]

    .astype(float)

    .to_numpy()

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


    model = (
        make_selection_model()
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        scores = cross_val_score(

            model,

            X,

            y_train,

            cv=cv,

            scoring="r2",

            n_jobs=-1,

            error_score="raise",

        )


    return (
        float(
            np.mean(
                scores
            )
        ),

        float(
            np.std(
                scores,
                ddof=1
            )
        ),

        [
            float(x)
            for x in scores
        ],

    )


# ============================================================
# GENERIC FORWARD SELECTION
# ============================================================

def forward_select(
    fixed_features,
    candidates,
    stage_name
):

    selected = []

    remaining = list(
        candidates
    )

    detailed_rows = []

    accepted_rows = []

    current_score = (
        -np.inf
    )

    step = 1


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"FORWARD SELECTION: {stage_name}"
    )

    print(
        "=" * 100
    )


    while remaining:

        candidate_results = []


        print(
            f"\nSTEP {step}"
        )


        for candidate in remaining:

            features = (

                list(
                    fixed_features
                )

                +

                list(
                    selected
                )

                +

                [
                    candidate
                ]

            )


            mean_r2, std_r2, fold_scores = (
                evaluate_features(
                    features
                )
            )


            candidate_results.append({

                "candidate":
                    candidate,

                "mean_r2":
                    mean_r2,

                "std_r2":
                    std_r2,

                "fold_scores":
                    fold_scores,

                "features":
                    features,

            })


            detailed_rows.append({

                "stage":
                    stage_name,

                "step":
                    step,

                "candidate":
                    candidate,

                "n_features":
                    len(
                        features
                    ),

                "mean_cv_r2":
                    mean_r2,

                "std_cv_r2":
                    std_r2,

                "fold_0_r2":
                    fold_scores[
                        0
                    ],

                "fold_1_r2":
                    fold_scores[
                        1
                    ],

                "fold_2_r2":
                    fold_scores[
                        2
                    ],

                "fold_3_r2":
                    fold_scores[
                        3
                    ],

                "fold_4_r2":
                    fold_scores[
                        4
                    ],

                "feature_set":
                    "|".join(
                        features
                    ),

            })


            print(

                f"  {candidate:12s} "
                f"R²={mean_r2:.4f} "
                f"± {std_r2:.4f}"

            )


        # ----------------------------------------------------
        # BEST CANDIDATE THIS STEP
        # ----------------------------------------------------

        best = max(

            candidate_results,

            key=lambda x:
                x[
                    "mean_r2"
                ]

        )


        best_candidate = (
            best[
                "candidate"
            ]
        )


        best_score = (
            best[
                "mean_r2"
            ]
        )


        best_std = (
            best[
                "std_r2"
            ]
        )


        best_folds = (
            best[
                "fold_scores"
            ]
        )


        print(
            "\nBest candidate:",
            best_candidate
        )

        print(
            "Best mean CV R²:",
            f"{best_score:.6f}"
        )


        # ----------------------------------------------------
        # FIRST FEATURE IS ALWAYS ACCEPTED
        #
        # Afterwards candidate must improve mean CV R².
        # ----------------------------------------------------

        if (
            len(
                selected
            ) == 0
            or
            best_score
            >
            (
                current_score
                +
                IMPROVEMENT_TOLERANCE
            )
        ):

            previous_score = (
                current_score
            )


            selected.append(
                best_candidate
            )


            remaining.remove(
                best_candidate
            )


            current_score = (
                best_score
            )


            improvement = (

                np.nan

                if not np.isfinite(
                    previous_score
                )

                else
                (
                    best_score
                    -
                    previous_score
                )

            )


            accepted_rows.append({

                "stage":
                    stage_name,

                "step":
                    step,

                "selected_feature":
                    best_candidate,

                "n_features":
                    (
                        len(
                            fixed_features
                        )
                        +
                        len(
                            selected
                        )
                    ),

                "mean_cv_r2":
                    best_score,

                "std_cv_r2":
                    best_std,

                "improvement":
                    improvement,

                "fold_0_r2":
                    best_folds[
                        0
                    ],

                "fold_1_r2":
                    best_folds[
                        1
                    ],

                "fold_2_r2":
                    best_folds[
                        2
                    ],

                "fold_3_r2":
                    best_folds[
                        3
                    ],

                "fold_4_r2":
                    best_folds[
                        4
                    ],

            })


            print(
                "ACCEPTED"
            )


            print(
                "Selected:",
                selected
            )


            step += 1


        else:

            print(
                "\nSTOPPING."
            )

            print(
                "No remaining candidate improves "
                "mean CV R²."
            )

            print(
                "Current score:",
                f"{current_score:.6f}"
            )

            print(
                "Best attempted score:",
                f"{best_score:.6f}"
            )

            break


    return (
        selected,
        current_score,
        detailed_rows,
        accepted_rows,
    )


# ============================================================
# STAGE 1:
# FORWARD SELECT SPECTRAL INDICES
# ============================================================

(
    selected_indices,
    index_score,
    index_details,
    index_path,

) = forward_select(

    fixed_features=[],

    candidates=INDEX_FEATURES,

    stage_name=
        "indices",

)


# ============================================================
# SAVE INDEX SELECTION LOG
# ============================================================

index_details_df = pd.DataFrame(
    index_details
)


index_details_df.to_csv(
    INDEX_LOG_OUTPUT,
    index=False
)


print(
    "\n"
    + "=" * 100
)

print(
    "INDEX SELECTION COMPLETE"
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
    "\nIndex-only mean CV R²:",
    f"{index_score:.6f}"
)


# ============================================================
# STAGE 2:
# ADD RAW SPECTRAL BANDS TO SELECTED INDICES
# ============================================================

(
    selected_bands,
    final_score,
    band_details,
    band_path,

) = forward_select(

    fixed_features=
        selected_indices,

    candidates=
        BAND_FEATURES,

    stage_name=
        "bands",

)


band_details_df = pd.DataFrame(
    band_details
)


band_details_df.to_csv(
    BAND_LOG_OUTPUT,
    index=False
)


# ============================================================
# FINAL FEATURE SET
# ============================================================

FINAL_FEATURES = (

    selected_indices
    +
    selected_bands

)


print(
    "\n"
    + "=" * 100
)

print(
    "FINAL ILLINOIS FEATURE SELECTION"
)

print(
    "=" * 100
)


print(
    "\nSelected indices:",
    selected_indices
)


print(
    "Selected bands:",
    selected_bands
)


print(
    "\nFINAL FEATURES:"
)


for i, feature in enumerate(
    FINAL_FEATURES,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


print(
    "\nIndex-only CV R²:",
    f"{index_score:.6f}"
)


print(
    "Final CV R²:",
    f"{final_score:.6f}"
)


print(
    "Number of final predictors:",
    len(
        FINAL_FEATURES
    )
)


# ============================================================
# FINAL FOLD SCORES
# ============================================================

final_mean, final_std, final_folds = (

    evaluate_features(
        FINAL_FEATURES
    )

)


print(
    "\nFinal fold R² values:"
)


for fold, score in enumerate(
    final_folds
):

    print(
        f" Fold {fold}: "
        f"{score:.4f}"
    )


print(
    "\nMean:",
    f"{final_mean:.4f}"
)


print(
    "SD:",
    f"{final_std:.4f}"
)


# ============================================================
# SAVE JSON
# ============================================================

result = {

    "basin":
        "Illinois",

    "target":
        TARGET,

    "n_total_after_feature_qc":
        int(
            len(
                df
            )
        ),

    "n_train":
        int(
            (
                df[
                    "split"
                ]
                == "train"
            ).sum()
        ),

    "n_test":
        int(
            (
                df[
                    "split"
                ]
                == "test"
            ).sum()
        ),

    "available_santos_indices":
        INDEX_FEATURES,

    "unavailable_santos_indices": [

        "GCI"

    ],

    "gci_reason":
        (
            "B09 is not available in the "
            "USGS ACOLITE matched product."
        ),

    "candidate_bands":
        BAND_FEATURES,

    "selected_indices":
        selected_indices,

    "selected_bands":
        selected_bands,

    "final_original_features":
        FINAL_FEATURES,

    "index_only_mean_cv_r2":
        float(
            index_score
        ),

    "final_mean_cv_r2":
        float(
            final_mean
        ),

    "final_std_cv_r2":
        float(
            final_std
        ),

    "final_fold_r2":
        final_folds,

    "selection_model": {

        "model":
            "XGBRegressor",

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

        "random_state":
            RANDOM_STATE,

    },

    "validation":
        (
            "Locked acquisition-date grouped "
            "PredefinedSplit on Illinois training data only."
        ),

}


with open(
    FINAL_FEATURE_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )


# ============================================================
# SAVE SELECTION PATHS TOO
# ============================================================

INDEX_PATH_OUTPUT = Path(
    "05_results/tables/21_illinois_index_selection_path.csv"
)

BAND_PATH_OUTPUT = Path(
    "05_results/tables/21_illinois_band_selection_path.csv"
)


pd.DataFrame(
    index_path
).to_csv(
    INDEX_PATH_OUTPUT,
    index=False
)


pd.DataFrame(
    band_path
).to_csv(
    BAND_PATH_OUTPUT,
    index=False
)


# ============================================================
# FINISH
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
    FEATURE_OUTPUT
)

print(
    INDEX_LOG_OUTPUT
)

print(
    INDEX_PATH_OUTPUT
)

print(
    BAND_LOG_OUTPUT
)

print(
    BAND_PATH_OUTPUT
)

print(
    FINAL_FEATURE_OUTPUT
)


print(
    "\nThe held-out Illinois test set has NOT been "
    "used for feature selection."
)

print(
    "Next step: tune XGBoost using these selected "
    "features and the same locked training folds."
)
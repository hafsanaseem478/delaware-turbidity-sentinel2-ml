from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
)

from xgboost import XGBRegressor


# ============================================================
# PURPOSE
# ============================================================
#
# Primary Illinois experiment following Santos et al. (2025)
# as closely as possible with the available USGS ACOLITE data.
#
# Workflow:
#
# 1. Illinois model-ready observations
# 2. 16 available Santos indices
#    - GCI unavailable because B09 is absent
# 3. Complete-case feature dataset
# 4. Random 80/20 train-test split
# 5. Forward-stepwise ranking of indices with XGBoost + R²
# 6. Keep BEST index combination
# 7. Forward-stepwise ranking of raw bands
# 8. Keep BEST bands combination
# 9. Add cube root of EVERY selected predictor
#
# Test set is not used during feature selection.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "05_results/tables/00_all_basin_model_ready_rows.csv"
)

OUTPUT_DATA = Path(
    "01_data/processed/illinois_santos_primary.csv"
)

INDEX_PATH_OUTPUT = Path(
    "05_results/tables/20b_illinois_santos_index_path.csv"
)

BAND_PATH_OUTPUT = Path(
    "05_results/tables/20b_illinois_santos_band_path.csv"
)

FEATURE_JSON_OUTPUT = Path(
    "05_results/metrics/20b_illinois_santos_features.json"
)


OUTPUT_DATA.parent.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_PATH_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_JSON_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MeasurementValue"

SITE_COL = "site_no"

SCENE_COL = "scene"

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_FOLDS = 5


# ============================================================
# LOAD SCREENED DATA
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={
        SITE_COL: "string"
    }
)


print("=" * 100)
print("ILLINOIS — SANTOS-STYLE PRIMARY EXPERIMENT")
print("=" * 100)


# ============================================================
# ILLINOIS ONLY
# ============================================================

df = (

    df.loc[
        df["basin"]
        .astype(str)
        .str.strip()
        .eq("Illinois")
    ]

    .copy()

)


print(
    "\nIllinois input rows:",
    len(df)
)


if len(df) == 0:

    raise RuntimeError(
        "No Illinois observations found."
    )


# ============================================================
# REQUIRED RAW BANDS
# ============================================================

BANDS = [

    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",

]


raw_band_cols = {}


for band in BANDS:

    candidates = [

        f"{band}_buf250_med",
        f"{band}_buf250_median",

    ]


    found = None


    for candidate in candidates:

        if candidate in df.columns:

            found = candidate
            break


    if found is None:

        matches = [

            c
            for c in df.columns
            if (
                c.upper().startswith(
                    band.upper()
                )
                and "250" in c
                and "MED" in c.upper()
            )

        ]


        if len(matches) == 1:

            found = matches[0]


    if found is None:

        raise KeyError(
            f"250 m median column missing for {band}"
        )


    raw_band_cols[
        band
    ] = found


print(
    "\nDetected raw bands:"
)


for band, col in raw_band_cols.items():

    print(
        f" {band}: {col}"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


for col in raw_band_cols.values():

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = (

    df.dropna(
        subset=[
            SITE_COL,
            SCENE_COL,
            TARGET,
        ]
    )

    .copy()

)


df = df.loc[
    df[TARGET] >= 0
].copy()


# ============================================================
# ACOLITE STORED REFLECTANCE -> RHOW
# ============================================================

for band, raw_col in raw_band_cols.items():

    df[
        f"{band}_rhow"
    ] = (

        df[
            raw_col
        ]

        / 10000.0

    )


# ============================================================
# SHORT NAMES
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


    out = np.full(
        numerator.shape,
        np.nan,
        dtype=float
    )


    good = (

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


    out[
        good
    ] = (

        numerator[
            good
        ]

        /

        denominator[
            good
        ]

    )


    return out


# ============================================================
# SANTOS INDICES
# ============================================================
#
# 16/17 available.
#
# GCI requires B09:
#
#     B09 / B03 - 1
#
# B09 is not available in this USGS matched product.
#
# ============================================================


df["NDVI"] = safe_divide(
    B8 - B4,
    B8 + B4
)


df["GNDVI"] = safe_divide(
    B8 - B3,
    B8 + B3
)


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


df["NDMI"] = safe_divide(
    B8 - B11,
    B8 + B11
)


df["MSI"] = safe_divide(
    B11,
    B8
)


df["NBRI"] = safe_divide(
    B8 - B12,
    B8 + B12
)


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


df["NDWI"] = safe_divide(
    B3 - B8,
    B3 + B8
)


df["NDSI"] = safe_divide(
    B3 - B11,
    B3 + B11
)


df["NDGI"] = safe_divide(
    B3 - B4,
    B3 + B4
)


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


df["SIPI"] = safe_divide(
    B8 - B2,
    B8 - B4
)


df["SWM"] = safe_divide(
    B2 + B3,
    B8 + B11
)


df["AWEI"] = (

    4.0 * (
        B3 - B11
    )

    -

    (
        0.25 * B8
        +
        2.75 * B12
    )

)


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
# FEATURE LISTS
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
# INVALID REPORT
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
            -np.inf,
        ],
        np.nan
    )

)


print(
    "\nInvalid values by index:"
)


print(

    df[
        INDEX_FEATURES
    ]

    .isna()

    .sum()

    .to_string()

)


# ============================================================
# SAME SAMPLE FOR ALL CANDIDATES
# ============================================================

before = len(
    df
)


complete = (

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
        complete
    ]

    .copy()

)


print(
    "\nRows before feature QC:",
    before
)

print(
    "Rows after feature QC:",
    len(df)
)


# ============================================================
# STABLE ORDER / SAMPLE ID
# ============================================================

sort_cols = [

    SITE_COL,
    SCENE_COL,

]


df = (

    df

    .sort_values(
        sort_cols
    )

    .reset_index(
        drop=True
    )

)


df[
    "sample_id"
] = [

    f"IL_SANTOS_{i:05d}"

    for i in range(
        1,
        len(df) + 1
    )

]


# ============================================================
# SANTOS-STYLE 80/20 SPLIT
# ============================================================
#
# Paper reports Pareto 80/20 split.
#
# Random seed is OUR reproducibility choice because the paper
# does not provide one.
#
# ============================================================

all_indices = np.arange(
    len(df)
)


train_idx, test_idx = train_test_split(

    all_indices,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    shuffle=True,

)


df[
    "split"
] = "train"


df.loc[
    test_idx,
    "split"
] = "test"


train = (

    df.loc[
        df[
            "split"
        ]
        == "train"
    ]

    .copy()

    .reset_index(
        drop=True
    )

)


test = (

    df.loc[
        df[
            "split"
        ]
        == "test"
    ]

    .copy()

    .reset_index(
        drop=True
    )

)


print(
    "\n"
    + "=" * 100
)

print(
    "SANTOS-STYLE 80/20 SPLIT"
)

print(
    "=" * 100
)


print(
    "Total:",
    len(df)
)

print(
    "Train:",
    len(train)
)

print(
    "Test:",
    len(test)
)


for name, temp in [

    ("TRAIN", train),

    ("TEST", test),

]:

    y = temp[
        TARGET
    ]


    print(
        f"\n{name}"
    )

    print(
        f" mean   = {y.mean():.3f}"
    )

    print(
        f" sd     = {y.std():.3f}"
    )

    print(
        f" median = {y.median():.3f}"
    )

    print(
        f" p95    = {y.quantile(0.95):.3f}"
    )

    print(
        f" max    = {y.max():.3f}"
    )

    print(
        f" >20    = {(y > 20).sum()}"
    )

    print(
        f" >50    = {(y > 50).sum()}"
    )


# ============================================================
# ORDINARY FIVE-FOLD CV
#
# NOT date-grouped.
#
# Paper does not specify the exact resampling used in forward
# selection; ordinary 5-fold CV is used here as a reproducible
# implementation within the 80% training data.
#
# ============================================================

cv = KFold(

    n_splits=N_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE,

)


# ============================================================
# XGBOOST BENCHMARK FOR FEATURE SELECTION
#
# Santos states XGBoost/R² was used as benchmark but does not
# report the exact feature-selection hyperparameters.
#
# These are fixed and documented for reproducibility.
#
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


y_train = (

    train[
        TARGET
    ]

    .astype(float)

    .to_numpy()

)


def evaluate(
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

            make_selection_model(),

            X,

            y_train,

            cv=cv,

            scoring="r2",

            n_jobs=-1,

            error_score="raise",

        )


    return {

        "mean":
            float(
                scores.mean()
            ),

        "sd":
            float(
                scores.std(
                    ddof=1
                )
            ),

        "se":
            float(
                scores.std(
                    ddof=1
                )
                /
                np.sqrt(
                    len(scores)
                )
            ),

        "folds":
            [
                float(x)
                for x in scores
            ],

    }


# ============================================================
# FULL FORWARD RANKING
#
# IMPORTANT:
#
# Do NOT stop at the first non-improvement.
#
# Santos Table 5 displays IC1 ... IC17 and chooses the best
# combination. We therefore rank ALL candidates and then pick
# the prefix with maximum mean CV R².
#
# ============================================================

def full_forward_ranking(
    fixed_features,
    candidates,
    stage_name
):

    selected_order = []

    remaining = list(
        candidates
    )

    rows = []


    baseline = None


    if len(
        fixed_features
    ) > 0:

        baseline = evaluate(
            fixed_features
        )


        rows.append({

            "stage":
                stage_name,

            "step":
                0,

            "added_feature":
                "NONE",

            "n_features":
                len(
                    fixed_features
                ),

            "mean_cv_r2":
                baseline[
                    "mean"
                ],

            "sd_cv_r2":
                baseline[
                    "sd"
                ],

            "se_cv_r2":
                baseline[
                    "se"
                ],

            "combination":
                "|".join(
                    fixed_features
                ),

        })


    for step in range(
        1,
        len(candidates) + 1
    ):

        trials = []


        print(
            "\n"
            + "-" * 100
        )

        print(
            f"{stage_name.upper()} STEP {step}"
        )

        print(
            "-" * 100
        )


        for candidate in remaining:

            features = (

                list(
                    fixed_features
                )

                +

                selected_order

                +

                [
                    candidate
                ]

            )


            result = evaluate(
                features
            )


            trials.append(

                (
                    candidate,
                    result,
                    features,
                )

            )


            print(

                f"{candidate:12s} "

                f"R²={result['mean']:.4f} "

                f"±SE {result['se']:.4f}"

            )


        best_candidate, best_result, best_features = max(

            trials,

            key=lambda x:
                x[1]["mean"]

        )


        selected_order.append(
            best_candidate
        )


        remaining.remove(
            best_candidate
        )


        rows.append({

            "stage":
                stage_name,

            "step":
                step,

            "added_feature":
                best_candidate,

            "n_features":
                len(
                    best_features
                ),

            "mean_cv_r2":
                best_result[
                    "mean"
                ],

            "sd_cv_r2":
                best_result[
                    "sd"
                ],

            "se_cv_r2":
                best_result[
                    "se"
                ],

            "fold_0_r2":
                best_result[
                    "folds"
                ][0],

            "fold_1_r2":
                best_result[
                    "folds"
                ][1],

            "fold_2_r2":
                best_result[
                    "folds"
                ][2],

            "fold_3_r2":
                best_result[
                    "folds"
                ][3],

            "fold_4_r2":
                best_result[
                    "folds"
                ][4],

            "combination":
                "|".join(
                    best_features
                ),

        })


        print(
            "\nSELECTED THIS STEP:",
            best_candidate
        )

        print(
            "Combination R²:",
            f"{best_result['mean']:.6f}"
        )


    path = pd.DataFrame(
        rows
    )


    best_row_index = (

        path[
            "mean_cv_r2"
        ]

        .astype(float)

        .idxmax()

    )


    best_row = path.loc[
        best_row_index
    ]


    best_step = int(
        best_row[
            "step"
        ]
    )


    if stage_name == "indices":

        best_added = (

            selected_order[
                :best_step
            ]

        )


    else:

        if best_step == 0:

            best_added = []

        else:

            best_added = (

                selected_order[
                    :best_step
                ]

            )


    return (

        selected_order,

        best_added,

        path,

        best_row,

    )


# ============================================================
# STAGE 1 — INDICES
# ============================================================

(
    index_ranking,
    selected_indices,
    index_path,
    best_index_row,

) = full_forward_ranking(

    fixed_features=[],

    candidates=
        INDEX_FEATURES,

    stage_name=
        "indices",

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

print(
    selected_indices
)


print(
    "\nBest index CV R²:",
    f"{best_index_row['mean_cv_r2']:.6f}"
)


# ============================================================
# STAGE 2 — RAW BANDS
# ============================================================

(
    band_ranking,
    selected_bands,
    band_path,
    best_band_row,

) = full_forward_ranking(

    fixed_features=
        selected_indices,

    candidates=
        BAND_FEATURES,

    stage_name=
        "bands",

)


band_path.to_csv(
    BAND_PATH_OUTPUT,
    index=False
)


# ============================================================
# FINAL ORIGINAL FEATURES
# ============================================================

selected_original_features = (

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

for x in selected_indices:

    print(
        " -",
        x
    )


print(
    "\nSelected bands:"
)


if len(
    selected_bands
) == 0:

    print(
        " - none"
    )


else:

    for x in selected_bands:

        print(
            " -",
            x
        )


print(
    "\nOriginal predictors:",
    len(
        selected_original_features
    )
)


print(
    "Best final CV R²:",
    f"{best_band_row['mean_cv_r2']:.6f}"
)


# ============================================================
# CUBIC ROOT OF EVERY SELECTED PREDICTOR
#
# np.cbrt correctly handles negative predictors.
#
# ============================================================

cube_features = []


for feature in selected_original_features:

    new_col = (
        f"{feature}_cbrt"
    )


    df[
        new_col
    ] = np.cbrt(

        df[
            feature
        ]
        .astype(float)

    )


    cube_features.append(
        new_col
    )


final_model_features = (

    selected_original_features
    +
    cube_features

)


print(
    "\nCube-root predictors added:",
    len(
        cube_features
    )
)


print(
    "TOTAL model predictors:",
    len(
        final_model_features
    )
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_DATA,
    index=False
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "experiment":
        "Illinois Santos-style primary experiment",

    "random_state":
        RANDOM_STATE,

    "split":
        "random 80/20",

    "n_total":
        int(
            len(df)
        ),

    "n_train":
        int(
            (
                df["split"]
                == "train"
            ).sum()
        ),

    "n_test":
        int(
            (
                df["split"]
                == "test"
            ).sum()
        ),

    "available_indices":
        INDEX_FEATURES,

    "unavailable_indices": [
        "GCI"
    ],

    "gci_reason":
        "B09 absent from USGS ACOLITE product",

    "available_bands":
        BAND_FEATURES,

    "unavailable_sentinel_bands_relevant_to_santos": [
        "B09",
        "B10"
    ],

    "index_forward_ranking":
        index_ranking,

    "selected_indices":
        selected_indices,

    "band_forward_ranking":
        band_ranking,

    "selected_bands":
        selected_bands,

    "selected_original_features":
        selected_original_features,

    "cube_root_features":
        cube_features,

    "final_model_features":
        final_model_features,

    "index_best_mean_cv_r2":
        float(
            best_index_row[
                "mean_cv_r2"
            ]
        ),

    "final_selection_mean_cv_r2":
        float(
            best_band_row[
                "mean_cv_r2"
            ]
        ),

    "feature_selection_cv":
        (
            "ordinary shuffled 5-fold CV within "
            "the 80% training set"
        ),

    "note":
        (
            "The Santos paper does not report the "
            "exact resampling scheme or XGBoost "
            "hyperparameters used during forward selection."
        ),

}


with open(
    FEATURE_JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
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
    OUTPUT_DATA
)

print(
    INDEX_PATH_OUTPUT
)

print(
    BAND_PATH_OUTPUT
)

print(
    FEATURE_JSON_OUTPUT
)


print(
    "\nNext: normalization comparison + "
    "XGBoost GridSearchCV using the selected "
    "original + cubic-root predictors."
)
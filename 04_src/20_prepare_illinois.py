from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
)


# ============================================================
# PURPOSE
# ============================================================
#
# Build the final Illinois modeling base from the five-basin
# screening output.
#
# Then create:
#
# 1. Fresh Illinois sample IDs
# 2. Fresh acquisition-date grouped 80/20 holdout
# 3. Fresh 5-fold acquisition-date grouped CV within TRAINING
#
# IMPORTANT:
# Delaware splits/features/models are NOT reused.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "05_results/tables/00_all_basin_model_ready_rows.csv"
)

BASE_OUTPUT = Path(
    "01_data/processed/illinois_modeling_base.csv"
)

SPLIT_OUTPUT = Path(
    "01_data/processed/illinois_train_test_split.csv"
)

SUMMARY_OUTPUT = Path(
    "05_results/tables/20_illinois_dataset_summary.csv"
)


BASE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MeasurementValue"

SITE_COL = "site_no"

SCENE_COL = "scene"

DATETIME_COL = "_scene_datetime"

BASIN_NAME = "Illinois"

RANDOM_STATE = 42

TEST_FRACTION = 0.20

N_CV_FOLDS = 5


# ============================================================
# ACOLITE BANDS
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


# ============================================================
# LOAD SCREENED DATA
# ============================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"Missing screening file:\n{INPUT}"
    )


df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={
        SITE_COL: "string"
    }
)


print("=" * 100)
print("ILLINOIS DATASET PREPARATION")
print("=" * 100)

print(
    "\nRows in screening dataset:",
    len(df)
)


# ============================================================
# FILTER ILLINOIS
# ============================================================

if "basin" not in df.columns:

    raise KeyError(
        "The screening file has no 'basin' column."
    )


illinois = (

    df.loc[
        df["basin"]
        .astype(str)
        .str.strip()
        .eq(BASIN_NAME)
    ]

    .copy()

)


print(
    "Illinois rows:",
    len(illinois)
)


if len(illinois) == 0:

    raise RuntimeError(
        "No Illinois rows found."
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [

    SITE_COL,
    SCENE_COL,
    TARGET,
    DATETIME_COL,

]


for col in required:

    if col not in illinois.columns:

        raise KeyError(
            f"Required column missing: {col}"
        )


# ============================================================
# CLEAN TARGET
# ============================================================

illinois[TARGET] = pd.to_numeric(
    illinois[TARGET],
    errors="coerce"
)


illinois = (

    illinois

    .dropna(
        subset=[
            SITE_COL,
            SCENE_COL,
            TARGET,
            DATETIME_COL,
        ]
    )

    .copy()

)


illinois = (

    illinois.loc[
        illinois[TARGET] >= 0
    ]

    .copy()

)


# ============================================================
# PARSE SCENE DATETIME
# ============================================================

illinois[DATETIME_COL] = pd.to_datetime(

    illinois[DATETIME_COL],

    utc=True,

    errors="coerce"

)


illinois = (

    illinois

    .dropna(
        subset=[
            DATETIME_COL
        ]
    )

    .copy()

)


illinois[
    "acquisition_date"
] = (

    illinois[
        DATETIME_COL
    ]

    .dt.strftime(
        "%Y-%m-%d"
    )

)


# ============================================================
# CHECK DUPLICATE STATION-SCENE PAIRS
#
# Screening script should already have selected the nearest
# water-quality measurement per station + scene.
# ============================================================

duplicate_count = int(

    illinois.duplicated(

        subset=[
            SITE_COL,
            SCENE_COL,
        ],

        keep=False

    ).sum()

)


print(
    "\nDuplicate station-scene rows:",
    duplicate_count
)


if duplicate_count > 0:

    raise RuntimeError(
        "Duplicate station-scene rows found. "
        "Do not continue until investigated."
    )


# ============================================================
# DETECT EXISTING 250 m MEDIAN BANDS
# ============================================================

raw_band_cols = {}


for band in BANDS:

    candidates = [

        f"{band}_buf250_med",

        f"{band}_buf250_median",

    ]


    found = None


    for candidate in candidates:

        if candidate in illinois.columns:

            found = candidate

            break


    if found is None:

        # fallback search

        matches = [

            c
            for c in illinois.columns
            if (
                c.upper().startswith(
                    band.upper()
                )
                and
                "250" in c
                and
                "MED" in c.upper()
            )

        ]


        if len(matches) == 1:

            found = matches[0]


    if found is None:

        raise KeyError(
            f"Could not find 250 m median column for {band}"
        )


    raw_band_cols[
        band
    ] = found


print(
    "\nDetected 250 m median columns:"
)


for band, col in raw_band_cols.items():

    print(
        f" {band}: {col}"
    )


# ============================================================
# NUMERIC REFLECTANCE QC
# ============================================================

for col in raw_band_cols.values():

    illinois[col] = pd.to_numeric(
        illinois[col],
        errors="coerce"
    )


before_reflectance_qc = len(
    illinois
)


raw_matrix = illinois[
    list(
        raw_band_cols.values()
    )
]


valid_reflectance = (

    raw_matrix
    .notna()
    .all(
        axis=1
    )

    &

    pd.Series(

        np.isfinite(
            raw_matrix
            .to_numpy(
                dtype=float
            )
        ).all(
            axis=1
        ),

        index=illinois.index

    )

    &

    raw_matrix
    .ge(0)
    .all(
        axis=1
    )

)


illinois = (

    illinois.loc[
        valid_reflectance
    ]

    .copy()

)


print(
    "\nRows before reflectance check:",
    before_reflectance_qc
)

print(
    "Rows after reflectance check:",
    len(illinois)
)


# ============================================================
# CONVERT USGS STORED REFLECTANCE TO RHOW
#
# Same scaling used in the Delaware workflow:
#
# stored value / 10000
#
# Keep raw columns as well.
# ============================================================

SCALE_FACTOR = 10000.0


for band, raw_col in raw_band_cols.items():

    new_col = (
        f"{band}_rhow"
    )


    illinois[
        new_col
    ] = (

        illinois[
            raw_col
        ]

        / SCALE_FACTOR

    )


# ============================================================
# REFLECTANCE SANITY REPORT
# ============================================================

print(
    "\nReflectance ranges after /10000 scaling:"
)


for band in BANDS:

    col = (
        f"{band}_rhow"
    )


    print(

        f"{band}: "

        f"min={illinois[col].min():.5f}, "

        f"median={illinois[col].median():.5f}, "

        f"p99={illinois[col].quantile(0.99):.5f}, "

        f"max={illinois[col].max():.5f}"

    )


# ============================================================
# SORT FIRST
#
# Gives stable sample IDs independent of original CSV order.
# ============================================================

illinois = (

    illinois

    .sort_values(
        [
            SITE_COL,
            DATETIME_COL,
            SCENE_COL,
        ]
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# STABLE SAMPLE ID
# ============================================================

illinois[
    "sample_id"
] = [

    f"IL_{i:05d}"

    for i in range(
        1,
        len(illinois) + 1
    )

]


# ============================================================
# DATASET STATISTICS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "ILLINOIS MODELING BASE"
)

print(
    "=" * 100
)


print(
    "\nRows:",
    len(illinois)
)

print(
    "Stations:",
    illinois[
        SITE_COL
    ].nunique()
)

print(
    "Unique acquisition dates:",
    illinois[
        "acquisition_date"
    ].nunique()
)


print(
    "\nRows by station:"
)


station_name_col = (

    "station_nm"

    if "station_nm"
    in illinois.columns

    else SITE_COL

)


print(

    illinois[
        station_name_col
    ]

    .value_counts()

    .to_string()

)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

y = illinois[
    TARGET
]


print(
    "\nTurbidity distribution:"
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
    f" p90    = {y.quantile(0.90):.3f}"
)

print(
    f" p95    = {y.quantile(0.95):.3f}"
)

print(
    f" p99    = {y.quantile(0.99):.3f}"
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

print(
    f" >100   = {(y > 100).sum()}"
)


# ============================================================
# SAVE CLEAN MODELING BASE BEFORE SPLITTING
# ============================================================

illinois.to_csv(
    BASE_OUTPUT,
    index=False
)


# ============================================================
# FRESH PRIMARY HOLDOUT
#
# IMPORTANT:
#
# Split by acquisition DATE, not individual row.
#
# This prevents stations observed on the same satellite date
# from being divided between train and test.
# ============================================================

groups = (

    illinois[
        "acquisition_date"
    ]
    .astype(str)
    .values

)


gss = GroupShuffleSplit(

    n_splits=1,

    test_size=TEST_FRACTION,

    random_state=RANDOM_STATE,

)


train_idx, test_idx = next(

    gss.split(

        X=illinois,

        y=illinois[
            TARGET
        ],

        groups=groups

    )

)


illinois[
    "split"
] = "train"


illinois.loc[
    test_idx,
    "split"
] = "test"


# ============================================================
# CONFIRM NO ACQUISITION-DATE LEAKAGE
# ============================================================

train_dates = set(

    illinois.loc[
        train_idx,
        "acquisition_date"
    ]

)


test_dates = set(

    illinois.loc[
        test_idx,
        "acquisition_date"
    ]

)


overlap = (

    train_dates
    &
    test_dates

)


if overlap:

    raise RuntimeError(

        "Acquisition-date leakage detected "
        "between train and test."

    )


# ============================================================
# FIVE-FOLD GROUPED CV INSIDE TRAINING ONLY
#
# Test observations remain cv_fold = -1.
# ============================================================

illinois[
    "cv_fold"
] = -1


train_df = (

    illinois.loc[
        train_idx
    ]

    .copy()

)


train_groups = (

    train_df[
        "acquisition_date"
    ]

    .astype(str)

    .values

)


gkf = GroupKFold(
    n_splits=N_CV_FOLDS
)


for fold, (
    fold_train_pos,
    fold_val_pos
) in enumerate(

    gkf.split(

        X=train_df,

        y=train_df[
            TARGET
        ],

        groups=train_groups,

    )

):

    fold_val_indices = (

        train_df.index[
            fold_val_pos
        ]

    )


    illinois.loc[

        fold_val_indices,

        "cv_fold"

    ] = fold


# ============================================================
# VALIDATE CV ASSIGNMENT
# ============================================================

if (

    illinois.loc[
        illinois[
            "split"
        ] == "train",
        "cv_fold"
    ]

    .lt(0)

    .any()

):

    raise RuntimeError(
        "Some training rows did not receive a CV fold."
    )


if (

    illinois.loc[
        illinois[
            "split"
        ] == "test",
        "cv_fold"
    ]

    .ne(-1)

    .any()

):

    raise RuntimeError(
        "Test rows should have cv_fold = -1."
    )


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOCKED ILLINOIS VALIDATION SPLIT"
)

print(
    "=" * 100
)


print(
    "\nTrain rows:",
    int(
        (
            illinois[
                "split"
            ] == "train"
        ).sum()
    )
)

print(
    "Test rows:",
    int(
        (
            illinois[
                "split"
            ] == "test"
        ).sum()
    )
)


print(
    "\nTrain unique dates:",
    illinois.loc[
        illinois[
            "split"
        ] == "train",
        "acquisition_date"
    ].nunique()
)


print(
    "Test unique dates:",
    illinois.loc[
        illinois[
            "split"
        ] == "test",
        "acquisition_date"
    ].nunique()
)


print(
    "Train/test date overlap:",
    len(
        overlap
    )
)


# ============================================================
# TRAIN / TEST TARGET DISTRIBUTION
# ============================================================

for split_name in [
    "train",
    "test"
]:

    temp = illinois.loc[
        illinois[
            "split"
        ] == split_name,
        TARGET
    ]


    print(
        f"\n{split_name.upper()}"
    )

    print(
        f" n      = {len(temp)}"
    )

    print(
        f" mean   = {temp.mean():.3f}"
    )

    print(
        f" sd     = {temp.std():.3f}"
    )

    print(
        f" median = {temp.median():.3f}"
    )

    print(
        f" p95    = {temp.quantile(0.95):.3f}"
    )

    print(
        f" max    = {temp.max():.3f}"
    )

    print(
        f" >20    = {(temp > 20).sum()}"
    )

    print(
        f" >50    = {(temp > 50).sum()}"
    )

    print(
        f" >100   = {(temp > 100).sum()}"
    )


# ============================================================
# CV FOLD TARGET DISTRIBUTIONS
# ============================================================

print(
    "\nCV fold distributions:"
)


cv_rows = []


for fold in range(
    N_CV_FOLDS
):

    temp = illinois.loc[
        illinois[
            "cv_fold"
        ] == fold
    ]


    cv_rows.append({

        "fold":
            fold,

        "n":
            len(temp),

        "n_dates":
            temp[
                "acquisition_date"
            ].nunique(),

        "mean":
            temp[
                TARGET
            ].mean(),

        "sd":
            temp[
                TARGET
            ].std(),

        "median":
            temp[
                TARGET
            ].median(),

        "p95":
            temp[
                TARGET
            ].quantile(
                0.95
            ),

        "max":
            temp[
                TARGET
            ].max(),

        "n_gt_20":
            int(
                (
                    temp[
                        TARGET
                    ] > 20
                ).sum()
            ),

        "n_gt_50":
            int(
                (
                    temp[
                        TARGET
                    ] > 50
                ).sum()
            ),

    })


cv_summary = pd.DataFrame(
    cv_rows
)


print(

    cv_summary

    .round(
        3
    )

    .to_string(
        index=False
    )

)


# ============================================================
# FINAL DATASET SUMMARY FILE
# ============================================================

summary_rows = []


for (
    site_no,
    station_nm
), temp in illinois.groupby(

    [
        SITE_COL,
        station_name_col,
    ],

    dropna=False

):

    summary_rows.append({

        "site_no":
            site_no,

        "station_nm":
            station_nm,

        "n":
            len(temp),

        "mean":
            temp[
                TARGET
            ].mean(),

        "sd":
            temp[
                TARGET
            ].std(),

        "median":
            temp[
                TARGET
            ].median(),

        "p95":
            temp[
                TARGET
            ].quantile(
                0.95
            ),

        "max":
            temp[
                TARGET
            ].max(),

        "n_gt_20":
            int(
                (
                    temp[
                        TARGET
                    ] > 20
                ).sum()
            ),

        "n_gt_50":
            int(
                (
                    temp[
                        TARGET
                    ] > 50
                ).sum()
            ),

        "n_train":
            int(
                (
                    temp[
                        "split"
                    ] == "train"
                ).sum()
            ),

        "n_test":
            int(
                (
                    temp[
                        "split"
                    ] == "test"
                ).sum()
            ),

    })


station_summary = (

    pd.DataFrame(
        summary_rows
    )

    .sort_values(
        "n",
        ascending=False
    )

)


station_summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# SAVE LOCKED SPLIT
# ============================================================

illinois.to_csv(
    SPLIT_OUTPUT,
    index=False
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
    BASE_OUTPUT
)

print(
    SPLIT_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


print(
    "\nIllinois preparation complete."
)

print(
    "Do NOT recreate this split later."
)

print(
    "Next: rebuild Santos spectral indices using TRAINING data "
    "for feature selection, then fit XGBoost and GPR."
)
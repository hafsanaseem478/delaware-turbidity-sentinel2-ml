from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

INPUT_FILE = Path(
    "05_results/tables/00_all_basin_model_ready_rows.csv"
)

OUTPUT_FILE = Path(
    "01_data/processed/delaware_santos_primary.csv"
)

SUMMARY_FILE = Path(
    "05_results/metrics/30_delaware_santos_dataset.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "MeasurementValue"

RANDOM_STATE = 42

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

SANTOS_INDICES = [

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

print("=" * 100)

print(
    "DELAWARE â€” SANTOS-STYLE DATASET REBUILD"
)

print("=" * 100)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(

    INPUT_FILE,

    low_memory=False,

    dtype={
        "site_no": "string"
    },

)

print(
    "\nInput rows:",
    len(df)
)

if "site_no" in df.columns:

    df["site_no"] = (

        df["site_no"]

        .astype("string")

        .str.strip()

        .str.replace(
            r"\.0$",
            "",
            regex=True
        )

    )

possible_basin_columns = [

    "basin",
    "Basin",
    "basin_name",
    "BASIN",

]

basin_col = None

for col in possible_basin_columns:

    if col in df.columns:

        basin_col = col

        break

if basin_col is None:

    raise KeyError(

        "Could not locate the basin column.\n"
        f"Available columns:\n{df.columns.tolist()}"

    )

df = (

    df.loc[

        df[
            basin_col
        ]

        .astype(str)

        .str.contains(
            "Delaware",
            case=False,
            na=False
        )

    ]

    .copy()

)

print(
    "Delaware input rows:",
    len(df)
)

if len(df) == 0:

    raise RuntimeError(
        "No Delaware observations were found."
    )

if TARGET not in df.columns:

    raise KeyError(
        f"Target column '{TARGET}' not found."
    )

df[TARGET] = pd.to_numeric(

    df[TARGET],

    errors="coerce"

)

df = (

    df.loc[
        df[TARGET].notna()
    ]

    .copy()

)

if "site_no" in df.columns:

    print(
        "\nUSGS stations detected:"
    )

    for site in sorted(

        df["site_no"]
        .dropna()
        .astype(str)
        .unique()

    ):

        print(
            f" {site}"
        )

def find_band_column(
    dataframe,
    band
):

    candidates = [

        f"{band}_buf250_med",

        f"{band}_rhow",

        f"{band}_buf250_mean",

        f"{band}_center",

        band,

    ]

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    return None

band_source = {}

print(
    "\nDetected band columns:"
)

for band in BANDS:

    source = find_band_column(
        df,
        band
    )

    if source is None:

        raise KeyError(
            f"Could not locate a source column for {band}."
        )

    band_source[
        band
    ] = source

    print(
        f" {band}: {source}"
    )

print(
    "\nReflectance scaling:"
)

for band in BANDS:

    source = band_source[
        band
    ]

    values = pd.to_numeric(

        df[source],

        errors="coerce"

    )

    finite_values = (

        values.loc[
            np.isfinite(values)
        ]

    )

    if len(finite_values) == 0:

        raise RuntimeError(
            f"No finite values found for {band}."
        )

    median_absolute_value = float(

        finite_values
        .abs()
        .median()

    )

    if (

        source.endswith(
            "_rhow"
        )

        or

        median_absolute_value <= 2.0

    ):

        scale_factor = 1.0

        scaling_label = (
            "already reflectance"
        )

    else:

        scale_factor = 10000.0

        scaling_label = (
            "divided by 10000"
        )

    df[
        f"{band}_rhow"
    ] = (

        values
        /
        scale_factor

    )

    print(
        f" {band}: {scaling_label}"
    )

B01 = df["B01_rhow"]

B02 = df["B02_rhow"]

B03 = df["B03_rhow"]

B04 = df["B04_rhow"]

B05 = df["B05_rhow"]

B06 = df["B06_rhow"]

B07 = df["B07_rhow"]

B08 = df["B08_rhow"]

B8A = df["B8A_rhow"]

B11 = df["B11_rhow"]

B12 = df["B12_rhow"]

def safe_div(
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

    with np.errstate(

        divide="ignore",

        invalid="ignore",

    ):

        result = np.divide(

            numerator,

            denominator,

        )

    result = np.where(

        np.abs(
            denominator
        ) > 1e-12,

        result,

        np.nan,

    )

    return result

df["NDVI"] = safe_div(

    B08 - B04,

    B08 + B04,

)

df["GNDVI"] = safe_div(

    B08 - B03,

    B08 + B03,

)

df["EVI"] = (

    2.5

    *

    safe_div(

        B08 - B04,

        B08
        + 6.0 * B04
        - 7.5 * B02
        + 1.0,

    )

)

df["SAVI"] = (

    1.428

    *

    safe_div(

        B08 - B04,

        B08
        + B04
        + 0.428,

    )

)

df["NDMI"] = safe_div(

    B08 - B11,

    B08 + B11,

)

df["MSI"] = safe_div(

    B11,

    B08,

)

df["NBRI"] = safe_div(

    B08 - B12,

    B08 + B12,

)

df["BSI"] = safe_div(

    (
        B11
        + B04
    )

    -

    (
        B08
        + B02
    ),

    (
        B11
        + B04
    )

    +

    (
        B08
        + B02
    ),

)

df["NDWI"] = safe_div(

    B03 - B08,

    B03 + B08,

)

df["NDSI"] = safe_div(

    B03 - B11,

    B03 + B11,

)

df["NDGI"] = safe_div(

    B03 - B04,

    B03 + B04,

)

df["ARVI"] = safe_div(

    B08
    - 2.0 * B04
    + B02,

    B08
    + 2.0 * B04,

)

df["SIPI"] = safe_div(

    B08 - B02,

    B08 - B04,

)

df["SWM"] = safe_div(

    B02 + B03,

    B08 + B11,

)

df["AWEI"] = (

    4.0
    *
    (
        B03
        -
        B11
    )

    -

    (
        0.25 * B08
        +
        2.75 * B12
    )

)

df["3BSI"] = (

    (

        safe_div(
            1.0,
            B04
        )

        -

        safe_div(
            1.0,
            B05
        )

    )

    *

    B06

)

print(
    "\nInvalid values by index:"
)

invalid_counts = {}

for feature in SANTOS_INDICES:

    values = (

        pd.to_numeric(

            df[feature],

            errors="coerce"

        )

        .replace(

            [
                np.inf,
                -np.inf
            ],

            np.nan

        )

    )

    count = int(

        values
        .isna()
        .sum()

    )

    invalid_counts[
        feature
    ] = count

    print(
        f"{feature:<8} {count}"
    )

band_features = [

    f"{band}_rhow"

    for band in BANDS

]

required_model_columns = (

    SANTOS_INDICES

    +

    band_features

    +

    [
        TARGET
    ]

)

rows_before_qc = len(
    df
)

df[
    required_model_columns
] = (

    df[
        required_model_columns
    ]

    .replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan

    )

)

df = (

    df

    .dropna(
        subset=
            required_model_columns
    )

    .reset_index(
        drop=True
    )

)

rows_after_qc = len(
    df
)

print(
    "\nRows before feature QC:",
    rows_before_qc
)

print(
    "Rows after feature QC:",
    rows_after_qc
)

if "sample_id" not in df.columns:

    df[
        "sample_id"
    ] = [

        f"DEL_{i:05d}"

        for i in range(
            1,
            len(df) + 1
        )

    ]

else:

    df[
        "sample_id"
    ] = (

        df[
            "sample_id"
        ]

        .astype(str)

    )

all_indices = np.arange(
    len(df)
)

train_indices, test_indices = train_test_split(

    all_indices,

    test_size=
        0.20,

    random_state=
        RANDOM_STATE,

    shuffle=
        True,

)

df[
    "split"
] = "test"

df.loc[

    train_indices,

    "split"

] = "train"

train = (

    df.loc[
        df["split"] == "train"
    ]

    .copy()

)

test = (

    df.loc[
        df["split"] == "test"
    ]

    .copy()

)

def describe_target(
    data,
    name
):

    y = (

        data[
            TARGET
        ]

        .astype(float)

    )

    print(
        f"\n{name}"
    )

    print(
        f" n      = {len(y)}"
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

    print(
        f" >100   = {(y > 100).sum()}"
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
    "\nTotal:",
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

describe_target(
    train,
    "TRAIN"
)

describe_target(
    test,
    "TEST"
)

if "site_no" in df.columns:

    print(
        "\nStations:"
    )

    aggregation = {

        "n": (
            TARGET,
            "size"
        ),

        "mean": (
            TARGET,
            "mean"
        ),

        "max": (
            TARGET,
            "max"
        ),

    }

    if "station_nm" in df.columns:

        station_summary = (

            df

            .groupby(
                [
                    "site_no",
                    "station_nm"
                ],
                dropna=False
            )

            .agg(
                **aggregation
            )

            .sort_values(
                "n",
                ascending=False
            )

        )

    else:

        station_summary = (

            df

            .groupby(
                "site_no",
                dropna=False
            )

            .agg(
                **aggregation
            )

            .sort_values(
                "n",
                ascending=False
            )

        )

    print(

        station_summary

        .round(
            3
        )

        .to_string()

    )

expected_delaware_sites = {

    "01427207",
    "01438500",
    "01458500",
    "01463500",
    "014670261",
    "01467200",
    "01473500",

}

if "site_no" in df.columns:

    actual_sites = set(

        df[
            "site_no"
        ]

        .dropna()

        .astype(str)

        .unique()

    )

    print(
        "\nStation-ID check:"
    )

    print(
        " Expected:",
        sorted(
            expected_delaware_sites
        )
    )

    print(
        " Actual:  ",
        sorted(
            actual_sites
        )
    )

    if (
        actual_sites
        ==
        expected_delaware_sites
    ):

        print(
            " Station IDs are correct."
        )

    else:

        print(
            " WARNING: station IDs differ from expected Delaware IDs."
        )

train_ids = set(
    train["sample_id"]
)

test_ids = set(
    test["sample_id"]
)

overlap = (

    train_ids

    .intersection(
        test_ids
    )

)

if len(overlap) != 0:

    raise RuntimeError(
        "Train/test sample_id overlap detected."
    )

df.to_csv(

    OUTPUT_FILE,

    index=False

)

summary = {

    "study_area":
        "Delaware River Basin",

    "source_dataset":
        str(
            INPUT_FILE
        ),

    "reflectance":
        (
            "USGS ACOLITE-DSF "
            "aquatic reflectance"
        ),

    "santos_reference_reflectance":
        (
            "Sentinel-2 Level-1C "
            "TOA reflectance"
        ),

    "target":
        TARGET,

    "random_state":
        RANDOM_STATE,

    "split_method":
        (
            "Random 80/20 train/test split"
        ),

    "rows_before_feature_qc":
        int(
            rows_before_qc
        ),

    "rows_after_feature_qc":
        int(
            rows_after_qc
        ),

    "n_train":
        int(
            len(train)
        ),

    "n_test":
        int(
            len(test)
        ),

    "available_santos_indices":
        SANTOS_INDICES,

    "unavailable_santos_indices": {

        "GCI":
            (
                "Requires Sentinel-2 B09, "
                "which is unavailable in the "
                "USGS ACOLITE dataset."
            )

    },

    "available_bands":
        band_features,

    "unavailable_santos_bands": [

        "B09",
        "B10",

    ],

    "invalid_index_counts_before_qc":
        invalid_counts,

    "site_numbers":
        (
            sorted(
                df[
                    "site_no"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if "site_no" in df.columns

            else []
        ),

    "test_set_used_for_feature_selection":
        False,

}

with open(

    SUMMARY_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

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
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print(
    "\nDataset rebuild complete."
)

print(
    "The held-out TEST set must remain untouched during feature selection."
)


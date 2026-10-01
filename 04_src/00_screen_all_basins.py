from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PURPOSE
# ============================================================
#
# Screen all five USGS basins BEFORE deciding whether to
# continue with Delaware or switch basin.
#
# This script does NOT train any ML model.
#
# It compares:
#
# 1. Turbidity observations
# 2. Number of stations
# 3. Unique station-scene matchups
# 4. <=30 minute satellite / turbidity matches
# 5. Complete 250 m ACOLITE reflectance
# 6. River/flowline-only observations
# 7. Turbidity distribution and upper tail
# 8. Station balance
#
# The goal is to identify whether another basin provides
# substantially better data than Delaware.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path(
    "01_data/raw"
)

OUT_DIR = Path(
    "05_results/tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_OUT = (
    OUT_DIR
    / "00_all_basin_screening.csv"
)

STATION_OUT = (
    OUT_DIR
    / "00_all_basin_station_screening.csv"
)

READY_OUT = (
    OUT_DIR
    / "00_all_basin_model_ready_rows.csv"
)


# ============================================================
# FIND RAW MATCHED FILES
# ============================================================

files = sorted(
    RAW_DIR.rglob(
        "Matched_WQ_S2*.csv"
    )
)


if not files:

    raise FileNotFoundError(
        "Could not find Matched_WQ_S2*.csv "
        "inside 01_data/raw"
    )


print(
    "=" * 100
)

print(
    "ALL-BASIN TURBIDITY SCREENING"
)

print(
    "=" * 100
)


print(
    "\nFiles found:"
)


for f in files:

    print(
        " -",
        f
    )


# ============================================================
# READ HEADER
# ============================================================

header = pd.read_csv(
    files[0],
    nrows=0
)

columns = list(
    header.columns
)


print(
    "\nColumns in raw dataset:",
    len(columns)
)


# ============================================================
# HELPER: FIND COLUMN
# ============================================================

def find_column(
    candidates,
    required=True
):

    lower_lookup = {

        str(c).lower():
            c

        for c in columns
    }


    for candidate in candidates:

        key = (
            candidate
            .lower()
        )

        if key in lower_lookup:

            return lower_lookup[
                key
            ]


    if required:

        raise KeyError(

            "Could not find any of these columns:\n"
            +
            "\n".join(
                candidates
            )

        )


    return None


# ============================================================
# CORE COLUMNS
# ============================================================

SITE_COL = find_column(
    [
        "site_no",
        "site",
        "site_number",
        "MonitoringLocationIdentifier",
    ]
)


SCENE_COL = find_column(
    [
        "scene",
        "scene_id",
        "Scene",
        "s2_scene",
    ]
)


HUC_COL = find_column(
    [
        "huc4",
        "HUC4",
    ]
)


TARGET_COL = find_column(
    [
        "MeasurementValue",
        "measurement_value",
        "ResultMeasureValue",
        "value",
    ]
)


# ============================================================
# PARAMETER NAME
#
# IMPORTANT:
# Raw USGS files use parm_nm.
# ============================================================

CHAR_COL = find_column(
    [
        "parm_nm",
        "CharacteristicName",
        "characteristic_name",
        "parameter_name",
        "ParameterName",
        "characteristic",
    ],
    required=False
)


# ============================================================
# PARAMETER UNIT
#
# We will restrict the analysis to FNU so that basins are
# compared using the same turbidity unit.
# ============================================================

UNIT_COL = find_column(
    [
        "parm_unit",
        "unit",
        "ResultMeasure/MeasureUnitCode",
        "measurement_unit",
    ],
    required=False
)


# ============================================================
# OPTIONAL STATION INFORMATION
# ============================================================

STATION_COL = find_column(
    [
        "station_nm",
        "station_name",
        "MonitoringLocationName",
        "site_name",
    ],
    required=False
)


FEATURE_TYPE_COL = find_column(
    [
        "nhd_feature_type",
        "NHD_feature_type",
        "feature_type",
    ],
    required=False
)


# ============================================================
# TIME COLUMNS
#
# IMPORTANT:
# Raw USGS files use sample_datetime_UTC.
# ============================================================

MEASUREMENT_DT_COL = find_column(
    [
        "sample_datetime_UTC",
        "Datetime",
        "datetime",
        "MeasurementDateTime",
        "measurement_datetime",
        "measurement_datetime_UTC",
        "wq_datetime_UTC",
        "date_time",
    ],
    required=False
)


SCENE_DT_COL = find_column(
    [
        "scene_datetime_UTC",
        "scene_datetime",
        "SceneDateTime",
        "scene_date_time",
    ],
    required=False
)


DELTA_COL = find_column(
    [
        "delta_minutes",
        "time_diff_minutes",
        "time_difference_minutes",
        "time_delta_minutes",
    ],
    required=False
)


# ============================================================
# PRINT DETECTED COLUMNS
# ============================================================

print(
    "\nDetected key columns:"
)

print(
    "SITE       :",
    SITE_COL
)

print(
    "SCENE      :",
    SCENE_COL
)

print(
    "HUC4       :",
    HUC_COL
)

print(
    "TARGET     :",
    TARGET_COL
)

print(
    "PARAMETER  :",
    CHAR_COL
)

print(
    "UNIT       :",
    UNIT_COL
)

print(
    "STATION    :",
    STATION_COL
)

print(
    "NHD TYPE   :",
    FEATURE_TYPE_COL
)

print(
    "WQ TIME    :",
    MEASUREMENT_DT_COL
)

print(
    "SCENE TIME :",
    SCENE_DT_COL
)

print(
    "DELTA      :",
    DELTA_COL
)


# ============================================================
# SAFETY CHECK
# ============================================================

if CHAR_COL is None:

    raise RuntimeError(

        "Could not detect the water-quality parameter column. "
        "Expected parm_nm or equivalent. "
        "Stopping because we must explicitly filter turbidity."

    )


if MEASUREMENT_DT_COL is None and DELTA_COL is None:

    raise RuntimeError(

        "Could not find sample_datetime_UTC, another "
        "measurement datetime column, or an existing "
        "time-difference column."

    )


# ============================================================
# DETECT 250 m MEDIAN BAND COLUMNS
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


def find_band_250_median(
    band
):

    patterns = [

        rf"^{band}_buf250_med$",

        rf"^{band}.*buf250.*med$",

        rf"^{band}.*250.*median$",

        rf"^{band}.*250.*med$",

    ]


    for pattern in patterns:

        for col in columns:

            if re.search(
                pattern,
                col,
                flags=re.IGNORECASE
            ):

                return col


    return None


band_cols = {}


for band in BANDS:

    found = find_band_250_median(
        band
    )

    if found is not None:

        band_cols[
            band
        ] = found


print(
    "\nDetected 250 m median bands:",
    len(band_cols),
    "/",
    len(BANDS)
)


for band, col in band_cols.items():

    print(
        f" {band}: {col}"
    )


if len(band_cols) < 8:

    raise RuntimeError(

        "Too few 250 m median reflectance bands were "
        "detected. Check the raw column names."

    )


REFLECTANCE_COLS = list(
    band_cols.values()
)


# ============================================================
# BASIN DEFINITIONS
# ============================================================
#
# We group the relevant HUC4s belonging to the five study
# regions in the USGS release.
#
# ============================================================

HUC_TO_BASIN = {

    # Delaware
    "0204":
        "Delaware",

    # Illinois
    "0712":
        "Illinois",

    "0713":
        "Illinois",

    # Trinity / San Jacinto
    "1203":
        "Trinity-San Jacinto",

    "1204":
        "Trinity-San Jacinto",

    # Upper Colorado
    "1401":
        "Upper Colorado",

    "1402":
        "Upper Colorado",

    # Willamette
    "1709":
        "Willamette",

}


EXPECTED_BASINS = [

    "Delaware",

    "Illinois",

    "Trinity-San Jacinto",

    "Upper Colorado",

    "Willamette",

]


# ============================================================
# COLUMNS TO LOAD
# ============================================================

needed_cols = [

    SITE_COL,
    SCENE_COL,
    HUC_COL,
    TARGET_COL,
    CHAR_COL,

] + REFLECTANCE_COLS


for optional_col in [

    UNIT_COL,

    STATION_COL,

    FEATURE_TYPE_COL,

    MEASUREMENT_DT_COL,

    SCENE_DT_COL,

    DELTA_COL,

]:

    if (
        optional_col is not None
        and
        optional_col not in needed_cols
    ):

        needed_cols.append(
            optional_col
        )


# ============================================================
# READ RAW FILES IN CHUNKS
# ============================================================

frames = []

CHUNK_SIZE = 200000


for file in files:

    print(
        "\nReading:",
        file
    )


    reader = pd.read_csv(

        file,

        usecols=needed_cols,

        chunksize=CHUNK_SIZE,

        low_memory=False,

        dtype={
            SITE_COL:
                "string"
        }

    )


    for chunk_no, chunk in enumerate(
        reader,
        start=1
    ):

        original_chunk_n = len(
            chunk
        )


        # ====================================================
        # TURBIDITY ONLY
        # ====================================================

        turbidity_mask = (

            chunk[
                CHAR_COL
            ]

            .astype(str)

            .str.contains(
                "turbid",
                case=False,
                na=False
            )

        )


        chunk = (
            chunk.loc[
                turbidity_mask
            ]
            .copy()
        )


        # ====================================================
        # SAME TURBIDITY UNIT ONLY
        #
        # FNU gives us an apples-to-apples basin comparison.
        # ====================================================

        if UNIT_COL is not None:

            unit_clean = (

                chunk[
                    UNIT_COL
                ]

                .astype(str)

                .str.strip()

                .str.upper()

            )


            chunk = (
                chunk.loc[
                    unit_clean.eq(
                        "FNU"
                    )
                ]
                .copy()
            )


        # ====================================================
        # NUMERIC TARGET
        # ====================================================

        chunk[
            TARGET_COL
        ] = pd.to_numeric(

            chunk[
                TARGET_COL
            ],

            errors="coerce"

        )


        chunk = chunk.dropna(

            subset=[

                SITE_COL,

                SCENE_COL,

                HUC_COL,

                TARGET_COL,

            ]

        )


        # ====================================================
        # REMOVE IMPOSSIBLE NEGATIVE TURBIDITY
        # ====================================================

        chunk = chunk.loc[

            chunk[
                TARGET_COL
            ] >= 0

        ].copy()


        if len(chunk) > 0:

            frames.append(
                chunk
            )


        print(

            f"  chunk {chunk_no}: "

            f"{original_chunk_n:,} raw rows -> "

            f"{len(chunk):,} turbidity FNU rows"

        )


# ============================================================
# COMBINE TURBIDITY DATA
# ============================================================

if not frames:

    raise RuntimeError(
        "No turbidity FNU observations were found."
    )


df = pd.concat(
    frames,
    ignore_index=True
)


print(
    "\n"
    + "=" * 100
)

print(
    "TOTAL TURBIDITY FNU ROWS:",
    f"{len(df):,}"
)

print(
    "=" * 100
)


# ============================================================
# CHECK PARAMETER NAMES
# ============================================================

print(
    "\nTurbidity parameter names retained:"
)


print(

    df[
        CHAR_COL
    ]

    .astype(str)

    .value_counts()

    .head(20)

    .to_string()

)


if UNIT_COL is not None:

    print(
        "\nUnits retained:"
    )

    print(

        df[
            UNIT_COL
        ]

        .astype(str)

        .value_counts()

        .to_string()

    )


# ============================================================
# CLEAN HUC4
# ============================================================

df[
    "huc4_clean"
] = (

    df[
        HUC_COL
    ]

    .astype(str)

    .str.strip()

    .str.replace(
        r"\.0$",
        "",
        regex=True
    )

    .str.zfill(
        4
    )

)


df[
    "basin"
] = (

    df[
        "huc4_clean"
    ]

    .map(
        HUC_TO_BASIN
    )

)


# ============================================================
# SHOW HUC DISTRIBUTION
# ============================================================

print(
    "\nHUC4 distribution before basin filtering:"
)


print(

    df[
        "huc4_clean"
    ]

    .value_counts()

    .sort_index()

    .to_string()

)


# ============================================================
# UNMAPPED HUC4
# ============================================================

unmapped = (

    df.loc[
        df[
            "basin"
        ].isna(),
        "huc4_clean"
    ]

    .value_counts()

    .sort_index()

)


if len(
    unmapped
) > 0:

    print(
        "\nUnmapped HUC4 values:"
    )

    print(
        unmapped.to_string()
    )


# ============================================================
# KEEP ONLY FIVE STUDY BASINS
# ============================================================

df = (

    df.loc[
        df[
            "basin"
        ].notna()
    ]

    .copy()

)


print(
    "\nRows belonging to five target basin groups:",
    f"{len(df):,}"
)


# ============================================================
# CLEAN SITE / SCENE
# ============================================================

df[
    SITE_COL
] = (

    df[
        SITE_COL
    ]

    .astype(str)

    .str.strip()

)


df[
    SCENE_COL
] = (

    df[
        SCENE_COL
    ]

    .astype(str)

    .str.strip()

)


# ============================================================
# SCENE DATETIME
# ============================================================

def parse_scene_datetime(
    series
):

    extracted = (

        series

        .astype(str)

        .str.extract(

            r"(\d{8}T\d{6})",

            expand=False

        )

    )


    return pd.to_datetime(

        extracted,

        format="%Y%m%dT%H%M%S",

        utc=True,

        errors="coerce"

    )


if SCENE_DT_COL is not None:

    df[
        "_scene_datetime"
    ] = pd.to_datetime(

        df[
            SCENE_DT_COL
        ],

        utc=True,

        errors="coerce"

    )


else:

    print(
        "\nNo explicit scene datetime column found."
    )

    print(
        "Parsing acquisition datetime from scene IDs..."
    )


    df[
        "_scene_datetime"
    ] = parse_scene_datetime(

        df[
            SCENE_COL
        ]

    )


# ============================================================
# WATER QUALITY DATETIME / TIME DIFFERENCE
# ============================================================

if DELTA_COL is not None:

    print(
        "\nUsing existing temporal-difference column:",
        DELTA_COL
    )


    df[
        "_delta_minutes"
    ] = (

        pd.to_numeric(

            df[
                DELTA_COL
            ],

            errors="coerce"

        )

        .abs()

    )


else:

    print(
        "\nCalculating time difference from:"
    )

    print(
        " Measurement:",
        MEASUREMENT_DT_COL
    )

    print(
        " Scene:",
        SCENE_DT_COL
        if SCENE_DT_COL is not None
        else "parsed from scene ID"
    )


    df[
        "_measurement_datetime"
    ] = pd.to_datetime(

        df[
            MEASUREMENT_DT_COL
        ],

        utc=True,

        errors="coerce"

    )


    df[
        "_delta_minutes"
    ] = (

        (

            df[
                "_measurement_datetime"
            ]

            -

            df[
                "_scene_datetime"
            ]

        )

        .dt.total_seconds()

        .abs()

        / 60.0

    )


# ============================================================
# DATETIME QC
# ============================================================

print(
    "\nMissing scene datetime:",
    int(
        df[
            "_scene_datetime"
        ].isna().sum()
    )
)


print(
    "Missing temporal difference:",
    int(
        df[
            "_delta_minutes"
        ].isna().sum()
    )
)


# ============================================================
# RAW BASIN COUNTS
# ============================================================

raw_counts = (

    df

    .groupby(
        "basin"
    )

    .agg(

        raw_turbidity_rows=(
            TARGET_COL,
            "size"
        ),

        n_stations_raw=(
            SITE_COL,
            "nunique"
        ),

    )

)


# ============================================================
# UNIQUE STATION-SCENE COUNT
# ============================================================

unique_scene_counts = (

    df[
        [
            "basin",
            SITE_COL,
            SCENE_COL,
        ]
    ]

    .drop_duplicates()

    .groupby(
        "basin"
    )

    .size()

    .rename(
        "unique_station_scenes"
    )

)


# ============================================================
# ONE TURBIDITY OBSERVATION PER STATION + SCENE
#
# Pick the observation closest to the Sentinel-2 acquisition.
# ============================================================

candidate = (

    df

    .dropna(
        subset=[
            "_delta_minutes"
        ]
    )

    .sort_values(

        [

            SITE_COL,

            SCENE_COL,

            "_delta_minutes",

        ]

    )

    .drop_duplicates(

        subset=[

            SITE_COL,

            SCENE_COL,

        ],

        keep="first"

    )

    .copy()

)


print(
    "\nUnique station-scene candidates after closest-time selection:",
    f"{len(candidate):,}"
)


# ============================================================
# <= 30 MINUTE MATCHES
# ============================================================

candidate_30 = (

    candidate.loc[

        candidate[
            "_delta_minutes"
        ] <= 30

    ]

    .copy()

)


print(
    "Candidates within 30 minutes:",
    f"{len(candidate_30):,}"
)


# ============================================================
# CONVERT REFLECTANCE TO NUMERIC
# ============================================================

for col in REFLECTANCE_COLS:

    candidate_30[
        col
    ] = pd.to_numeric(

        candidate_30[
            col
        ],

        errors="coerce"

    )


# ============================================================
# REFLECTANCE QC
#
# For screening only:
#
# Require all available 250 m median bands to be:
#   - finite
#   - >= 0
#
# This closely approximates the current Delaware modeling
# workflow without rerunning the full pipeline for every basin.
# ============================================================

reflectance_matrix = (

    candidate_30[
        REFLECTANCE_COLS
    ]

)


finite_mask = pd.Series(

    np.isfinite(
        reflectance_matrix.to_numpy(
            dtype=float
        )
    ).all(
        axis=1
    ),

    index=candidate_30.index

)


nonnegative_mask = (

    reflectance_matrix
    .ge(0)
    .all(
        axis=1
    )

)


not_missing_mask = (

    reflectance_matrix
    .notna()
    .all(
        axis=1
    )

)


complete_reflectance = (

    not_missing_mask

    &

    finite_mask

    &

    nonnegative_mask

)


ready_all = (

    candidate_30.loc[
        complete_reflectance
    ]

    .copy()

)


print(
    "Complete 250 m reflectance:",
    f"{len(ready_all):,}"
)


# ============================================================
# IDENTIFY RIVER / FLOWLINE OBSERVATIONS
# ============================================================

if FEATURE_TYPE_COL is not None:

    print(
        "\nNHD feature-type values:"
    )


    print(

        ready_all[
            FEATURE_TYPE_COL
        ]

        .astype(str)

        .value_counts()

        .head(20)

        .to_string()

    )


    ready_all[
        "_is_river"
    ] = (

        ready_all[
            FEATURE_TYPE_COL
        ]

        .astype(str)

        .str.contains(
            "flowline",
            case=False,
            na=False
        )

    )


    river_method = (

        f"{FEATURE_TYPE_COL} contains 'Flowline'"

    )


else:

    # ========================================================
    # FALLBACK ONLY
    # ========================================================

    if STATION_COL is None:

        ready_all[
            "_is_river"
        ] = True


        river_method = (

            "No NHD feature-type or station-name column; "
            "all observations treated as potential river sites."

        )


    else:

        river_pattern = (

            r"river|creek|stream|branch|fork|"
            r"brook|run|channel"

        )


        ready_all[
            "_is_river"
        ] = (

            ready_all[
                STATION_COL
            ]

            .astype(str)

            .str.contains(
                river_pattern,
                case=False,
                regex=True,
                na=False
            )

        )


        river_method = (
            "Station-name keyword fallback"
        )


print(
    "\nRiver identification method:",
    river_method
)


ready_river = (

    ready_all.loc[
        ready_all[
            "_is_river"
        ]
    ]

    .copy()

)


print(
    "Model-ready river observations:",
    f"{len(ready_river):,}"
)


# ============================================================
# HELPER FOR PERCENTILES
# ============================================================

def percentile(
    q
):

    def f(
        x
    ):

        return x.quantile(
            q
        )

    return f


# ============================================================
# STAGE COUNTS
# ============================================================

stage_30 = (

    candidate_30

    .groupby(
        "basin"
    )

    .size()

    .rename(
        "n_within_30min"
    )

)


stage_complete = (

    ready_all

    .groupby(
        "basin"
    )

    .size()

    .rename(
        "n_complete_250m"
    )

)


stage_river = (

    ready_river

    .groupby(
        "basin"
    )

    .size()

    .rename(
        "n_model_ready_river"
    )

)


river_sites = (

    ready_river

    .groupby(
        "basin"
    )[
        SITE_COL
    ]

    .nunique()

    .rename(
        "n_river_stations"
    )

)


# ============================================================
# TURBIDITY STATISTICS
# ============================================================

target_stats = (

    ready_river

    .groupby(
        "basin"
    )[
        TARGET_COL
    ]

    .agg(

        turbidity_mean=
            "mean",

        turbidity_sd=
            "std",

        turbidity_median=
            "median",

        turbidity_min=
            "min",

        turbidity_max=
            "max",

        turbidity_p90=
            percentile(
                0.90
            ),

        turbidity_p95=
            percentile(
                0.95
            ),

        turbidity_p99=
            percentile(
                0.99
            ),

    )

)


# ============================================================
# UPPER-TAIL COUNTS
# ============================================================

tail_rows = []


for basin, g in ready_river.groupby(
    "basin"
):

    y = pd.to_numeric(

        g[
            TARGET_COL
        ],

        errors="coerce"

    )


    tail_rows.append({

        "basin":
            basin,

        "n_gt_10":
            int(
                (
                    y > 10
                ).sum()
            ),

        "n_gt_20":
            int(
                (
                    y > 20
                ).sum()
            ),

        "n_gt_50":
            int(
                (
                    y > 50
                ).sum()
            ),

        "n_gt_100":
            int(
                (
                    y > 100
                ).sum()
            ),

        "pct_gt_20":
            (
                100
                *
                float(
                    (
                        y > 20
                    ).mean()
                )
            ),

        "pct_gt_50":
            (
                100
                *
                float(
                    (
                        y > 50
                    ).mean()
                )
            ),

    })


tail = (

    pd.DataFrame(
        tail_rows
    )

    .set_index(
        "basin"
    )

)


# ============================================================
# STATION BALANCE
# ============================================================

balance_rows = []


for basin, g in ready_river.groupby(
    "basin"
):

    station_counts = (

        g

        .groupby(
            SITE_COL
        )

        .size()

        .sort_values(
            ascending=False
        )

    )


    n_total = int(
        station_counts.sum()
    )


    if (
        n_total > 0
        and
        len(
            station_counts
        ) > 0
    ):

        largest_n = int(
            station_counts.iloc[
                0
            ]
        )


        largest_share = (

            100

            *

            largest_n

            /

            n_total

        )


    else:

        largest_n = 0

        largest_share = (
            np.nan
        )


    balance_rows.append({

        "basin":
            basin,

        "largest_station_n":
            largest_n,

        "largest_station_share_pct":
            largest_share,

        "n_stations_25plus":
            int(
                (
                    station_counts
                    >= 25
                ).sum()
            ),

        "n_stations_50plus":
            int(
                (
                    station_counts
                    >= 50
                ).sum()
            ),

        "n_stations_100plus":
            int(
                (
                    station_counts
                    >= 100
                ).sum()
            ),

    })


balance = (

    pd.DataFrame(
        balance_rows
    )

    .set_index(
        "basin"
    )

)


# ============================================================
# TEMPORAL COVERAGE
# ============================================================

date_rows = []


for basin, g in ready_river.groupby(
    "basin"
):

    valid_dates = (

        g[
            "_scene_datetime"
        ]

        .dropna()

    )


    if len(
        valid_dates
    ) > 0:

        first_scene = (

            valid_dates
            .min()
            .strftime(
                "%Y-%m-%d"
            )

        )


        last_scene = (

            valid_dates
            .max()
            .strftime(
                "%Y-%m-%d"
            )

        )


        n_unique_dates = (

            valid_dates

            .dt.date

            .nunique()

        )


    else:

        first_scene = None

        last_scene = None

        n_unique_dates = 0


    date_rows.append({

        "basin":
            basin,

        "first_scene":
            first_scene,

        "last_scene":
            last_scene,

        "n_unique_dates":
            n_unique_dates,

    })


dates = (

    pd.DataFrame(
        date_rows
    )

    .set_index(
        "basin"
    )

)


# ============================================================
# COMBINE SUMMARY
# ============================================================

summary = (

    raw_counts

    .join(
        unique_scene_counts,
        how="outer"
    )

    .join(
        stage_30,
        how="outer"
    )

    .join(
        stage_complete,
        how="outer"
    )

    .join(
        stage_river,
        how="outer"
    )

    .join(
        river_sites,
        how="outer"
    )

    .join(
        target_stats,
        how="outer"
    )

    .join(
        tail,
        how="outer"
    )

    .join(
        balance,
        how="outer"
    )

    .join(
        dates,
        how="outer"
    )

)


# ============================================================
# SURVIVAL RATES
# ============================================================

summary[
    "pct_unique_survive_30min"
] = (

    100

    *

    summary[
        "n_within_30min"
    ]

    /

    summary[
        "unique_station_scenes"
    ]

)


summary[
    "pct_30min_survive_reflectance"
] = (

    100

    *

    summary[
        "n_complete_250m"
    ]

    /

    summary[
        "n_within_30min"
    ]

)


summary[
    "pct_complete_that_are_river"
] = (

    100

    *

    summary[
        "n_model_ready_river"
    ]

    /

    summary[
        "n_complete_250m"
    ]

)


summary[
    "turbidity_range"
] = (

    summary[
        "turbidity_max"
    ]

    -

    summary[
        "turbidity_min"
    ]

)


# ============================================================
# REORDER FIVE BASINS
# ============================================================

summary = summary.reindex(
    EXPECTED_BASINS
)


# ============================================================
# FINAL DISPLAY COLUMNS
# ============================================================

display_cols = [

    "n_river_stations",

    "unique_station_scenes",

    "n_within_30min",

    "n_model_ready_river",

    "pct_30min_survive_reflectance",

    "turbidity_mean",

    "turbidity_sd",

    "turbidity_median",

    "turbidity_p90",

    "turbidity_p95",

    "turbidity_max",

    "n_gt_20",

    "n_gt_50",

    "n_gt_100",

    "n_stations_50plus",

    "n_stations_100plus",

    "largest_station_share_pct",

]


print(
    "\n"
    + "=" * 100
)

print(
    "FINAL FIVE-BASIN SCREENING"
)

print(
    "=" * 100
)


print(

    summary[
        display_cols
    ]

    .round(
        2
    )

    .to_string()

)


# ============================================================
# STATION-LEVEL SUMMARY
# ============================================================

station_rows = []


for (
    basin,
    site
), g in ready_river.groupby(

    [
        "basin",
        SITE_COL,
    ]

):

    y = pd.to_numeric(

        g[
            TARGET_COL
        ],

        errors="coerce"

    )


    if STATION_COL is not None:

        names = (

            g[
                STATION_COL
            ]

            .dropna()

            .astype(str)

        )


        station_name = (

            names.iloc[
                0
            ]

            if len(
                names
            ) > 0

            else ""

        )


    else:

        station_name = ""


    station_rows.append({

        "basin":
            basin,

        "site_no":
            site,

        "station_nm":
            station_name,

        "n":
            len(
                g
            ),

        "mean":
            y.mean(),

        "sd":
            y.std(),

        "median":
            y.median(),

        "p90":
            y.quantile(
                0.90
            ),

        "p95":
            y.quantile(
                0.95
            ),

        "p99":
            y.quantile(
                0.99
            ),

        "max":
            y.max(),

        "n_gt_10":
            int(
                (
                    y > 10
                ).sum()
            ),

        "n_gt_20":
            int(
                (
                    y > 20
                ).sum()
            ),

        "n_gt_50":
            int(
                (
                    y > 50
                ).sum()
            ),

        "n_gt_100":
            int(
                (
                    y > 100
                ).sum()
            ),

    })


station_summary = pd.DataFrame(
    station_rows
)


if len(
    station_summary
) > 0:

    station_summary = (

        station_summary

        .sort_values(

            [
                "basin",
                "n",
            ],

            ascending=[
                True,
                False,
            ]

        )

    )


# ============================================================
# SAVE MODEL-READY SCREENING ROWS
# ============================================================

ready_export_cols = [

    "basin",

    "huc4_clean",

    SITE_COL,

    SCENE_COL,

    TARGET_COL,

    "_delta_minutes",

    "_scene_datetime",

]


if CHAR_COL is not None:

    ready_export_cols.append(
        CHAR_COL
    )


if UNIT_COL is not None:

    ready_export_cols.append(
        UNIT_COL
    )


if STATION_COL is not None:

    ready_export_cols.append(
        STATION_COL
    )


if FEATURE_TYPE_COL is not None:

    ready_export_cols.append(
        FEATURE_TYPE_COL
    )


ready_export_cols += (
    REFLECTANCE_COLS
)


ready_export_cols = list(
    dict.fromkeys(
        ready_export_cols
    )
)


ready_river[
    ready_export_cols
].to_csv(

    READY_OUT,

    index=False

)


# ============================================================
# SAVE TABLES
# ============================================================

summary.to_csv(
    SUMMARY_OUT
)


station_summary.to_csv(

    STATION_OUT,

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
    SUMMARY_OUT
)

print(
    STATION_OUT
)

print(
    READY_OUT
)


# ============================================================
# TOP STATIONS IN EACH BASIN
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TOP STATIONS WITHIN EACH BASIN"
)

print(
    "=" * 100
)


for basin in EXPECTED_BASINS:

    if len(
        station_summary
    ) == 0:

        temp = pd.DataFrame()

    else:

        temp = (

            station_summary.loc[

                station_summary[
                    "basin"
                ]
                == basin

            ]

            .head(
                12
            )

        )


    print(
        f"\n--- {basin} ---"
    )


    if len(
        temp
    ) == 0:

        print(
            "No model-ready river observations."
        )


    else:

        print(

            temp[
                [
                    "site_no",
                    "station_nm",
                    "n",
                    "mean",
                    "sd",
                    "median",
                    "p95",
                    "max",
                    "n_gt_20",
                    "n_gt_50",
                    "n_gt_100",
                ]
            ]

            .round(
                2
            )

            .to_string(
                index=False
            )

        )


# ============================================================
# SIMPLE DATA QUALITY NOTES
#
# NOT a ranking.
# Just print useful facts for comparison.
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "BASIN DATA CHARACTERISTICS"
)

print(
    "=" * 100
)


for basin in EXPECTED_BASINS:

    if (
        basin not in summary.index
        or
        pd.isna(
            summary.loc[
                basin,
                "n_model_ready_river"
            ]
        )
    ):

        print(
            f"\n{basin}: no usable river dataset."
        )

        continue


    row = summary.loc[
        basin
    ]


    print(
        f"\n{basin}"
    )

    print(
        "  Model-ready river observations:",
        int(
            row[
                "n_model_ready_river"
            ]
        )
    )

    print(
        "  River stations:",
        int(
            row[
                "n_river_stations"
            ]
        )
    )

    print(
        "  Turbidity median:",
        round(
            row[
                "turbidity_median"
            ],
            2
        )
    )

    print(
        "  Turbidity SD:",
        round(
            row[
                "turbidity_sd"
            ],
            2
        )
    )

    print(
        "  Turbidity p95:",
        round(
            row[
                "turbidity_p95"
            ],
            2
        )
    )

    print(
        "  Turbidity max:",
        round(
            row[
                "turbidity_max"
            ],
            2
        )
    )

    print(
        "  Observations >20 FNU:",
        int(
            row[
                "n_gt_20"
            ]
        )
    )

    print(
        "  Observations >50 FNU:",
        int(
            row[
                "n_gt_50"
            ]
        )
    )

    print(
        "  Stations with >=50 observations:",
        int(
            row[
                "n_stations_50plus"
            ]
        )
    )

    print(
        "  Largest station share:",
        f"{row['largest_station_share_pct']:.1f}%"
    )


print(
    "\n"
    + "=" * 100
)

print(
    "SCREENING COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nDo not choose a basin from maximum turbidity alone."
)

print(
    "Look for the combination of:"
)

print(
    "  - substantially larger model-ready n"
)

print(
    "  - several well-sampled river stations"
)

print(
    "  - substantially greater turbidity SD / upper-tail coverage"
)

print(
    "  - no extreme dependence on one single station"
)
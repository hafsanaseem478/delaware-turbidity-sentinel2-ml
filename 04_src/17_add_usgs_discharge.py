from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

RAW_Q_OUTPUT = Path(
    "01_data/processed/usgs_daily_discharge_00060.csv"
)

MERGED_OUTPUT = Path(
    "01_data/processed/delaware_with_discharge.csv"
)

COVERAGE_OUTPUT = Path(
    "05_results/tables/17_discharge_coverage.csv"
)

RAW_Q_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

COVERAGE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CURRENT 1402-ROW MODELING DATASET
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={
        "site_no": "string"
    }
)

df["scene_datetime_UTC"] = pd.to_datetime(
    df["scene_datetime_UTC"],
    utc=True,
    errors="coerce"
)

if df["scene_datetime_UTC"].isna().any():

    raise ValueError(
        "Some scene_datetime_UTC values could not be parsed."
    )


# Scene calendar date
df["scene_date"] = (
    df["scene_datetime_UTC"]
    .dt.tz_convert(None)
    .dt.normalize()
)


sites = sorted(
    df["site_no"]
    .dropna()
    .astype(str)
    .unique()
)


print("=" * 90)
print("USGS DISCHARGE DOWNLOAD")
print("=" * 90)

print(
    "\nModeling rows:",
    len(df)
)

print(
    "Sites:",
    len(sites)
)

for site in sites:
    print(" -", site)


# ============================================================
# STUDY DATE RANGE
# ============================================================

START_DATE = (
    df["scene_date"].min()
    - pd.Timedelta(days=10)
)

END_DATE = (
    df["scene_date"].max()
    + pd.Timedelta(days=1)
)


print(
    "\nDownload range:",
    START_DATE.strftime("%Y-%m-%d"),
    "to",
    END_DATE.strftime("%Y-%m-%d")
)


# ============================================================
# USGS DAILY VALUES API
#
# Parameter 00060 = discharge
# Statistic 00003 = daily mean
# ============================================================

BASE_URL = (
    "https://api.waterdata.usgs.gov/"
    "ogcapi/v1/collections/daily/items"
)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
        "Delaware-Turbidity-Research/1.0"
})


# ============================================================
# ROBUST REQUEST FUNCTION
#
# Retries automatically when USGS times out.
# ============================================================

def request_with_retry(
    url,
    params=None,
    max_attempts=5
):

    for attempt in range(
        1,
        max_attempts + 1
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=300,
            )

            response.raise_for_status()

            return response


        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
        ) as e:

            print(
                f"    Attempt {attempt}/{max_attempts} failed:"
            )

            print(
                f"    {repr(e)}"
            )

            if attempt == max_attempts:
                raise

            wait_seconds = (
                5 * attempt
            )

            print(
                f"    Waiting {wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )


# ============================================================
# DOWNLOAD ONE DATE CHUNK
# ============================================================

def download_chunk(
    site_no,
    start_date,
    end_date
):

    params = {

        "f":
            "json",

        "limit":
            10000,

        "monitoring_location_id":
            f"USGS-{site_no}",

        "parameter_code":
            "00060",

        "statistic_id":
            "00003",

        "datetime":
            (
                f"{start_date}/"
                f"{end_date}"
            ),
    }


    all_features = []

    url = BASE_URL

    first_page = True


    while url:

        if first_page:

            response = request_with_retry(
                url,
                params=params
            )

            first_page = False

        else:

            # "next" URL already contains
            # pagination information
            response = request_with_retry(
                url
            )


        data = response.json()


        features = data.get(
            "features",
            []
        )

        all_features.extend(
            features
        )


        # ----------------------------------------------------
        # FOLLOW NEXT PAGE
        # ----------------------------------------------------

        next_url = None

        for link in data.get(
            "links",
            []
        ):

            if (
                link.get("rel")
                == "next"
            ):

                next_url = (
                    link.get("href")
                )

                break


        url = next_url


    return all_features


# ============================================================
# DOWNLOAD ONE SITE
#
# IMPORTANT:
# Downloads year by year.
#
# This avoids the large multi-year request that caused
# Trenton to time out previously.
# ============================================================

def download_site(
    site_no
):

    site_features = []


    first_year = (
        START_DATE.year
    )

    last_year = (
        END_DATE.year
    )


    for year in range(
        first_year,
        last_year + 1
    ):

        chunk_start = max(

            START_DATE,

            pd.Timestamp(
                year=year,
                month=1,
                day=1
            )
        )


        chunk_end = min(

            END_DATE,

            pd.Timestamp(
                year=year,
                month=12,
                day=31
            )
        )


        start_string = (
            chunk_start
            .strftime("%Y-%m-%d")
        )

        end_string = (
            chunk_end
            .strftime("%Y-%m-%d")
        )


        print(
            f"    {start_string} "
            f"to {end_string}"
        )


        try:

            features = download_chunk(
                site_no,
                start_string,
                end_string
            )

            print(
                "      rows:",
                len(features)
            )

            site_features.extend(
                features
            )


        except Exception as e:

            print(
                "      CHUNK FAILED:"
            )

            print(
                "     ",
                repr(e)
            )


        # polite pause
        time.sleep(
            0.25
        )


    # ========================================================
    # PARSE USGS RESPONSE
    # ========================================================

    rows = []


    for feature in site_features:

        properties = feature.get(
            "properties",
            {}
        )


        rows.append({

            "site_no":
                str(site_no),

            "date":
                properties.get(
                    "time"
                ),

            "discharge_cfs":
                properties.get(
                    "value"
                ),

            "unit":
                properties.get(
                    "unit_of_measure"
                ),

            "qualifier":
                str(
                    properties.get(
                        "qualifier"
                    )
                ),

            "approval_status":
                str(
                    properties.get(
                        "approval_status",
                        properties.get(
                            "approvals_status"
                        )
                    )
                ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# DOWNLOAD ALL SEVEN SITES
# ============================================================

q_frames = []


for site in sites:

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"Downloading discharge for {site}"
    )

    print(
        "-" * 90
    )


    try:

        q_site = download_site(
            site
        )


        print(
            "\nTotal rows returned:",
            len(q_site)
        )


        if len(q_site) > 0:

            q_frames.append(
                q_site
            )

        else:

            print(
                "No daily discharge data "
                "returned for this station."
            )


    except Exception as e:

        print(
            "\nSITE FAILED:"
        )

        print(
            repr(e)
        )


# ============================================================
# CHECK
# ============================================================

if not q_frames:

    raise RuntimeError(
        "USGS returned no discharge data "
        "for any station."
    )


# ============================================================
# COMBINE
# ============================================================

q = pd.concat(
    q_frames,
    ignore_index=True
)


print("\n" + "=" * 90)
print("RAW DISCHARGE DOWNLOAD COMPLETE")
print("=" * 90)

print(
    "Rows downloaded:",
    len(q)
)

print(
    "Stations with discharge:",
    q["site_no"].nunique()
)


print("\nRows by site:")

print(
    q["site_no"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# CLEAN
# ============================================================

q["date"] = pd.to_datetime(
    q["date"],
    errors="coerce",
    utc=True
)


# Convert to timezone-naive calendar date
q["date"] = (
    q["date"]
    .dt.tz_convert(None)
    .dt.normalize()
)


q["discharge_cfs"] = pd.to_numeric(
    q["discharge_cfs"],
    errors="coerce"
)


q = q.dropna(
    subset=[
        "date",
        "discharge_cfs",
    ]
).copy()


# ============================================================
# CHECK DUPLICATE SITE-DATE VALUES
# ============================================================

duplicate_count = int(

    q.duplicated(
        subset=[
            "site_no",
            "date",
        ],
        keep=False
    ).sum()
)


print(
    "\nDuplicate site-date discharge rows:",
    duplicate_count
)


# If overlapping records exist,
# use median for that site-day.

q = (

    q.groupby(
        [
            "site_no",
            "date",
        ],
        as_index=False
    )

    .agg(

        discharge_cfs=(
            "discharge_cfs",
            "median"
        )

    )

)


# ============================================================
# ENGINEER HYDROLOGIC FEATURES
# ============================================================

feature_frames = []


for site_no, site_q in q.groupby(
    "site_no"
):

    print(
        "\nEngineering hydrology for:",
        site_no
    )


    site_q = (

        site_q
        .sort_values(
            "date"
        )
        .set_index(
            "date"
        )

    )


    # --------------------------------------------------------
    # FULL DAILY CALENDAR
    # --------------------------------------------------------

    full_index = pd.date_range(

        start=site_q.index.min(),

        end=site_q.index.max(),

        freq="D"

    )


    site_q = site_q.reindex(
        full_index
    )


    site_q.index.name = (
        "date"
    )


    site_q["site_no"] = (
        str(site_no)
    )


    # ========================================================
    # CURRENT DAILY DISCHARGE
    #
    # Keep it in the file for future experiments,
    # although our first conservative test will primarily
    # use antecedent values.
    # ========================================================

    site_q[
        "Q_current_cfs"
    ] = (
        site_q[
            "discharge_cfs"
        ]
    )


    # ========================================================
    # PREVIOUS-DAY DISCHARGE
    # ========================================================

    site_q[
        "Q_lag1_cfs"
    ] = (

        site_q[
            "discharge_cfs"
        ]
        .shift(1)

    )


    site_q[
        "Q_lag2_cfs"
    ] = (

        site_q[
            "discharge_cfs"
        ]
        .shift(2)

    )


    # ========================================================
    # ANTECEDENT 3-DAY MEAN
    #
    # t-1, t-2, t-3
    # ========================================================

    antecedent = (

        site_q[
            "discharge_cfs"
        ]
        .shift(1)

    )


    site_q[
        "Q_pre3_mean_cfs"
    ] = (

        antecedent
        .rolling(
            window=3,
            min_periods=3
        )
        .mean()

    )


    # ========================================================
    # ANTECEDENT 7-DAY MEAN
    #
    # Require at least 5 valid days.
    # ========================================================

    site_q[
        "Q_pre7_mean_cfs"
    ] = (

        antecedent
        .rolling(
            window=7,
            min_periods=5
        )
        .mean()

    )


    # ========================================================
    # DISCHARGE CHANGE
    # ========================================================

    site_q[
        "dQ_1d_cfs"
    ] = (

        site_q[
            "Q_lag1_cfs"
        ]

        -

        site_q[
            "Q_lag2_cfs"
        ]

    )


    # ========================================================
    # RISING / FALLING LIMB INDICATOR
    # ========================================================

    site_q[
        "Q_rising"
    ] = (

        site_q[
            "dQ_1d_cfs"
        ]
        > 0

    ).astype(float)


    # ========================================================
    # LOG TRANSFORMATIONS
    #
    # River discharge is highly skewed.
    # ========================================================

    site_q[
        "log_Q_current"
    ] = np.log1p(

        site_q[
            "Q_current_cfs"
        ]
        .clip(lower=0)

    )


    site_q[
        "log_Q_lag1"
    ] = np.log1p(

        site_q[
            "Q_lag1_cfs"
        ]
        .clip(lower=0)

    )


    site_q[
        "log_Q_pre3"
    ] = np.log1p(

        site_q[
            "Q_pre3_mean_cfs"
        ]
        .clip(lower=0)

    )


    site_q[
        "log_Q_pre7"
    ] = np.log1p(

        site_q[
            "Q_pre7_mean_cfs"
        ]
        .clip(lower=0)

    )


    feature_frames.append(

        site_q.reset_index()

    )


# ============================================================
# COMBINE ENGINEERED HYDROLOGY
# ============================================================

hydro = pd.concat(
    feature_frames,
    ignore_index=True
)


# ============================================================
# SAVE DISCHARGE TABLE
# ============================================================

hydro.to_csv(
    RAW_Q_OUTPUT,
    index=False
)


# ============================================================
# FEATURES TO MERGE
# ============================================================

HYDRO_FEATURES = [

    "Q_current_cfs",

    "Q_lag1_cfs",

    "Q_pre3_mean_cfs",

    "Q_pre7_mean_cfs",

    "dQ_1d_cfs",

    "Q_rising",

    "log_Q_current",

    "log_Q_lag1",

    "log_Q_pre3",

    "log_Q_pre7",

]


# ============================================================
# MERGE WITH EXISTING 1402 OBSERVATIONS
# ============================================================

merged = df.merge(

    hydro[
        [
            "site_no",
            "date",
        ]
        + HYDRO_FEATURES
    ],

    left_on=[
        "site_no",
        "scene_date",
    ],

    right_on=[
        "site_no",
        "date",
    ],

    how="left",

    validate="many_to_one",
)


# ============================================================
# SANITY CHECK
#
# Merge must NOT change the number of modeling observations.
# ============================================================

if len(merged) != len(df):

    raise RuntimeError(
        "Merge changed row count. "
        f"Before={len(df)}, "
        f"After={len(merged)}"
    )


print(
    "\nMerge row-count check: PASSED"
)


# ============================================================
# COMPLETE-HYDROLOGY DEFINITION
#
# For the first experiment, require:
#
# lag1
# 3-day mean
# 7-day mean
# dQ
# rising/falling
#
# Current-day Q is NOT required here.
# ============================================================

MODEL_HYDRO_FEATURES = [

    "log_Q_lag1",

    "log_Q_pre3",

    "log_Q_pre7",

    "dQ_1d_cfs",

    "Q_rising",

]


complete_hydrology = (

    merged[
        MODEL_HYDRO_FEATURES
    ]

    .notna()

    .all(
        axis=1
    )

)


# ============================================================
# COVERAGE BY STATION
# ============================================================

coverage_rows = []


for (
    site_no,
    station_nm
), group in merged.groupby(

    [
        "site_no",
        "station_nm",
    ],

    dropna=False

):

    complete = (

        group[
            MODEL_HYDRO_FEATURES
        ]

        .notna()

        .all(
            axis=1
        )

    )


    coverage_rows.append({

        "site_no":
            site_no,

        "station_nm":
            station_nm,

        "n_total":
            len(group),

        "n_complete_hydrology":
            int(
                complete.sum()
            ),

        "coverage_pct":
            100
            * complete.mean(),

    })


coverage = pd.DataFrame(
    coverage_rows
)


coverage = coverage.sort_values(
    [
        "n_complete_hydrology",
        "site_no",
    ],
    ascending=[
        False,
        True,
    ]
)


# ============================================================
# PRINT COVERAGE
# ============================================================

print("\n" + "=" * 90)
print("DISCHARGE COVERAGE")
print("=" * 90)


print(

    coverage.to_string(

        index=False,

        float_format=lambda x:
            f"{x:.1f}"

    )

)


print(
    "\nTotal modeling rows:",
    len(merged)
)


print(
    "Rows with complete hydrology:",
    int(
        complete_hydrology.sum()
    )
)


print(
    "Overall coverage:",
    f"{100 * complete_hydrology.mean():.1f}%"
)


# ============================================================
# SHOW COMPLETE OBSERVATIONS BY STATION
# ============================================================

print(
    "\nComplete hydrology observations by station:"
)


print(

    merged.loc[
        complete_hydrology,
        "station_nm"
    ]

    .value_counts()

    .to_string()

)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    MERGED_OUTPUT,
    index=False
)


coverage.to_csv(
    COVERAGE_OUTPUT,
    index=False
)


print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    RAW_Q_OUTPUT
)

print(
    MERGED_OUTPUT
)

print(
    COVERAGE_OUTPUT
)


print(
    "\nNext step: if Trenton discharge was recovered "
    "and complete hydrology coverage is around 800+ rows, "
    "run 18_test_discharge_gain.py."
)
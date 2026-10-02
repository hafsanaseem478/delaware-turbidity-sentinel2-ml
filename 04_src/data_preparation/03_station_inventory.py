from pathlib import Path
import pandas as pd

RAW_DIR = Path("01_data/raw")

FILES = sorted(
    RAW_DIR.rglob("Matched_WQ_S2*.csv")
)

if not FILES:
    raise FileNotFoundError(
        "No Matched_WQ_S2 CSV files found anywhere inside 01_data/raw"
    )

print("Files found:")

for file in FILES:
    print(" -", file)

frames = []

for file in FILES:

    print("Reading:", file)

    temp = pd.read_csv(
        file,
        low_memory=False,
        dtype={
            "site_no": "string",
            "huc4": "string",
        }
    )

    frames.append(temp)

df = pd.concat(
    frames,
    ignore_index=True
)

df["huc4_clean"] = (
    df["huc4"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(4)
)

print("\nHUC4 values found:")
print(
    df["huc4_clean"]
    .value_counts()
    .head(20)
)

delaware = df[
    (
        df["huc4_clean"] == "0204"
    )
    &
    (
        df["parm_nm"]
        .astype(str)
        .str.contains(
            "turbid",
            case=False,
            na=False
        )
    )
].copy()

print("\nDelaware turbidity rows:", len(delaware))
print(
    "Delaware stations:",
    delaware["site_no"].nunique()
)

delaware[
    "scene_datetime_UTC"
] = pd.to_datetime(
    delaware["scene_datetime_UTC"],
    errors="coerce"
)

summary = (

    delaware.groupby(
        [
            "site_no",
            "station_nm",
        ],
        dropna=False
    )

    .agg(

        raw_rows=(
            "site_no",
            "size"
        ),

        unique_scenes=(
            "scene",
            "nunique"
        ),

        mean_turbidity=(
            "MeasurementValue",
            "mean"
        ),

        median_turbidity=(
            "MeasurementValue",
            "median"
        ),

        std_turbidity=(
            "MeasurementValue",
            "std"
        ),

        max_turbidity=(
            "MeasurementValue",
            "max"
        ),

        min_turbidity=(
            "MeasurementValue",
            "min"
        ),

        first_scene=(
            "scene_datetime_UTC",
            "min"
        ),

        last_scene=(
            "scene_datetime_UTC",
            "max"
        ),

    )

    .reset_index()

    .sort_values(
        "unique_scenes",
        ascending=False
    )
)

print("\n" + "=" * 110)

print(
    "ALL DELAWARE BASIN TURBIDITY STATIONS"
)

print("=" * 110)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nNumber of stations:",
    len(summary)
)

print(
    "Total unique station-scene combinations:",
    delaware[
        ["site_no", "scene"]
    ]
    .drop_duplicates()
    .shape[0]
)

Path(
    "05_results/tables"
).mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    "05_results/tables/"
    "02b_delaware_station_inventory.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "05_results/tables/"
    "02b_delaware_station_inventory.csv"
)


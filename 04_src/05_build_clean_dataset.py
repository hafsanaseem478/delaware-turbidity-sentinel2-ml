from pathlib import Path
import pandas as pd

INPUT = Path(
    "01_data/processed/delaware_basin_candidate_scenes.csv"
)

OUTPUT = Path(
    "01_data/processed/delaware_turbidity_clean.csv"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

# --------------------------------------------------
# Sentinel-2 bands
# --------------------------------------------------

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

center_cols = [
    f"{band}_center"
    for band in BANDS
]

# --------------------------------------------------
# Make important fields numeric
# --------------------------------------------------

df["MeasurementValue"] = pd.to_numeric(
    df["MeasurementValue"],
    errors="coerce"
)

df["time_diff_minutes"] = pd.to_numeric(
    df["time_diff_minutes"],
    errors="coerce"
)

df["l2flag_center"] = pd.to_numeric(
    df["l2flag_center"],
    errors="coerce"
)

print("=" * 80)
print("STARTING DATASET")
print("=" * 80)

print("Samples:", len(df))

# --------------------------------------------------
# Inspect ACOLITE flags BEFORE filtering
# --------------------------------------------------

print("\nACOLITE center-pixel flag values:")

print(
    df["l2flag_center"]
    .value_counts(dropna=False)
    .sort_index()
)

# --------------------------------------------------
# FILTER 1:
# Ground measurement must be within 30 min
# of satellite acquisition
# --------------------------------------------------

df = df[
    df["time_diff_minutes"] <= 30
].copy()

print("\nAfter 30-minute temporal filter:")
print(len(df))

# --------------------------------------------------
# FILTER 2:
# ACOLITE center pixel must be unflagged
#
# 0 = no ACOLITE quality flag
# --------------------------------------------------

df = df[
    df["l2flag_center"] == 0
].copy()

print("\nAfter ACOLITE quality-flag filter:")
print(len(df))

# --------------------------------------------------
# FILTER 3:
# Turbidity must exist and cannot be negative
# --------------------------------------------------

df = df[
    df["MeasurementValue"].notna()
    & (df["MeasurementValue"] >= 0)
].copy()

print("\nAfter turbidity QC:")
print(len(df))

# --------------------------------------------------
# Make sure all center bands exist
# --------------------------------------------------

df = df.dropna(
    subset=center_cols
).copy()

print("\nAfter requiring all Sentinel-2 bands:")
print(len(df))

# --------------------------------------------------
# Convert USGS stored digital numbers
# to actual ACOLITE aquatic reflectance
#
# USGS stores aquatic reflectance × 10,000
# --------------------------------------------------

for band in BANDS:

    source = f"{band}_center"
    output = f"{band}_rhow"

    df[output] = (
        pd.to_numeric(df[source], errors="coerce")
        / 10000.0
    )

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("FINAL CLEAN DATASET")
print("=" * 80)

print("\nTotal samples:")
print(len(df))

print("\nSamples per station:")

station_summary = (
    df.groupby(["site_no", "station_nm"])
    .agg(
        samples=("scene", "size"),
        median_turbidity=("MeasurementValue", "median"),
        mean_turbidity=("MeasurementValue", "mean"),
        max_turbidity=("MeasurementValue", "max"),
    )
    .sort_values("samples", ascending=False)
)

print(station_summary.to_string())

print("\nTurbidity distribution:")

print(
    df["MeasurementValue"].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print("\nAquatic reflectance ranges:")

rhow_cols = [
    f"{band}_rhow"
    for band in BANDS
]

print(
    df[rhow_cols]
    .describe()
    .T[
        [
            "min",
            "25%",
            "50%",
            "75%",
            "max"
        ]
    ]
)

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
from pathlib import Path
import pandas as pd

INPUT = Path(
    "01_data/processed/delaware_basin_candidate_scenes.csv"
)

OUTPUT = Path(
    "01_data/processed/delaware_turbidity_modeling_base.csv"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

# Sentinel-2 bands available from USGS ACOLITE dataset
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

median_cols = [
    f"{band}_buf250_med"
    for band in BANDS
]

# --------------------------------------------------
# Convert important fields
# --------------------------------------------------

df["MeasurementValue"] = pd.to_numeric(
    df["MeasurementValue"],
    errors="coerce"
)

df["time_diff_minutes"] = pd.to_numeric(
    df["time_diff_minutes"],
    errors="coerce"
)

df["n_mask"] = pd.to_numeric(
    df["n_mask"],
    errors="coerce"
)

print("=" * 80)
print("START")
print("=" * 80)

print("Candidate samples:", len(df))

# --------------------------------------------------
# 1. Require ground measurement within 30 minutes
# --------------------------------------------------

df = df[
    df["time_diff_minutes"] <= 30
].copy()

print("\nAfter 30-minute temporal filter:")
print(len(df))

# --------------------------------------------------
# 2. Require valid 250 m median for EVERY band
# --------------------------------------------------

df = df.dropna(
    subset=median_cols
).copy()

print("\nAfter requiring complete 250 m median reflectance:")
print(len(df))

# --------------------------------------------------
# 3. Turbidity QC
# --------------------------------------------------

df = df[
    df["MeasurementValue"].notna()
    & (df["MeasurementValue"] >= 0)
].copy()

print("\nAfter turbidity QC:")
print(len(df))

# --------------------------------------------------
# Convert ACOLITE DN → aquatic reflectance
# DN / 10,000
# --------------------------------------------------

for band in BANDS:

    source = f"{band}_buf250_med"
    output = f"{band}_rhow"

    df[output] = (
        pd.to_numeric(
            df[source],
            errors="coerce"
        )
        / 10000.0
    )

rhow_cols = [
    f"{band}_rhow"
    for band in BANDS
]

# --------------------------------------------------
# Check for negative median reflectance
# Do NOT silently alter it
# --------------------------------------------------

negative_rows = (
    df[rhow_cols] < 0
).any(axis=1)

print("\nSamples containing any negative median reflectance:")
print(negative_rows.sum())

# For our primary modelling dataset,
# remove negative-reflectance samples if any exist.

df = df[
    ~negative_rows
].copy()

print("\nAfter negative-reflectance QC:")
print(len(df))

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("FINAL MODELLING BASE")
print("=" * 80)

print("\nTotal samples:")
print(len(df))

print("\nSamples per station:")

summary = (
    df.groupby(["site_no", "station_nm"])
    .agg(
        samples=("scene", "size"),
        median_turbidity=("MeasurementValue", "median"),
        mean_turbidity=("MeasurementValue", "mean"),
        max_turbidity=("MeasurementValue", "max"),
        median_valid_pixels=("n_mask", "median"),
        min_valid_pixels=("n_mask", "min"),
    )
    .sort_values(
        "samples",
        ascending=False
    )
)

print(summary.to_string())

print("\nTurbidity distribution:")

print(
    df["MeasurementValue"].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

print("\nAquatic reflectance ranges:")

print(
    df[rhow_cols]
    .describe()
    .T[
        [
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]
)

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
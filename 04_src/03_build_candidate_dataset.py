from pathlib import Path
import pandas as pd

INPUT = Path("01_data/processed/delaware_turbidity_matched.csv")
OUTPUT = Path("01_data/processed/delaware_basin_candidate_scenes.csv")

# River stations only.
# We are NOT using lake sites / small coastal creeks.
STATIONS = {
    "01463500": "Delaware River at Trenton NJ",
    "014670261": "Delaware River at Pennypack Woods PA",
    "01458500": "Delaware River at Frenchtown NJ",
    "01473500": "Schuylkill River at Norristown PA",
    "01438500": "Delaware River at Montague NJ",
    "01467200": "Delaware River at Penn's Landing Philadelphia PA",
    "01427207": "Delaware River at Lordville NY",
}

# ---------------------------------------------------------
# Load Delaware turbidity data
# ---------------------------------------------------------
df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

df = df[df["site_no"].isin(STATIONS.keys())].copy()

print("=" * 80)
print("SELECTED RIVER STATIONS")
print("=" * 80)

print(df[["site_no", "station_nm"]]
      .drop_duplicates()
      .sort_values("site_no")
      .to_string(index=False))

# ---------------------------------------------------------
# Convert dates / measurements
# ---------------------------------------------------------
df["scene_datetime_UTC"] = pd.to_datetime(
    df["scene_datetime_UTC"],
    utc=True,
    errors="coerce"
)

df["sample_datetime_UTC"] = pd.to_datetime(
    df["sample_datetime_UTC"],
    utc=True,
    errors="coerce"
)

df["MeasurementValue"] = pd.to_numeric(
    df["MeasurementValue"],
    errors="coerce"
)

# Remove rows where essential information is missing
df = df.dropna(
    subset=[
        "scene",
        "scene_datetime_UTC",
        "sample_datetime_UTC",
        "MeasurementValue",
    ]
).copy()

# ---------------------------------------------------------
# How far is USGS measurement from satellite acquisition?
# ---------------------------------------------------------
df["time_diff_minutes"] = (
    (df["sample_datetime_UTC"] - df["scene_datetime_UTC"])
    .abs()
    .dt.total_seconds()
    / 60
)

# ---------------------------------------------------------
# ONE ground measurement per station + satellite scene
#
# For now:
# choose the turbidity measurement CLOSEST in time
# to the Sentinel-2 acquisition.
#
# This is NOT our final time-quality filtering yet.
# ---------------------------------------------------------
df = df.sort_values(
    ["site_no", "scene", "time_diff_minutes"]
)

candidate = (
    df.drop_duplicates(
        subset=["site_no", "scene"],
        keep="first"
    )
    .copy()
)

candidate.to_csv(OUTPUT, index=False)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("CANDIDATE STATION-SCENE DATASET")
print("=" * 80)

print("\nOriginal selected rows:")
print(len(df))

print("\nOne-row-per-station-scene samples:")
print(len(candidate))

print("\nSamples per station:")

station_summary = (
    candidate.groupby(["site_no", "station_nm"])
    .agg(
        samples=("scene", "size"),
        first_scene=("scene_datetime_UTC", "min"),
        last_scene=("scene_datetime_UTC", "max"),
        median_turbidity=("MeasurementValue", "median"),
        max_turbidity=("MeasurementValue", "max"),
        median_time_diff_min=("time_diff_minutes", "median"),
        p95_time_diff_min=(
            "time_diff_minutes",
            lambda x: x.quantile(0.95)
        ),
    )
    .sort_values("samples", ascending=False)
)

print(station_summary.to_string())

print("\n" + "=" * 80)
print("OVERALL TEMPORAL MATCH")
print("=" * 80)

print(
    candidate["time_diff_minutes"].describe(
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

print("\nMatches within:")

for minutes in [15, 30, 60, 120, 180, 360, 720, 1440]:

    n = (candidate["time_diff_minutes"] <= minutes).sum()

    pct = 100 * n / len(candidate)

    print(
        f"{minutes:4d} minutes : "
        f"{n:4d} / {len(candidate)} "
        f"({pct:.1f}%)"
    )

print("\nSaved:")
print(OUTPUT)
from pathlib import Path
import pandas as pd

RAW_DIR = Path("01_data/raw/usgs_matched/Matched_WQ_S2")
OUT_DIR = Path("01_data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("Matched_WQ_S2_*.csv"))

DELAWARE_HUC4 = "0204"

filtered_chunks = []

for file in files:

    print(f"\nReading: {file.name}")

    for chunk in pd.read_csv(
        file,
        chunksize=150_000,
        low_memory=False,
        dtype={
            "site_no": "string",
            "huc4": "string",
            "DepthCd": "string",
            "DepthPcode": "string",
            "DepthUnits": "string",
            "nhd_identifier": "string",
        },
    ):

        chunk["huc4_clean"] = (
            chunk["huc4"]
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )

        turbidity_mask = (
            chunk["parm_nm"]
            .astype("string")
            .str.contains("turbid", case=False, na=False)
        )

        delaware_mask = chunk["huc4_clean"] == DELAWARE_HUC4

        subset = chunk[turbidity_mask & delaware_mask].copy()

        if len(subset) > 0:
            filtered_chunks.append(subset)

if not filtered_chunks:
    raise ValueError("No Delaware turbidity observations were found.")

df = pd.concat(filtered_chunks, ignore_index=True)

df["MeasurementValue"] = pd.to_numeric(
    df["MeasurementValue"], errors="coerce"
)

df["scene_datetime_UTC"] = pd.to_datetime(
    df["scene_datetime_UTC"], errors="coerce"
)

df["sample_datetime_UTC"] = pd.to_datetime(
    df["sample_datetime_UTC"], errors="coerce"
)

output_file = OUT_DIR / "delaware_turbidity_matched.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 75)
print("DELAWARE TURBIDITY DATASET")
print("=" * 75)

print("\nRows:", len(df))
print("Unique USGS stations:", df["site_no"].nunique())

print("\nParameter names:")
print(df["parm_nm"].value_counts(dropna=False))

print("\nUnits:")
print(df["parm_unit"].value_counts(dropna=False))

print("\nSatellite date range:")
print(df["scene_datetime_UTC"].min(), "to", df["scene_datetime_UTC"].max())

print("\nTurbidity measurement range:")
print(df["MeasurementValue"].describe())

print("\nSTATIONS:")
station_summary = (
    df.groupby(["site_no", "station_nm"])
      .agg(
          rows=("MeasurementValue", "size"),
          valid_turbidity=("MeasurementValue", "count"),
          unique_scenes=("scene", "nunique"),
          first_date=("scene_datetime_UTC", "min"),
          last_date=("scene_datetime_UTC", "max"),
          latitude=("Lat", "median"),
          longitude=("Long", "median"),
      )
      .sort_values("unique_scenes", ascending=False)
)

print(station_summary.to_string())

print("\n" + "=" * 75)
print("REFLECTANCE AVAILABILITY")
print("=" * 75)

for col in [
    "B02_center",
    "B02_buf250_mean",
    "B02_buf250_med",
    "B03_center",
    "B03_buf250_mean",
    "B03_buf250_med",
    "B04_center",
    "B04_buf250_mean",
    "B04_buf250_med",
]:
    if col in df.columns:
        print(f"{col:20s}: {df[col].notna().sum():6d} valid")

print("\nSaved to:")
print(output_file)


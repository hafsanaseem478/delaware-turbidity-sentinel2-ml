from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "01_data/processed/delaware_turbidity_modeling_base.csv"
)

OUTPUT = Path(
    "01_data/processed/delaware_turbidity_features.csv"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

print("=" * 80)
print("SANTOS-STYLE SPECTRAL FEATURE GENERATION")
print("=" * 80)

print("\nInput samples:", len(df))

B2 = df["B02_rhow"]
B3 = df["B03_rhow"]
B4 = df["B04_rhow"]
B5 = df["B05_rhow"]
B6 = df["B06_rhow"]
B8 = df["B08_rhow"]
B11 = df["B11_rhow"]
B12 = df["B12_rhow"]

def safe_divide(numerator, denominator):

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=float
    )

    valid = np.abs(denominator) > 1e-12

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return result

features = {}

features["NDVI"] = safe_divide(
    B8 - B4,
    B8 + B4
)

features["GNDVI"] = safe_divide(
    B8 - B3,
    B8 + B3
)

features["EVI"] = safe_divide(
    2.5 * (B8 - B4),
    B8 + 6 * B4 - 7.5 * B2 + 1
)

features["SAVI"] = (
    safe_divide(
        B8 - B4,
        B8 + B4 + 0.428
    )
    * 1.428
)

features["NDMI"] = safe_divide(
    B8 - B11,
    B8 + B11
)

features["MSI"] = safe_divide(
    B11,
    B8
)

features["NBRI"] = safe_divide(
    B8 - B12,
    B8 + B12
)

features["BSI"] = safe_divide(
    (B11 + B4) - (B8 + B2),
    (B11 + B4) + (B8 + B2)
)

features["NDWI"] = safe_divide(
    B3 - B8,
    B3 + B8
)

features["NDSI"] = safe_divide(
    B3 - B11,
    B3 + B11
)

features["NDGI"] = safe_divide(
    B3 - B4,
    B3 + B4
)

features["ARVI"] = safe_divide(
    B8 - 2 * B4 + B2,
    B8 + 2 * B4
)

features["SIPI"] = safe_divide(
    B8 - B2,
    B8 - B4
)

features["SWM"] = safe_divide(
    B2 + B3,
    B8 + B11
)

features["AWEI"] = (
    4 * (B3 - B11)
    - (
        0.25 * B8
        + 2.75 * B12
    )
)

features["3BSI"] = (
    safe_divide(
        np.ones(len(df)),
        B4
    )
    -
    safe_divide(
        np.ones(len(df)),
        B5
    )
) * B6

feature_df = pd.DataFrame(
    features,
    index=df.index
)

df = pd.concat(
    [df, feature_df],
    axis=1
)

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

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

cube_root_features = {
    f"{feature}_cbrt":
        np.cbrt(df[feature])
    for feature in INDEX_FEATURES
}

cube_root_df = pd.DataFrame(
    cube_root_features,
    index=df.index
)

df = pd.concat(
    [df, cube_root_df],
    axis=1
)

print("\n" + "=" * 80)
print("FEATURE AVAILABILITY")
print("=" * 80)

for feature in INDEX_FEATURES:

    valid = df[feature].notna().sum()

    print(
        f"{feature:8s}: "
        f"{valid:4d} / {len(df)}"
    )

complete_rows = (
    df[INDEX_FEATURES]
    .notna()
    .all(axis=1)
)

print("\nSamples with all 16 indices:")
print(
    f"{complete_rows.sum()} / {len(df)}"
)

print("\nSamples lost because of index calculation:")
print(
    len(df) - complete_rows.sum()
)

print("\n" + "=" * 80)
print("INDEX RANGES")
print("=" * 80)

print(
    df[INDEX_FEATURES]
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

df = df[
    complete_rows
].copy()

print("\n" + "=" * 80)
print("TURBIDITY")
print("=" * 80)

print(
    df["MeasurementValue"]
    .describe(
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

print("\n" + "=" * 80)
print("FINAL FEATURE DATASET")
print("=" * 80)

print("Samples:", len(df))
print("Original Santos indices:", 17)
print("Available in our dataset:", 16)
print("Unavailable: GCI (requires B09)")
print("Cube-root index features:", 16)

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)


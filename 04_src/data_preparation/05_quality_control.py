from pathlib import Path
import pandas as pd

INPUT = Path(
    "01_data/processed/delaware_basin_candidate_scenes.csv"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

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

mean_cols = [
    f"{band}_buf250_mean"
    for band in BANDS
]

median_cols = [
    f"{band}_buf250_med"
    for band in BANDS
]

print("=" * 80)
print("ACOLITE REFLECTANCE QUALITY CHECK")
print("=" * 80)

print("\nTotal candidate station-scene samples:")
print(len(df))

def check_group(name, columns):

    existing = [
        col for col in columns
        if col in df.columns
    ]

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for col in existing:

        valid = df[col].notna().sum()
        pct = 100 * valid / len(df)

        print(
            f"{col:22s}"
            f"{valid:5d} valid "
            f"({pct:5.1f}%)"
        )

    complete = df[existing].notna().all(axis=1)

    print("\nSamples with ALL bands available:")

    print(
        f"{complete.sum()} / {len(df)} "
        f"({100 * complete.mean():.1f}%)"
    )

    return complete

center_complete = check_group(
    "CENTER PIXEL",
    center_cols
)

mean_complete = check_group(
    "250 m BUFFER MEAN",
    mean_cols
)

median_complete = check_group(
    "250 m BUFFER MEDIAN",
    median_cols
)

df["center_complete"] = center_complete
df["mean_complete"] = mean_complete
df["median_complete"] = median_complete

print("\n" + "=" * 80)
print("COMPLETENESS BY STATION")
print("=" * 80)

summary = (
    df.groupby(["site_no", "station_nm"])
    .agg(
        candidate_samples=("scene", "size"),

        center_complete=(
            "center_complete",
            "sum"
        ),

        mean250_complete=(
            "mean_complete",
            "sum"
        ),

        median250_complete=(
            "median_complete",
            "sum"
        ),
    )
)

summary["center_pct"] = (
    100
    * summary["center_complete"]
    / summary["candidate_samples"]
)

summary["median250_pct"] = (
    100
    * summary["median250_complete"]
    / summary["candidate_samples"]
)

print(summary.to_string())

within_30 = df[
    df["time_diff_minutes"] <= 30
].copy()

print("\n" + "=" * 80)
print("AFTER 30-MINUTE TIME FILTER")
print("=" * 80)

print(
    "Samples remaining:",
    len(within_30)
)

median_existing = [
    col for col in median_cols
    if col in within_30.columns
]

complete_30 = (
    within_30[median_existing]
    .notna()
    .all(axis=1)
)

print(
    "Samples with complete "
    "250 m median reflectance:",
    complete_30.sum()
)

print(
    "Percentage:",
    round(
        100 * complete_30.mean(),
        1
    ),
    "%"
)

print("\n" + "=" * 80)
print("250 m MEDIAN REFLECTANCE RANGES")
print("=" * 80)

print(
    df[median_cols]
    .describe()
    .T[
        [
            "count",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]
)


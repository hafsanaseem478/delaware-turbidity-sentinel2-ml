from pathlib import Path
import pandas as pd

data_dir = Path("01_data/raw/usgs_matched")

csv_files = list(data_dir.rglob("*.csv"))

print(f"\nFound {len(csv_files)} CSV files\n")

for file in csv_files:
    print("=" * 70)
    print("FILE:", file)

    df = pd.read_csv(file)

    print("Shape:", df.shape)

    print("\nCOLUMNS:")
    for col in df.columns:
        print(" -", col)

    print("\nFIRST 5 ROWS:")
    print(df.head())

    print()
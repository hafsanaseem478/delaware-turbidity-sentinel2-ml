from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

from sklearn.model_selection import (
    PredefinedSplit,
    cross_val_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import ConvergenceWarning

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

OUTPUT = Path(
    "05_results/metrics/10_gpr_tuning.json"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    selection = json.load(f)

features = selection["final_original_features"]

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

X = train[features]
y = train["MeasurementValue"].values

cv = PredefinedSplit(
    test_fold=train["cv_fold"].values
)

print("=" * 80)
print("GAUSSIAN PROCESS REGRESSION")
print("=" * 80)

print("\nFeatures:")

for feature in features:
    print(" -", feature)

print("\nTraining samples:", len(train))

print(
    "Held-out test samples:",
    (df["split"] == "test").sum()
)

print("\nTest set is NOT used.")

kernels = {

    "RBF":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2)
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2)
        ),

    "Matern_1.5":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2)
        ),

    "Matern_2.5":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2)
        ),

    "RationalQuadratic":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2)
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2)
        ),
}

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

results = []

for name, kernel in kernels.items():

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    model = Pipeline([
        (
            "scale",
            StandardScaler()
        ),

        (
            "gpr",
            GaussianProcessRegressor(
                kernel=kernel,

                normalize_y=True,

                alpha=1e-8,

                n_restarts_optimizer=0,

                random_state=42,
            )
        ),
    ])

    scores = cross_val_score(
        model,
        X,
        y,
        scoring="r2",
        cv=cv,
        n_jobs=1,
    )

    mean_r2 = float(
        scores.mean()
    )

    std_r2 = float(
        scores.std()
    )

    print("\nFold RÂ²:")

    for i, score in enumerate(
        scores,
        start=1
    ):

        print(
            f"Fold {i}: {score:.4f}"
        )

    print(
        "\nMean CV RÂ²:",
        round(mean_r2, 4)
    )

    print(
        "CV standard deviation:",
        round(std_r2, 4)
    )

    results.append({
        "kernel": name,
        "mean_cv_r2": mean_r2,
        "std_cv_r2": std_r2,
        "fold_r2": [
            float(x)
            for x in scores
        ],
    })

results = sorted(
    results,
    key=lambda x: x["mean_cv_r2"],
    reverse=True
)

best = results[0]

print("\n" + "=" * 80)
print("BEST GPR")
print("=" * 80)

print(
    "Kernel:",
    best["kernel"]
)

print(
    "Mean CV RÂ²:",
    round(
        best["mean_cv_r2"],
        4
    )
)

print(
    "CV standard deviation:",
    round(
        best["std_cv_r2"],
        4
    )
)

output = {
    "features": features,
    "preprocessing": "StandardScaler",
    "normalize_y": True,
    "best_kernel": best["kernel"],
    "best_mean_cv_r2": best["mean_cv_r2"],
    "best_std_cv_r2": best["std_cv_r2"],
    "all_results": results,
}

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )

print("\nSaved:")
print(OUTPUT)

print(
    "\n281 held-out test observations remain untouched."
)


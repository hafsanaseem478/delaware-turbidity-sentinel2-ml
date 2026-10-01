from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FILES
# ============================================================

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

OUTPUT = Path(
    "05_results/metrics/11_gpr_ard_tuning.json"
)


# ============================================================
# LOAD
# ============================================================

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


# ============================================================
# TRAINING DATA ONLY
# ============================================================

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

n_features = len(features)


print("=" * 80)
print("FINAL GPR ARD TUNING")
print("=" * 80)

print("\nTraining samples:", len(train))
print(
    "Held-out test samples:",
    (df["split"] == "test").sum()
)

print("\nFeatures:", n_features)

for f in features:
    print(" -", f)

print("\n281-sample test set is NOT used.")


# ============================================================
# ARD
#
# np.ones(n_features) gives each predictor its own
# independently optimized length scale.
# ============================================================

ard_initial = np.ones(n_features)


kernels = {

    # Previous best family — retained as baseline
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
            alpha_bounds=(1e-2, 1e2),
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2),
        ),

    # ARD RBF
    "RBF_ARD":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        RBF(
            length_scale=ard_initial,
            length_scale_bounds=(1e-2, 1e2),
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2),
        ),

    # ARD Matérn nu = 1.5
    "Matern_1.5_ARD":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        Matern(
            length_scale=ard_initial,
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5,
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2),
        ),

    # ARD Matérn nu = 2.5
    "Matern_2.5_ARD":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        Matern(
            length_scale=ard_initial,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2),
        ),
}


# ============================================================
# GPR FUNCTION
#
# Scaling happens separately INSIDE every fold.
# ============================================================

def make_pipeline(kernel):

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "gpr",
            GaussianProcessRegressor(
                kernel=clone(kernel),

                normalize_y=True,

                # tiny numerical stabilizer
                alpha=1e-8,

                # one additional optimizer restart
                # to reduce dependence on initialization
                n_restarts_optimizer=1,

                random_state=42,
            )
        ),
    ])


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# ============================================================
# MANUAL LOCKED-FOLD CV
#
# Manual loop lets us inspect optimized kernels.
# ============================================================

all_results = []


for kernel_name, kernel in kernels.items():

    print("\n" + "=" * 80)
    print(kernel_name)
    print("=" * 80)

    fold_scores = []
    optimized_kernels = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model = make_pipeline(kernel)

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_val
        )

        score = r2_score(
            y_val,
            predictions
        )

        fold_scores.append(
            float(score)
        )

        fitted_kernel = (
            model
            .named_steps["gpr"]
            .kernel_
        )

        optimized_kernels.append(
            str(fitted_kernel)
        )

        print(
            f"Fold {fold_number} R²: "
            f"{score:.4f}"
        )

    mean_r2 = float(
        np.mean(fold_scores)
    )

    std_r2 = float(
        np.std(fold_scores)
    )

    print(
        "\nMean CV R²:",
        round(mean_r2, 4)
    )

    print(
        "CV standard deviation:",
        round(std_r2, 4)
    )

    all_results.append({
        "kernel": kernel_name,
        "mean_cv_r2": mean_r2,
        "std_cv_r2": std_r2,
        "fold_r2": fold_scores,
        "optimized_kernels": optimized_kernels,
    })


# ============================================================
# SELECT BEST
# ============================================================

all_results = sorted(
    all_results,
    key=lambda x: x["mean_cv_r2"],
    reverse=True
)

best = all_results[0]


print("\n" + "=" * 80)
print("FINAL BEST GPR")
print("=" * 80)

print(
    "Kernel:",
    best["kernel"]
)

print(
    "Mean CV R²:",
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

print("\nFold scores:")

for i, score in enumerate(
    best["fold_r2"],
    start=1
):
    print(
        f"Fold {i}: {score:.4f}"
    )


# ============================================================
# COMPARISON TO XGBOOST
# ============================================================

XGB_R2 = 0.7213

print("\n" + "=" * 80)
print("TRAINING CV COMPARISON")
print("=" * 80)

print(
    "Tuned XGBoost R²:",
    XGB_R2
)

print(
    "Best GPR R²:",
    round(
        best["mean_cv_r2"],
        4
    )
)

print(
    "Difference:",
    round(
        XGB_R2
        - best["mean_cv_r2"],
        4
    )
)


# ============================================================
# SAVE
# ============================================================

output = {
    "features": features,

    "n_train": int(len(train)),

    "n_test_untouched": int(
        (df["split"] == "test").sum()
    ),

    "preprocessing": "StandardScaler",

    "normalize_y": True,

    "n_restarts_optimizer": 1,

    "best_kernel":
        best["kernel"],

    "best_mean_cv_r2":
        best["mean_cv_r2"],

    "best_std_cv_r2":
        best["std_cv_r2"],

    "best_fold_r2":
        best["fold_r2"],

    "all_results":
        all_results,
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
    "\nTest set remains untouched."
)
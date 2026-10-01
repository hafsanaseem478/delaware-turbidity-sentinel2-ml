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
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import PredefinedSplit
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

CSV_OUTPUT = Path(
    "05_results/tables/12_gpr_target_transform.csv"
)

JSON_OUTPUT = Path(
    "05_results/metrics/12_gpr_target_transform.json"
)

CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


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
    feature_info = json.load(f)

FEATURES = feature_info["final_original_features"]


# ============================================================
# TRAINING DATA ONLY
# ============================================================

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

X = train[FEATURES].copy()

y_raw = (
    train["MeasurementValue"]
    .astype(float)
    .values
)

cv = PredefinedSplit(
    test_fold=train["cv_fold"].values
)

n_features = len(FEATURES)


print("=" * 80)
print("GPR TARGET-TRANSFORMATION DIAGNOSTIC")
print("=" * 80)

print("\nTraining samples:", len(train))

print(
    "Held-out test samples:",
    (df["split"] == "test").sum()
)

print("\nFeatures:", n_features)

for feature in FEATURES:
    print(" -", feature)

print("\nTest set is NOT used.")


# ============================================================
# KERNELS
# ============================================================

ard_start = np.ones(n_features)

KERNELS = {

    # Previous best non-ARD family
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

    "RBF_ARD":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        RBF(
            length_scale=ard_start,
            length_scale_bounds=(1e-2, 1e2),
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2),
        ),

    "Matern_1.5_ARD":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        Matern(
            length_scale=ard_start,
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5,
        )
        +
        WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-5, 1e2),
        ),

    "Matern_2.5_ARD":
        ConstantKernel(
            1.0,
            (1e-2, 1e2)
        )
        *
        Matern(
            length_scale=ard_start,
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
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# ============================================================
# RUN ONE CV EXPERIMENT
#
# target_mode:
#   raw
#   log1p
#
# IMPORTANT:
# log1p predictions are converted BACK to FNU
# before evaluating R² / RMSE / MAE.
# ============================================================

def evaluate_gpr(
    kernel_name,
    kernel,
    target_mode
):

    fold_results = []
    optimized_kernels = []

    for fold_number, (
        train_idx,
        val_idx
    ) in enumerate(
        cv.split(),
        start=1
    ):

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train_raw = y_raw[train_idx]
        y_val_raw = y_raw[val_idx]


        # ----------------------------------------------------
        # Scale X using TRAINING FOLD ONLY
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_val_scaled = scaler.transform(
            X_val
        )


        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        if target_mode == "raw":

            y_train_model = y_train_raw.copy()

        elif target_mode == "log1p":

            y_train_model = np.log1p(
                y_train_raw
            )

        else:
            raise ValueError(
                f"Unknown target mode: {target_mode}"
            )


        # ----------------------------------------------------
        # GPR
        # ----------------------------------------------------

        model = GaussianProcessRegressor(
            kernel=clone(kernel),

            normalize_y=True,

            alpha=1e-8,

            # Give optimizer several starting points
            n_restarts_optimizer=3,

            random_state=42,
        )

        model.fit(
            X_train_scaled,
            y_train_model
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        pred_model_scale = model.predict(
            X_val_scaled
        )


        # ----------------------------------------------------
        # Convert prediction back to ORIGINAL FNU
        # ----------------------------------------------------

        if target_mode == "raw":

            pred_fnu = pred_model_scale

        else:

            pred_fnu = np.expm1(
                pred_model_scale
            )


        # ----------------------------------------------------
        # IMPORTANT:
        # Don't secretly clip negative predictions.
        # Count them instead.
        # ----------------------------------------------------

        negative_predictions = int(
            np.sum(pred_fnu < 0)
        )


        # ----------------------------------------------------
        # Metrics on ORIGINAL FNU scale
        # ----------------------------------------------------

        r2 = r2_score(
            y_val_raw,
            pred_fnu
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_val_raw,
                pred_fnu
            )
        )

        mae = mean_absolute_error(
            y_val_raw,
            pred_fnu
        )


        fold_results.append({
            "fold": fold_number,
            "r2": float(r2),
            "rmse_fnu": float(rmse),
            "mae_fnu": float(mae),
            "negative_predictions":
                negative_predictions,
        })

        optimized_kernels.append(
            str(model.kernel_)
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    r2_values = np.array([
        x["r2"]
        for x in fold_results
    ])

    rmse_values = np.array([
        x["rmse_fnu"]
        for x in fold_results
    ])

    mae_values = np.array([
        x["mae_fnu"]
        for x in fold_results
    ])

    return {

        "kernel":
            kernel_name,

        "target":
            target_mode,

        "mean_cv_r2":
            float(r2_values.mean()),

        "std_cv_r2":
            float(r2_values.std()),

        "mean_rmse_fnu":
            float(rmse_values.mean()),

        "mean_mae_fnu":
            float(mae_values.mean()),

        "total_negative_predictions":
            int(
                sum(
                    x["negative_predictions"]
                    for x in fold_results
                )
            ),

        "fold_results":
            fold_results,

        "optimized_kernels":
            optimized_kernels,
    }


# ============================================================
# RUN ALL CONTROLLED COMPARISONS
# ============================================================

all_results = []

for kernel_name, kernel in KERNELS.items():

    for target_mode in [
        "raw",
        "log1p",
    ]:

        print("\n" + "=" * 80)

        print(
            f"{kernel_name} | TARGET = {target_mode}"
        )

        print("=" * 80)

        result = evaluate_gpr(
            kernel_name,
            kernel,
            target_mode
        )

        all_results.append(result)

        for fold in result["fold_results"]:

            print(
                f"Fold {fold['fold']} "
                f"R² = {fold['r2']:.4f} "
                f"| RMSE = {fold['rmse_fnu']:.3f} "
                f"| MAE = {fold['mae_fnu']:.3f}"
            )

        print(
            "\nMean CV R²:",
            round(
                result["mean_cv_r2"],
                4
            )
        )

        print(
            "CV R² std:",
            round(
                result["std_cv_r2"],
                4
            )
        )

        print(
            "Mean RMSE:",
            round(
                result["mean_rmse_fnu"],
                3
            ),
            "FNU"
        )

        print(
            "Mean MAE:",
            round(
                result["mean_mae_fnu"],
                3
            ),
            "FNU"
        )

        print(
            "Negative predictions:",
            result[
                "total_negative_predictions"
            ]
        )


# ============================================================
# RANK BY R²
# ============================================================

all_results = sorted(
    all_results,
    key=lambda x: x["mean_cv_r2"],
    reverse=True
)

best = all_results[0]


# ============================================================
# SIMPLE TABLE
# ============================================================

table_rows = []

for result in all_results:

    table_rows.append({
        "kernel":
            result["kernel"],

        "target":
            result["target"],

        "mean_cv_r2":
            result["mean_cv_r2"],

        "std_cv_r2":
            result["std_cv_r2"],

        "mean_rmse_fnu":
            result["mean_rmse_fnu"],

        "mean_mae_fnu":
            result["mean_mae_fnu"],

        "negative_predictions":
            result[
                "total_negative_predictions"
            ],
    })

results_df = pd.DataFrame(
    table_rows
)


print("\n" + "=" * 80)
print("FINAL GPR TARGET COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 80)
print("BEST GPR CONFIGURATION")
print("=" * 80)

print(
    "Kernel:",
    best["kernel"]
)

print(
    "Target:",
    best["target"]
)

print(
    "Mean CV R²:",
    round(
        best["mean_cv_r2"],
        4
    )
)

print(
    "CV R² std:",
    round(
        best["std_cv_r2"],
        4
    )
)

print(
    "Mean RMSE:",
    round(
        best["mean_rmse_fnu"],
        3
    ),
    "FNU"
)

print(
    "Mean MAE:",
    round(
        best["mean_mae_fnu"],
        3
    ),
    "FNU"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    CSV_OUTPUT,
    index=False
)

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "features": FEATURES,
            "n_train": int(len(train)),
            "n_test_untouched": int(
                (df["split"] == "test").sum()
            ),
            "n_restarts_optimizer": 3,
            "best_configuration": best,
            "all_results": all_results,
        },
        f,
        indent=4
    )


print("\nSaved:")
print(CSV_OUTPUT)
print(JSON_OUTPUT)

print(
    "\n281 held-out test observations "
    "remain untouched."
)
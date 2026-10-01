from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
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
    "05_results/tables/13_gpr_matern32_nm_cv.csv"
)

JSON_OUTPUT = Path(
    "05_results/metrics/13_gpr_matern32_nm.json"
)

CSV_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

JSON_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
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

    feature_info = json.load(f)


FEATURES = (
    feature_info["final_original_features"]
)


# ============================================================
# TRAINING DATA ONLY
# ============================================================

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

X = train[FEATURES].astype(float)

y = (
    train["MeasurementValue"]
    .astype(float)
    .values
)


cv = PredefinedSplit(
    test_fold=train["cv_fold"].values
)


print("=" * 80)
print("LITERATURE-ALIGNED NM/GPR")
print("MATERN 3/2 + NELDER-MEAD")
print("=" * 80)

print("\nTraining samples:", len(train))

print(
    "Held-out samples NOT USED:",
    (df["split"] == "test").sum()
)

print("\nFeatures:")

for feature in FEATURES:
    print(" -", feature)


# ============================================================
# NELDER-MEAD OPTIMIZER
#
# sklearn passes theta in LOG hyperparameter space.
#
# obj_func = negative log marginal likelihood,
# therefore minimizing it = maximizing marginal likelihood.
# ============================================================

def nelder_mead_optimizer(
    obj_func,
    initial_theta,
    bounds
):

    result = minimize(
        fun=lambda theta: obj_func(
            theta,
            eval_gradient=False
        ),

        x0=initial_theta,

        method="Nelder-Mead",

        bounds=bounds,

        options={
            "maxiter": 2000,
            "xatol": 1e-6,
            "fatol": 1e-6,
            "adaptive": True,
        },
    )

    return (
        result.x,
        result.fun
    )


# ============================================================
# PAPER-ALIGNED KERNEL
#
# García-Nieto et al.:
#
# l = 0.3
# a = 1
# noise variance = 0
#
# sklearn cannot optimize an EXACT zero noise variance because
# kernel parameters are optimized in log-space.
#
# Therefore 1e-8 is used as a numerical approximation to zero.
#
# k = a * Matern32(l) + noise
# ============================================================

def make_kernel():

    return (

        ConstantKernel(
            constant_value=1.0,

            # broad numerical bounds;
            # paper reports initial value,
            # not these sklearn-specific bounds
            constant_value_bounds=(
                1e-6,
                1e6
            )
        )

        *

        Matern(
            length_scale=0.3,

            length_scale_bounds=(
                1e-4,
                1e4
            ),

            # Matérn 3/2
            nu=1.5
        )

        +

        WhiteKernel(
            # numerical approximation
            # to paper's initial sigma² = 0
            noise_level=1e-8,

            noise_level_bounds=(
                1e-10,
                1e2
            )
        )
    )


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# ============================================================
# LOCKED 5-FOLD CV
# ============================================================

fold_results = []


for fold_number, (
    train_idx,
    val_idx
) in enumerate(
    cv.split(),
    start=1
):

    print("\n" + "-" * 80)
    print(f"FOLD {fold_number}")
    print("-" * 80)


    # --------------------------------------------------------
    # Fold data
    # --------------------------------------------------------

    X_train = X.iloc[
        train_idx
    ]

    X_val = X.iloc[
        val_idx
    ]

    y_train = y[
        train_idx
    ]

    y_val = y[
        val_idx
    ]


    # --------------------------------------------------------
    # Standardize predictors
    #
    # Fit scaler ONLY on training portion of fold.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_val_scaled = (
        scaler.transform(
            X_val
        )
    )


    # --------------------------------------------------------
    # GPR
    # --------------------------------------------------------

    gpr = GaussianProcessRegressor(

        kernel=make_kernel(),

        # Nelder-Mead rather than sklearn default L-BFGS-B
        optimizer=nelder_mead_optimizer,

        # centers/scales target internally
        normalize_y=True,

        # numerical jitter only.
        # Actual modeled noise is WhiteKernel.
        alpha=1e-10,

        random_state=42,
    )


    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    gpr.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    pred, pred_std = gpr.predict(
        X_val_scaled,
        return_std=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    r2 = r2_score(
        y_val,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            pred
        )
    )

    mae = mean_absolute_error(
        y_val,
        pred
    )

    mbe = np.mean(
        pred - y_val
    )


    # --------------------------------------------------------
    # OPTIMIZED PARAMETERS
    # --------------------------------------------------------

    fitted_kernel = gpr.kernel_

    # Structure:
    #
    # ConstantKernel * Matern + WhiteKernel
    #
    amplitude = float(
        fitted_kernel.k1.k1.constant_value
    )

    length_scale = float(
        fitted_kernel.k1.k2.length_scale
    )

    noise_variance = float(
        fitted_kernel.k2.noise_level
    )


    print(
        f"R²:   {r2:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f} FNU"
    )

    print(
        f"MAE:  {mae:.4f} FNU"
    )

    print(
        f"MBE:  {mbe:.4f} FNU"
    )


    print(
        "\nOptimized NM/GPR parameters:"
    )

    print(
        f"Amplitude a:        "
        f"{amplitude:.6f}"
    )

    print(
        f"Length scale l:     "
        f"{length_scale:.6f}"
    )

    print(
        f"Noise variance σ²:  "
        f"{noise_variance:.8f}"
    )

    print(
        "\nOptimized kernel:"
    )

    print(
        fitted_kernel
    )


    fold_results.append({

        "fold":
            fold_number,

        "r2":
            float(r2),

        "rmse_fnu":
            float(rmse),

        "mae_fnu":
            float(mae),

        "mbe_fnu":
            float(mbe),

        "amplitude_a":
            amplitude,

        "length_scale_l":
            length_scale,

        "noise_variance":
            noise_variance,

        "mean_prediction_std":
            float(
                np.mean(pred_std)
            ),
    })


# ============================================================
# CV SUMMARY
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


mean_r2 = float(
    results_df["r2"].mean()
)

std_r2 = float(
    results_df["r2"].std(
        ddof=0
    )
)

mean_rmse = float(
    results_df[
        "rmse_fnu"
    ].mean()
)

mean_mae = float(
    results_df[
        "mae_fnu"
    ].mean()
)

mean_mbe = float(
    results_df[
        "mbe_fnu"
    ].mean()
)


print("\n" + "=" * 80)
print("NM/GPR MATERN32 CV RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "fold",
            "r2",
            "rmse_fnu",
            "mae_fnu",
            "mbe_fnu",
            "amplitude_a",
            "length_scale_l",
            "noise_variance",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\nMean CV R²:", round(mean_r2, 4))

print(
    "CV R² std:",
    round(std_r2, 4)
)

print(
    "Mean RMSE:",
    round(mean_rmse, 4),
    "FNU"
)

print(
    "Mean MAE:",
    round(mean_mae, 4),
    "FNU"
)

print(
    "Mean MBE:",
    round(mean_mbe, 4),
    "FNU"
)


# ============================================================
# COMPARISON WITH PREVIOUS MODELS
# ============================================================

print("\n" + "=" * 80)
print("TRAINING CV COMPARISON")
print("=" * 80)

print(
    "Tuned XGBoost:           0.7213"
)

print(
    "Previous GPR:            0.5985"
)

print(
    "Literature NM/GPR:      ",
    round(mean_r2, 4)
)


# ============================================================
# FIT FINAL NM/GPR TO ALL 1121 TRAINING SAMPLES
#
# This still DOES NOT use held-out test observations.
# ============================================================

print("\n" + "=" * 80)
print("FIT NM/GPR ON FULL TRAINING SET")
print("=" * 80)


full_scaler = StandardScaler()

X_full_scaled = (
    full_scaler.fit_transform(X)
)


final_gpr = GaussianProcessRegressor(

    kernel=make_kernel(),

    optimizer=nelder_mead_optimizer,

    normalize_y=True,

    alpha=1e-10,

    random_state=42,
)


final_gpr.fit(
    X_full_scaled,
    y
)


final_kernel = (
    final_gpr.kernel_
)


final_amplitude = float(
    final_kernel.k1.k1.constant_value
)

final_length_scale = float(
    final_kernel.k1.k2.length_scale
)

final_noise = float(
    final_kernel.k2.noise_level
)


print(
    "\nFinal optimized kernel:"
)

print(
    final_kernel
)

print(
    "\nAmplitude a:",
    final_amplitude
)

print(
    "Length scale l:",
    final_length_scale
)

print(
    "Noise variance σ²:",
    final_noise
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    CSV_OUTPUT,
    index=False
)


summary = {

    "method":
        "NM/GPR Matern32",

    "literature_basis":
        "Garcia-Nieto et al. 2026",

    "features":
        FEATURES,

    "n_train":
        int(len(train)),

    "n_test_not_used":
        int(
            (df["split"] == "test").sum()
        ),

    "kernel":
        "ConstantKernel * Matern(nu=1.5) + WhiteKernel",

    "initial_parameters": {

        "length_scale_l":
            0.3,

        "amplitude_a":
            1.0,

        "noise_variance":
            1e-8,

        "note":
            "Paper initializes noise variance at zero; "
            "1e-8 used because sklearn optimizes "
            "positive kernel parameters in log-space."
    },

    "optimizer":
        "Nelder-Mead",

    "cv":
        "existing locked 5-fold PredefinedSplit",

    "mean_cv_r2":
        mean_r2,

    "std_cv_r2":
        std_r2,

    "mean_cv_rmse_fnu":
        mean_rmse,

    "mean_cv_mae_fnu":
        mean_mae,

    "mean_cv_mbe_fnu":
        mean_mbe,

    "fold_results":
        fold_results,

    "full_training_optimized_parameters": {

        "amplitude_a":
            final_amplitude,

        "length_scale_l":
            final_length_scale,

        "noise_variance":
            final_noise,

        "kernel":
            str(final_kernel),
    }
}


with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\nSaved:")
print(CSV_OUTPUT)
print(JSON_OUTPUT)

print(
    "\nIMPORTANT: held-out observations "
    "were NOT evaluated by this script."
)
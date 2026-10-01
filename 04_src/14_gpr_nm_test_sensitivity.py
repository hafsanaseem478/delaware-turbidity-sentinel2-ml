from pathlib import Path
import json

import numpy as np
import pandas as pd

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

from sklearn.preprocessing import StandardScaler


# ============================================================
# FILES
# ============================================================

DATA_FILE = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

GPR_FILE = Path(
    "05_results/metrics/13_gpr_matern32_nm.json"
)

METRIC_OUTPUT = Path(
    "05_results/metrics/"
    "14_gpr_nm_test_sensitivity_metrics.csv"
)

PRED_OUTPUT = Path(
    "05_results/metrics/"
    "14_gpr_nm_test_sensitivity_predictions.csv"
)

JSON_OUTPUT = Path(
    "05_results/metrics/"
    "14_gpr_nm_test_sensitivity.json"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:
    feature_info = json.load(f)

with open(
    GPR_FILE,
    "r",
    encoding="utf-8"
) as f:
    gpr_info = json.load(f)


features = feature_info[
    "final_original_features"
]


# ============================================================
# TRAIN / TEST
# ============================================================

train = (
    df[df["split"] == "train"]
    .sort_values("sample_id")
    .copy()
)

test = (
    df[df["split"] == "test"]
    .sort_values("sample_id")
    .copy()
)


X_train = train[features].astype(float)
y_train = (
    train["MeasurementValue"]
    .astype(float)
    .values
)

X_test = test[features].astype(float)
y_test = (
    test["MeasurementValue"]
    .astype(float)
    .values
)


print("=" * 80)
print("POST-HOC GPR SENSITIVITY TEST")
print("MATERN 3/2 + NELDER-MEAD PARAMETERS")
print("=" * 80)

print("\nTraining samples:", len(train))
print("Test samples:", len(test))

print(
    "\nNOTE: These 281 observations have already "
    "been viewed previously."
)

print(
    "This result must be reported as a "
    "post-hoc sensitivity analysis."
)


# ============================================================
# LOAD ALREADY-OPTIMIZED PARAMETERS
#
# These came from the FULL TRAINING SET in script 13.
# We DO NOT optimize them again here.
# ============================================================

params = gpr_info[
    "full_training_optimized_parameters"
]

amplitude = float(
    params["amplitude_a"]
)

length_scale = float(
    params["length_scale_l"]
)

noise_variance = float(
    params["noise_variance"]
)


print("\nFrozen parameters:")

print(
    f"Amplitude a:       {amplitude:.8f}"
)

print(
    f"Length scale l:    {length_scale:.8f}"
)

print(
    f"Noise variance σ²: {noise_variance:.8f}"
)


# ============================================================
# STANDARDIZE USING TRAINING DATA ONLY
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# FIXED KERNEL
#
# bounds="fixed" prevents further optimization.
# ============================================================

kernel = (

    ConstantKernel(
        constant_value=amplitude,
        constant_value_bounds="fixed"
    )

    *

    Matern(
        length_scale=length_scale,
        length_scale_bounds="fixed",
        nu=1.5
    )

    +

    WhiteKernel(
        noise_level=noise_variance,
        noise_level_bounds="fixed"
    )
)


# ============================================================
# FROZEN GPR
# ============================================================

gpr = GaussianProcessRegressor(

    kernel=kernel,

    # CRITICAL:
    # no optimization on this run
    optimizer=None,

    normalize_y=True,

    alpha=1e-10,

    random_state=42,
)


print("\nFitting frozen sensitivity model...")

gpr.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

prediction, prediction_std = gpr.predict(
    X_test_scaled,
    return_std=True
)


# ============================================================
# METRICS
# ============================================================

r2 = r2_score(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

nrmse = (
    rmse
    / np.mean(y_test)
)

mae = mean_absolute_error(
    y_test,
    prediction
)

mbe = np.mean(
    prediction - y_test
)


print("\n" + "=" * 80)
print("POST-HOC SENSITIVITY RESULTS")
print("=" * 80)

print(
    f"R²:     {r2:.4f}"
)

print(
    f"RMSE:   {rmse:.4f} FNU"
)

print(
    f"nRMSE:  {nrmse:.4f}"
)

print(
    f"MAE:    {mae:.4f} FNU"
)

print(
    f"MBE:    {mbe:.4f} FNU"
)

print(
    f"Mean predictive SD: "
    f"{prediction_std.mean():.4f} FNU"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "model":
            "GPR_Matern32_NelderMead_posthoc",

        "r2":
            float(r2),

        "rmse_fnu":
            float(rmse),

        "nrmse":
            float(nrmse),

        "mae_fnu":
            float(mae),

        "mbe_fnu":
            float(mbe),

        "mean_prediction_std":
            float(
                prediction_std.mean()
            ),
    }
])

metrics.to_csv(
    METRIC_OUTPUT,
    index=False
)


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "sample_id":
        test["sample_id"].values,

    "site_no":
        test["site_no"].values,

    "station_nm":
        test["station_nm"].values,

    "observed_turbidity_fnu":
        y_test,

    "gpr_nm_prediction_fnu":
        prediction,

    "gpr_nm_prediction_std":
        prediction_std,

    "gpr_nm_95_lower":
        prediction
        - 1.96 * prediction_std,

    "gpr_nm_95_upper":
        prediction
        + 1.96 * prediction_std,
})


predictions.to_csv(
    PRED_OUTPUT,
    index=False
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary = {

    "analysis_type":
        "post-hoc sensitivity analysis",

    "important_note":
        (
            "The 281 test observations had already "
            "been evaluated before development of "
            "this literature-guided GPR variant."
        ),

    "model":
        "Matern 3/2 GPR",

    "optimizer_used_during_training_tuning":
        "Nelder-Mead",

    "optimizer_used_during_this_test":
        "None - hyperparameters frozen",

    "n_train":
        int(len(train)),

    "n_test":
        int(len(test)),

    "features":
        features,

    "frozen_parameters": {

        "amplitude_a":
            amplitude,

        "length_scale_l":
            length_scale,

        "noise_variance":
            noise_variance,
    },

    "metrics": {

        "r2":
            float(r2),

        "rmse_fnu":
            float(rmse),

        "nrmse":
            float(nrmse),

        "mae_fnu":
            float(mae),

        "mbe_fnu":
            float(mbe),

        "mean_prediction_std":
            float(
                prediction_std.mean()
            ),
    },
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
print(METRIC_OUTPUT)
print(PRED_OUTPUT)
print(JSON_OUTPUT)

print(
    "\nDo NOT replace the original final-test "
    "results with this sensitivity result."
)
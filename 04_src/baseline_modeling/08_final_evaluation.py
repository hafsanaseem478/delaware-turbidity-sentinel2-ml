from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

INPUT = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/07_selected_features.json"
)

XGB_FILE = Path(
    "05_results/metrics/09_xgboost_tuning.json"
)

GPR_FILE = Path(
    "05_results/metrics/11_gpr_ard_tuning.json"
)

METRIC_OUTPUT = Path(
    "05_results/metrics/12_final_test_metrics.csv"
)

PRED_OUTPUT = Path(
    "05_results/metrics/12_final_test_predictions.csv"
)

SUMMARY_OUTPUT = Path(
    "05_results/metrics/12_final_test_summary.json"
)

df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={"site_no": "string"}
)

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    feature_info = json.load(f)

with open(XGB_FILE, "r", encoding="utf-8") as f:
    xgb_info = json.load(f)

with open(GPR_FILE, "r", encoding="utf-8") as f:
    gpr_info = json.load(f)

features = feature_info["final_original_features"]

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

X_train = train[features]
y_train = train["MeasurementValue"].values

X_test = test[features]
y_test = test["MeasurementValue"].values

print("=" * 80)
print("FINAL HELD-OUT TEST EVALUATION")
print("=" * 80)

print("\nTraining samples:", len(train))
print("Test samples:", len(test))

print("\nFeatures:", len(features))

for feature in features:
    print(" -", feature)

xgb_params = xgb_info["best_parameters"]

xgb = XGBRegressor(
    objective="reg:squarederror",

    learning_rate=xgb_params["learning_rate"],
    max_depth=xgb_params["max_depth"],
    min_child_weight=xgb_params["min_child_weight"],
    n_estimators=xgb_params["n_estimators"],
    subsample=xgb_params["subsample"],

    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,

    random_state=42,
    n_jobs=-1,
)

print("\nFitting frozen XGBoost...")

xgb.fit(
    X_train,
    y_train
)

xgb_pred = xgb.predict(
    X_test
)

best_kernel_name = gpr_info["best_kernel"]

n_features = len(features)

ard = np.ones(n_features)

def build_gpr_kernel(name):

    if name == "RationalQuadratic":

        return (
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
            )
        )

    elif name == "RBF_ARD":

        return (
            ConstantKernel(
                1.0,
                (1e-2, 1e2)
            )
            *
            RBF(
                length_scale=ard,
                length_scale_bounds=(1e-2, 1e2),
            )
            +
            WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1e-5, 1e2),
            )
        )

    elif name == "Matern_1.5_ARD":

        return (
            ConstantKernel(
                1.0,
                (1e-2, 1e2)
            )
            *
            Matern(
                length_scale=ard,
                length_scale_bounds=(1e-2, 1e2),
                nu=1.5,
            )
            +
            WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1e-5, 1e2),
            )
        )

    elif name == "Matern_2.5_ARD":

        return (
            ConstantKernel(
                1.0,
                (1e-2, 1e2)
            )
            *
            Matern(
                length_scale=ard,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            +
            WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1e-5, 1e2),
            )
        )

    else:
        raise ValueError(
            f"Unknown frozen GPR kernel: {name}"
        )

gpr_kernel = build_gpr_kernel(
    best_kernel_name
)

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

gpr = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "gpr",
        GaussianProcessRegressor(
            kernel=gpr_kernel,
            normalize_y=True,
            alpha=1e-8,

            n_restarts_optimizer=1,

            random_state=42,
        )
    ),
])

print(
    "\nFitting frozen GPR:",
    best_kernel_name
)

gpr.fit(
    X_train,
    y_train
)

X_test_scaled = (
    gpr.named_steps["scale"]
    .transform(X_test)
)

gpr_model = gpr.named_steps["gpr"]

gpr_pred, gpr_std = (
    gpr_model.predict(
        X_test_scaled,
        return_std=True
    )
)

def metrics(
    y_true,
    y_pred,
    model_name
):

    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    nrmse = (
        rmse
        / np.mean(y_true)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mbe = np.mean(
        y_pred - y_true
    )

    return {
        "model": model_name,
        "r2": float(r2),
        "rmse_fnu": float(rmse),
        "nrmse": float(nrmse),
        "mae_fnu": float(mae),
        "mbe_fnu": float(mbe),
    }

xgb_metrics = metrics(
    y_test,
    xgb_pred,
    "XGBoost"
)

gpr_metrics = metrics(
    y_test,
    gpr_pred,
    "GPR"
)

results = pd.DataFrame([
    xgb_metrics,
    gpr_metrics,
])

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

predictions = pd.DataFrame({

    "sample_id":
        test["sample_id"].values,

    "site_no":
        test["site_no"].values,

    "station_nm":
        test["station_nm"].values,

    "observed_turbidity_fnu":
        y_test,

    "xgboost_prediction_fnu":
        xgb_pred,

    "gpr_prediction_fnu":
        gpr_pred,

    "gpr_prediction_std":
        gpr_std,

    "gpr_95_lower":
        gpr_pred
        - 1.96 * gpr_std,

    "gpr_95_upper":
        gpr_pred
        + 1.96 * gpr_std,
})

predictions.to_csv(
    PRED_OUTPUT,
    index=False
)

results.to_csv(
    METRIC_OUTPUT,
    index=False
)

summary = {

    "n_train":
        int(len(train)),

    "n_test":
        int(len(test)),

    "features":
        features,

    "xgboost_cv_r2":
        float(
            xgb_info["best_mean_cv_r2"]
        ),

    "gpr_cv_r2":
        float(
            gpr_info["best_mean_cv_r2"]
        ),

    "gpr_kernel":
        best_kernel_name,

    "final_test_results": {
        "XGBoost":
            xgb_metrics,

        "GPR":
            gpr_metrics,
    },
}

with open(
    SUMMARY_OUTPUT,
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
print(SUMMARY_OUTPUT)

print("\nIMPORTANT:")
print(
    "This is the final held-out test evaluation. "
    "Do not retune models based on these test results."
)


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# FILES
# ============================================================

SPLIT_FILE = Path(
    "01_data/processed/delaware_train_test_split.csv"
)

PRED_FILE = Path(
    "05_results/metrics/12_final_test_predictions.csv"
)

TABLE_DIR = Path(
    "05_results/tables"
)

FIG_DIR = Path(
    "06_figures"
)

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    SPLIT_FILE,
    low_memory=False,
    dtype={"site_no": "string"}
)

pred = pd.read_csv(
    PRED_FILE,
    low_memory=False,
    dtype={"site_no": "string"}
)

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

pred = (
    pred
    .sort_values("sample_id")
    .reset_index(drop=True)
)

test = (
    test
    .sort_values("sample_id")
    .reset_index(drop=True)
)


# ============================================================
# SAFETY CHECKS
# ============================================================

print("=" * 80)
print("FINAL-PREDICTION DIAGNOSTIC ANALYSIS")
print("=" * 80)

print("\nThis script DOES NOT train or tune any model.")

assert len(test) == 281, (
    f"Expected 281 test samples, found {len(test)}."
)

assert len(pred) == 281, (
    f"Expected 281 predictions, found {len(pred)}."
)

assert (
    pred["sample_id"].tolist()
    == test["sample_id"].tolist()
), "Prediction/test sample IDs do not match."

assert np.allclose(
    pred["observed_turbidity_fnu"].values,
    test["MeasurementValue"].astype(float).values,
    rtol=0,
    atol=1e-10,
), "Observed turbidity values do not match."

print("\nData integrity checks: PASSED")
print("Test observations:", len(test))
print("Training observations:", len(train))


# ============================================================
# BASIC ARRAYS
# ============================================================

y = pred[
    "observed_turbidity_fnu"
].astype(float).values

xgb = pred[
    "xgboost_prediction_fnu"
].astype(float).values

gpr = pred[
    "gpr_prediction_fnu"
].astype(float).values


# residual = prediction - observation
pred["xgb_residual"] = xgb - y
pred["gpr_residual"] = gpr - y

pred["xgb_abs_error"] = np.abs(
    pred["xgb_residual"]
)

pred["gpr_abs_error"] = np.abs(
    pred["gpr_residual"]
)

pred["xgb_squared_error"] = (
    pred["xgb_residual"] ** 2
)

pred["gpr_squared_error"] = (
    pred["gpr_residual"] ** 2
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    n = len(observed)

    rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted
        )
    )

    mae = mean_absolute_error(
        observed,
        predicted
    )

    mbe = np.mean(
        predicted - observed
    )

    mean_obs = np.mean(
        observed
    )

    nrmse = (
        rmse / mean_obs
        if mean_obs != 0
        else np.nan
    )

    # R² becomes unstable / meaningless
    # with very small groups or no target variance.
    if (
        n >= 5
        and np.var(observed) > 0
    ):
        r2 = r2_score(
            observed,
            predicted
        )
    else:
        r2 = np.nan

    return {
        "n": int(n),
        "r2": float(r2)
            if not np.isnan(r2)
            else np.nan,
        "rmse_fnu": float(rmse),
        "nrmse": float(nrmse),
        "mae_fnu": float(mae),
        "mbe_fnu": float(mbe),
    }


# ============================================================
# 1. CONFIRM OVERALL METRICS
# ============================================================

overall_rows = []

for model_name, prediction in [
    ("XGBoost", xgb),
    ("GPR", gpr),
]:

    metrics = calculate_metrics(
        y,
        prediction
    )

    metrics["model"] = model_name

    overall_rows.append(
        metrics
    )


overall = pd.DataFrame(
    overall_rows
)[[
    "model",
    "n",
    "r2",
    "rmse_fnu",
    "nrmse",
    "mae_fnu",
    "mbe_fnu",
]]


print("\n" + "=" * 80)
print("OVERALL FINAL TEST METRICS")
print("=" * 80)

print(
    overall.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


overall.to_csv(
    TABLE_DIR
    / "15_overall_test_metrics_check.csv",
    index=False
)


# ============================================================
# 2. WHICH MODEL WINS INDIVIDUAL OBSERVATIONS?
# ============================================================

xgb_abs_win = (
    pred["xgb_abs_error"]
    <
    pred["gpr_abs_error"]
)

gpr_abs_win = (
    pred["gpr_abs_error"]
    <
    pred["xgb_abs_error"]
)

ties = (
    pred["xgb_abs_error"]
    ==
    pred["gpr_abs_error"]
)


print("\n" + "=" * 80)
print("ABSOLUTE-ERROR WIN COUNTS")
print("=" * 80)

print(
    "XGBoost smaller absolute error:",
    int(xgb_abs_win.sum()),
    f"({100*xgb_abs_win.mean():.1f}%)"
)

print(
    "GPR smaller absolute error:",
    int(gpr_abs_win.sum()),
    f"({100*gpr_abs_win.mean():.1f}%)"
)

print(
    "Ties:",
    int(ties.sum())
)


# ============================================================
# 3. TURBIDITY RANGES
#
# Thresholds defined ONLY from training data.
#
# low       <= training median
# moderate  median to training 90th percentile
# high      > training 90th percentile
#
# This avoids choosing thresholds from test results.
# ============================================================

train_y = (
    train["MeasurementValue"]
    .astype(float)
)

median_threshold = float(
    train_y.quantile(0.50)
)

p90_threshold = float(
    train_y.quantile(0.90)
)


print("\n" + "=" * 80)
print("TRAINING-DEFINED TURBIDITY RANGES")
print("=" * 80)

print(
    f"Training median: {median_threshold:.3f} FNU"
)

print(
    f"Training 90th percentile: "
    f"{p90_threshold:.3f} FNU"
)


def assign_range(value):

    if value <= median_threshold:
        return "Low"

    elif value <= p90_threshold:
        return "Moderate"

    else:
        return "High"


pred["turbidity_range"] = (
    pred["observed_turbidity_fnu"]
    .apply(assign_range)
)


range_rows = []


for range_name in [
    "Low",
    "Moderate",
    "High",
]:

    subset = pred[
        pred["turbidity_range"]
        == range_name
    ]

    for model_name, col in [
        (
            "XGBoost",
            "xgboost_prediction_fnu"
        ),
        (
            "GPR",
            "gpr_prediction_fnu"
        ),
    ]:

        metrics = calculate_metrics(
            subset[
                "observed_turbidity_fnu"
            ],
            subset[col]
        )

        metrics[
            "turbidity_range"
        ] = range_name

        metrics[
            "model"
        ] = model_name

        range_rows.append(
            metrics
        )


range_results = pd.DataFrame(
    range_rows
)[[
    "turbidity_range",
    "model",
    "n",
    "r2",
    "rmse_fnu",
    "mae_fnu",
    "mbe_fnu",
]]


print("\n" + "=" * 80)
print("PERFORMANCE BY TURBIDITY RANGE")
print("=" * 80)

print(
    range_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


range_results.to_csv(
    TABLE_DIR
    / "15_performance_by_turbidity_range.csv",
    index=False
)


# ============================================================
# 4. STATION PERFORMANCE
# ============================================================

station_rows = []


for (
    site_no,
    station_name
), group in pred.groupby(
    [
        "site_no",
        "station_nm",
    ],
    dropna=False
):

    for model_name, col in [
        (
            "XGBoost",
            "xgboost_prediction_fnu"
        ),
        (
            "GPR",
            "gpr_prediction_fnu"
        ),
    ]:

        metrics = calculate_metrics(
            group[
                "observed_turbidity_fnu"
            ],
            group[col]
        )

        metrics["site_no"] = (
            str(site_no)
        )

        metrics["station_nm"] = (
            station_name
        )

        metrics["model"] = (
            model_name
        )

        metrics[
            "observed_mean_fnu"
        ] = float(
            group[
                "observed_turbidity_fnu"
            ].mean()
        )

        metrics[
            "observed_max_fnu"
        ] = float(
            group[
                "observed_turbidity_fnu"
            ].max()
        )

        station_rows.append(
            metrics
        )


station_results = pd.DataFrame(
    station_rows
)[[
    "site_no",
    "station_nm",
    "model",
    "n",
    "observed_mean_fnu",
    "observed_max_fnu",
    "r2",
    "rmse_fnu",
    "mae_fnu",
    "mbe_fnu",
]]


station_results = (
    station_results
    .sort_values(
        [
            "site_no",
            "model",
        ]
    )
)


print("\n" + "=" * 80)
print("PERFORMANCE BY STATION")
print("=" * 80)

print(
    station_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


station_results.to_csv(
    TABLE_DIR
    / "15_performance_by_station.csv",
    index=False
)


# ============================================================
# 5. CV FOLD VS TEST TURBIDITY DISTRIBUTIONS
# ============================================================

distribution_rows = []


# Training CV folds
for fold in sorted(
    train["cv_fold"]
    .astype(int)
    .unique()
):

    values = (
        train[
            train["cv_fold"].astype(int)
            == fold
        ]["MeasurementValue"]
        .astype(float)
    )

    distribution_rows.append({
        "group":
            f"CV Fold {fold + 1}",
        "n":
            len(values),
        "mean_fnu":
            values.mean(),
        "std_fnu":
            values.std(),
        "median_fnu":
            values.median(),
        "p90_fnu":
            values.quantile(0.90),
        "p95_fnu":
            values.quantile(0.95),
        "p99_fnu":
            values.quantile(0.99),
        "max_fnu":
            values.max(),
    })


# Held-out test
values = (
    test["MeasurementValue"]
    .astype(float)
)

distribution_rows.append({
    "group":
        "Held-out Test",
    "n":
        len(values),
    "mean_fnu":
        values.mean(),
    "std_fnu":
        values.std(),
    "median_fnu":
        values.median(),
    "p90_fnu":
        values.quantile(0.90),
    "p95_fnu":
        values.quantile(0.95),
    "p99_fnu":
        values.quantile(0.99),
    "max_fnu":
        values.max(),
})


distribution = pd.DataFrame(
    distribution_rows
)


print("\n" + "=" * 80)
print("CV-FOLD VS TEST TURBIDITY DISTRIBUTION")
print("=" * 80)

print(
    distribution.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


distribution.to_csv(
    TABLE_DIR
    / "15_cv_test_turbidity_distribution.csv",
    index=False
)


# ============================================================
# 6. HIGH-TURBIDITY ERROR CONTRIBUTION
#
# This is particularly useful for explaining:
#
# XGB lower MAE
# GPR lower RMSE
# ============================================================

high = pred[
    pred["turbidity_range"]
    == "High"
].copy()

non_high = pred[
    pred["turbidity_range"]
    != "High"
].copy()


error_contribution = pd.DataFrame([
    {
        "group":
            "Low + Moderate",

        "n":
            len(non_high),

        "xgb_mean_abs_error":
            non_high[
                "xgb_abs_error"
            ].mean(),

        "gpr_mean_abs_error":
            non_high[
                "gpr_abs_error"
            ].mean(),

        "xgb_mean_squared_error":
            non_high[
                "xgb_squared_error"
            ].mean(),

        "gpr_mean_squared_error":
            non_high[
                "gpr_squared_error"
            ].mean(),
    },

    {
        "group":
            "High",

        "n":
            len(high),

        "xgb_mean_abs_error":
            high[
                "xgb_abs_error"
            ].mean(),

        "gpr_mean_abs_error":
            high[
                "gpr_abs_error"
            ].mean(),

        "xgb_mean_squared_error":
            high[
                "xgb_squared_error"
            ].mean(),

        "gpr_mean_squared_error":
            high[
                "gpr_squared_error"
            ].mean(),
    },
])


print("\n" + "=" * 80)
print("ERROR CONTRIBUTION BY TURBIDITY LEVEL")
print("=" * 80)

print(
    error_contribution.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


error_contribution.to_csv(
    TABLE_DIR
    / "15_error_contribution_by_turbidity.csv",
    index=False
)


# ============================================================
# 7. GPR UNCERTAINTY DIAGNOSTIC
# ============================================================

if (
    "gpr_prediction_std"
    in pred.columns
):

    pred[
        "gpr_interval_contains_observation"
    ] = (
        (
            pred[
                "observed_turbidity_fnu"
            ]
            >=
            pred["gpr_95_lower"]
        )
        &
        (
            pred[
                "observed_turbidity_fnu"
            ]
            <=
            pred["gpr_95_upper"]
        )
    )

    uncertainty_rows = []

    for range_name in [
        "Low",
        "Moderate",
        "High",
    ]:

        subset = pred[
            pred["turbidity_range"]
            == range_name
        ]

        uncertainty_rows.append({
            "turbidity_range":
                range_name,

            "n":
                len(subset),

            "mean_predictive_sd_fnu":
                subset[
                    "gpr_prediction_std"
                ].mean(),

            "95_interval_coverage":
                subset[
                    "gpr_interval_contains_observation"
                ].mean(),
        })


    uncertainty = pd.DataFrame(
        uncertainty_rows
    )


    print("\n" + "=" * 80)
    print("GPR UNCERTAINTY BY TURBIDITY RANGE")
    print("=" * 80)

    print(
        uncertainty.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


    uncertainty.to_csv(
        TABLE_DIR
        / "15_gpr_uncertainty_by_range.csv",
        index=False
    )


# ============================================================
# 8. FIGURE — OBSERVED VS XGBOOST
# ============================================================

max_axis = max(
    np.max(y),
    np.max(xgb),
    np.max(gpr)
)

plt.figure(figsize=(7, 6))

plt.scatter(
    y,
    xgb,
    alpha=0.65,
    s=28
)

plt.plot(
    [0, max_axis],
    [0, max_axis],
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "Observed turbidity (FNU)"
)

plt.ylabel(
    "XGBoost predicted turbidity (FNU)"
)

plt.title(
    "Observed vs Predicted Turbidity — XGBoost"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "15_observed_vs_predicted_xgboost.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. FIGURE — OBSERVED VS GPR
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y,
    gpr,
    alpha=0.65,
    s=28
)

plt.plot(
    [0, max_axis],
    [0, max_axis],
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "Observed turbidity (FNU)"
)

plt.ylabel(
    "GPR predicted turbidity (FNU)"
)

plt.title(
    "Observed vs Predicted Turbidity — GPR"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "15_observed_vs_predicted_gpr.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. RESIDUAL FIGURE — XGBOOST
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y,
    pred["xgb_residual"],
    alpha=0.65,
    s=28
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "Observed turbidity (FNU)"
)

plt.ylabel(
    "Residual: predicted - observed (FNU)"
)

plt.title(
    "XGBoost Residuals"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "15_residuals_xgboost.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. RESIDUAL FIGURE — GPR
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y,
    pred["gpr_residual"],
    alpha=0.65,
    s=28
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "Observed turbidity (FNU)"
)

plt.ylabel(
    "Residual: predicted - observed (FNU)"
)

plt.title(
    "GPR Residuals"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "15_residuals_gpr.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. CV FOLD / TEST DISTRIBUTION BOXPLOT
# ============================================================

plot_groups = []
plot_labels = []


for fold in sorted(
    train["cv_fold"]
    .astype(int)
    .unique()
):

    values = (
        train[
            train["cv_fold"].astype(int)
            == fold
        ]["MeasurementValue"]
        .astype(float)
        .values
    )

    plot_groups.append(
        values
    )

    plot_labels.append(
        f"Fold {fold + 1}"
    )


plot_groups.append(
    test[
        "MeasurementValue"
    ].astype(float).values
)

plot_labels.append(
    "Test"
)


plt.figure(
    figsize=(9, 6)
)

plt.boxplot(
    plot_groups,
    showfliers=True
)

plt.xticks(
    range(1, len(plot_labels) + 1),
    plot_labels
)
plt.ylabel(
    "Observed turbidity (FNU)"
)

plt.title(
    "Turbidity Distribution: CV Folds vs Test Set"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "15_cv_test_turbidity_boxplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE ENRICHED PREDICTION TABLE
# ============================================================

pred.to_csv(
    TABLE_DIR
    / "15_final_predictions_with_errors.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nNo model was trained or tuned.")

print("\nTables saved to:")
print(TABLE_DIR)

print("\nFigures saved to:")
print(FIG_DIR)

print(
    "\nPrimary models remain unchanged."
)
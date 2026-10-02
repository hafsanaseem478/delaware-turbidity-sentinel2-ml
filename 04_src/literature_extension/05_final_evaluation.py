from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBRegressor

DATA_FILE = Path(
    "01_data/processed/delaware_santos_features.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/31_delaware_santos_features.json"
)

BASELINE_CONFIG_FILE = Path(
    "05_results/metrics/32_delaware_santos_best_configs.json"
)

GPR_CONFIG_FILE = Path(
    "05_results/metrics/34_delaware_gpr_config.json"
)

SUMMARY_OUTPUT = Path(
    "05_results/tables/35_delaware_final_test_metrics.csv"
)

PREDICTION_OUTPUT = Path(
    "05_results/tables/35_delaware_final_test_predictions.csv"
)

CONFIG_OUTPUT = Path(
    "05_results/metrics/35_delaware_final_test.json"
)

SUMMARY_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "MeasurementValue"

RANDOM_STATE = 42

for path in [

    DATA_FILE,
    FEATURE_FILE,
    BASELINE_CONFIG_FILE,
    GPR_CONFIG_FILE,

]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
    dtype={
        "site_no": "string"
    }
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)

with open(
    BASELINE_CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    baseline_info = json.load(f)

with open(
    GPR_CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    gpr_info = json.load(f)

FEATURES = feature_info[
    "final_model_features"
]

if len(FEATURES) != 20:

    raise RuntimeError(
        f"Expected 20 frozen predictors, found {len(FEATURES)}."
    )

train = (

    df.loc[
        df["split"]
        .astype(str)
        .eq("train")
    ]

    .copy()

)

test = (

    df.loc[
        df["split"]
        .astype(str)
        .eq("test")
    ]

    .copy()

)

if len(train) != 1121:

    raise RuntimeError(
        f"Expected 1121 training rows, found {len(train)}."
    )

if len(test) != 281:

    raise RuntimeError(
        f"Expected 281 test rows, found {len(test)}."
    )

X_train = (

    train[
        FEATURES
    ]

    .astype(float)

)

X_test = (

    test[
        FEATURES
    ]

    .astype(float)

)

y_train = (

    pd.to_numeric(
        train[TARGET],
        errors="raise"
    )

    .astype(float)

    .to_numpy()

)

y_test = (

    pd.to_numeric(
        test[TARGET],
        errors="raise"
    )

    .astype(float)

    .to_numpy()

)

if not np.isfinite(
    X_train.to_numpy()
).all():

    raise RuntimeError(
        "Invalid training predictor values."
    )

if not np.isfinite(
    X_test.to_numpy()
).all():

    raise RuntimeError(
        "Invalid test predictor values."
    )

print("=" * 100)

print(
    "DELAWARE â€” FINAL HELD-OUT TEST"
)

print("=" * 100)

print(
    "\nTraining rows:",
    len(train)
)

print(
    "Test rows:",
    len(test)
)

print(
    "Frozen predictors:",
    len(FEATURES)
)

print(
    "\nObserved TEST turbidity:"
)

print(
    f" Mean   = {np.mean(y_test):.3f} FNU"
)

print(
    f" SD     = {np.std(y_test, ddof=1):.3f} FNU"
)

print(
    f" Median = {np.median(y_test):.3f} FNU"
)

print(
    f" Max    = {np.max(y_test):.3f} FNU"
)

def make_preprocessor(name):

    if name == "None":

        return "passthrough"

    if name == "Standard":

        return StandardScaler()

    if name == "YeoJohnson":

        return PowerTransformer(
            method="yeo-johnson",
            standardize=False
        )

    if name == "YeoJohnson+Standard":

        return Pipeline(
            steps=[
                (
                    "yeo_johnson",
                    PowerTransformer(
                        method="yeo-johnson",
                        standardize=False,
                    ),
                ),
                (
                    "standard",
                    StandardScaler(),
                ),
            ]
        )

    raise ValueError(
        f"Unknown preprocessing: {name}"
    )

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

    r2 = float(

        r2_score(
            observed,
            predicted
        )

    )

    rmse = float(

        np.sqrt(

            mean_squared_error(
                observed,
                predicted
            )

        )

    )

    mean_observed = float(
        np.mean(observed)
    )

    nrmse = float(

        rmse
        /
        mean_observed

    )

    mae = float(

        mean_absolute_error(
            observed,
            predicted
        )

    )

    mbe = float(

        np.mean(
            predicted
            -
            observed
        )

    )

    return {

        "R2":
            r2,

        "RMSE_FNU":
            rmse,

        "nRMSE":
            nrmse,

        "MAE_FNU":
            mae,

        "MBE_FNU":
            mbe,

    }

def build_baseline_model(
    model_name,
    parameters
):

    if model_name == "RandomForest":

        return RandomForestRegressor(

            **parameters,

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1,

        )

    if model_name == "SVR":

        return SVR(

            kernel="rbf",

            **parameters,

        )

    if model_name == "kNN":

        return KNeighborsRegressor(

            **parameters

        )

    if model_name == "XGBoost":

        return XGBRegressor(

            **parameters,

            objective=
                "reg:squarederror",

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1,

            verbosity=
                0,

        )

    if model_name == "LASSO":

        return Lasso(

            **parameters,

            max_iter=
                100000,

            random_state=
                RANDOM_STATE,

        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )

metric_rows = []

prediction_table = pd.DataFrame({

    "sample_id":

        (
            test["sample_id"].astype(str)
            if "sample_id" in test.columns
            else test.index.astype(str)
        ),

    "site_no":

        (
            test["site_no"].astype(str)
            if "site_no" in test.columns
            else ""
        ),

    "station_nm":

        (
            test["station_nm"].astype(str)
            if "station_nm" in test.columns
            else ""
        ),

    "observed_turbidity_FNU":
        y_test,

})

BASELINE_ORDER = [

    "RandomForest",
    "SVR",
    "kNN",
    "XGBoost",
    "LASSO",

]

best_configs = baseline_info[
    "best_configs"
]

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

print(
    "\n"
    + "=" * 100
)

print(
    "SANTOS-STYLE BASELINE TEST RESULTS"
)

print(
    "=" * 100
)

for model_name in BASELINE_ORDER:

    if model_name not in best_configs:

        raise KeyError(

            f"{model_name} not found in "
            "32_delaware_santos_best_configs.json"

        )

    model_config = best_configs[
        model_name
    ]

    preprocessing_name = model_config[
        "preprocessing"
    ]

    parameters = model_config[
        "parameters"
    ]

    train_cv_r2 = float(
        model_config[
            "cv_r2"
        ]
    )

    print(
        f"\n{model_name}"
    )

    print(
        f" Frozen preprocessing: "
        f"{preprocessing_name}"
    )

    print(
        f" Training CV RÂ²: "
        f"{train_cv_r2:.6f}"
    )

    print(
        f" Frozen parameters: "
        f"{parameters}"
    )

    estimator = build_baseline_model(

        model_name,

        parameters

    )

    pipeline = Pipeline(

        steps=[

            (
                "prep",

                make_preprocessor(
                    preprocessing_name
                )

            ),

            (
                "model",

                estimator

            ),

        ]

    )

    pipeline.fit(

        X_train,

        y_train,

    )

    predictions = pipeline.predict(

        X_test

    )

    metrics = calculate_metrics(

        y_test,

        predictions

    )

    metric_rows.append({

        "model":
            model_name,

        "type":
            "Santos-style baseline",

        "preprocessing":
            preprocessing_name,

        "training_cv_r2":
            train_cv_r2,

        **metrics,

    })

    prediction_table[
        f"pred_{model_name}"
    ] = predictions

    print(
        f" TEST RÂ²    = {metrics['R2']:.6f}"
    )

    print(
        f" TEST RMSE  = {metrics['RMSE_FNU']:.6f} FNU"
    )

    print(
        f" TEST nRMSE = {metrics['nRMSE']:.6f}"
    )

    print(
        f" TEST MAE   = {metrics['MAE_FNU']:.6f} FNU"
    )

    print(
        f" TEST MBE   = {metrics['MBE_FNU']:.6f} FNU"
    )

print(
    "\n"
    + "=" * 100
)

print(
    "GPR EXTENSION â€” FINAL TEST"
)

print(
    "=" * 100
)

best_gpr_kernel = gpr_info[
    "best_kernel"
]

if best_gpr_kernel != "RationalQuadratic":

    raise RuntimeError(

        "Expected frozen GPR kernel "
        f"'RationalQuadratic', found {best_gpr_kernel}."

    )

gpr_cv_r2 = float(
    gpr_info[
        "best_cv_r2"
    ]
)

print(
    "\nFrozen kernel family:",
    best_gpr_kernel
)

print(
    "Training CV RÂ²:",
    f"{gpr_cv_r2:.6f}"
)

gpr_kernel = (

    ConstantKernel(

        constant_value=1.0,

        constant_value_bounds=(
            1e-3,
            1e3
        )

    )

    *

    RationalQuadratic(

        length_scale=1.0,

        alpha=1.0,

        length_scale_bounds=(
            1e-2,
            1e3
        ),

        alpha_bounds=(
            1e-2,
            1e3
        ),

    )

    +

    WhiteKernel(

        noise_level=1.0,

        noise_level_bounds=(
            1e-5,
            1e3
        )

    )

)

gpr_scaler = StandardScaler()

X_train_gpr = gpr_scaler.fit_transform(

    X_train

)

X_test_gpr = gpr_scaler.transform(

    X_test

)

gpr_model = GaussianProcessRegressor(

    kernel=
        gpr_kernel,

    alpha=
        1e-6,

    normalize_y=
        True,

    n_restarts_optimizer=
        1,

    random_state=
        RANDOM_STATE,

)

gpr_model.fit(

    X_train_gpr,

    y_train

)

print(
    "\nFinal optimized kernel:"
)

print(
    gpr_model.kernel_
)

gpr_prediction = gpr_model.predict(

    X_test_gpr

)

gpr_metrics = calculate_metrics(

    y_test,

    gpr_prediction

)

metric_rows.append({

    "model":
        "GPR",

    "type":
        "Proposed extension",

    "preprocessing":
        "Standard",

    "training_cv_r2":
        gpr_cv_r2,

    **gpr_metrics,

})

prediction_table[
    "pred_GPR"
] = gpr_prediction

print(
    f"\n TEST RÂ²    = {gpr_metrics['R2']:.6f}"
)

print(
    f" TEST RMSE  = {gpr_metrics['RMSE_FNU']:.6f} FNU"
)

print(
    f" TEST nRMSE = {gpr_metrics['nRMSE']:.6f}"
)

print(
    f" TEST MAE   = {gpr_metrics['MAE_FNU']:.6f} FNU"
)

print(
    f" TEST MBE   = {gpr_metrics['MBE_FNU']:.6f} FNU"
)

results = pd.DataFrame(
    metric_rows
)

results = results.sort_values(

    "R2",

    ascending=False

).reset_index(
    drop=True
)

print(
    "\n"
    + "=" * 100
)

print(
    "FINAL HELD-OUT TEST COMPARISON"
)

print(
    "=" * 100
)

display_columns = [

    "model",

    "type",

    "preprocessing",

    "training_cv_r2",

    "R2",

    "RMSE_FNU",

    "nRMSE",

    "MAE_FNU",

    "MBE_FNU",

]

print(

    results[
        display_columns
    ]

    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )

)

results.to_csv(

    SUMMARY_OUTPUT,

    index=False

)

prediction_table.to_csv(

    PREDICTION_OUTPUT,

    index=False

)

final_metadata = {

    "study":
        (
            "Delaware River turbidity â€” "
            "Santos-style baselines plus GPR extension"
        ),

    "n_total":
        int(
            len(df)
        ),

    "n_train":
        int(
            len(train)
        ),

    "n_test":
        int(
            len(test)
        ),

    "n_features":
        int(
            len(FEATURES)
        ),

    "features":
        FEATURES,

    "target":
        TARGET,

    "test_evaluation":
        (
            "Single held-out evaluation after "
            "training/CV model selection"
        ),

    "nrmse_definition":
        "RMSE / mean observed test turbidity",

    "mbe_definition":
        "mean(predicted - observed)",

    "gpr_final_kernel":
        str(
            gpr_model.kernel_
        ),

    "results":
        results.to_dict(
            orient="records"
        ),

}

with open(

    CONFIG_OUTPUT,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        final_metadata,

        f,

        indent=4

    )

print(
    "\n"
    + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)

print(
    SUMMARY_OUTPUT
)

print(
    PREDICTION_OUTPUT
)

print(
    CONFIG_OUTPUT
)

print(
    "\nFINAL TEST EVALUATION COMPLETE."
)

print(
    "Do not retune models based on these test results."
)


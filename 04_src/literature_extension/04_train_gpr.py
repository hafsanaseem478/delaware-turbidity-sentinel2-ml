from pathlib import Path
import json
import math
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning

DATA_FILE = Path(
    "01_data/processed/delaware_santos_features.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/31_delaware_santos_features.json"
)

SUMMARY_OUTPUT = Path(
    "05_results/tables/34_delaware_gpr_cv_summary.csv"
)

FOLD_OUTPUT = Path(
    "05_results/tables/34_delaware_gpr_fold_results.csv"
)

CONFIG_OUTPUT = Path(
    "05_results/metrics/34_delaware_gpr_config.json"
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

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Missing dataset:\n{DATA_FILE}"
    )

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Missing feature metadata:\n{FEATURE_FILE}"
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
        f"Expected 281 held-out rows, found {len(test)}."
    )

X = (

    train[
        FEATURES
    ]

    .astype(float)

    .to_numpy()

)

y = (

    pd.to_numeric(
        train[TARGET],
        errors="raise"
    )

    .astype(float)

    .to_numpy()

)

if not np.isfinite(X).all():

    raise RuntimeError(
        "Non-finite predictor values detected."
    )

if not np.isfinite(y).all():

    raise RuntimeError(
        "Non-finite target values detected."
    )

print("=" * 100)

print(
    "DELAWARE â€” GPR EXTENSION TRAINING"
)

print("=" * 100)

print(
    "\nTraining rows:",
    len(train)
)

print(
    "Held-out test rows:",
    len(test)
)

print(
    "Frozen predictors:",
    len(FEATURES)
)

print(
    "\nTEST SET STATUS: LOCKED / NOT USED"
)

print(
    "\nTarget remains in original FNU."
)

print(
    "Predictors are standardized within each CV training fold."
)

cv = KFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE,

)

KERNELS = {

    "Matern32":

        ConstantKernel(

            constant_value=1.0,

            constant_value_bounds=(
                1e-3,
                1e3
            )

        )

        *

        Matern(

            length_scale=1.0,

            length_scale_bounds=(
                1e-2,
                1e3
            ),

            nu=1.5,

        )

        +

        WhiteKernel(

            noise_level=1.0,

            noise_level_bounds=(
                1e-5,
                1e3
            )

        ),

    "RationalQuadratic":

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

        ),

    "RBF":

        ConstantKernel(

            constant_value=1.0,

            constant_value_bounds=(
                1e-3,
                1e3
            )

        )

        *

        RBF(

            length_scale=1.0,

            length_scale_bounds=(
                1e-2,
                1e3
            )

        )

        +

        WhiteKernel(

            noise_level=1.0,

            noise_level_bounds=(
                1e-5,
                1e3
            )

        ),

}

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

summary_rows = []

fold_rows = []

for kernel_name, kernel_template in KERNELS.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"GPR KERNEL â€” {kernel_name}"
    )

    print(
        "=" * 100
    )

    fold_scores = []

    fold_kernels = []

    start_kernel = time.time()

    for fold_number, (

        fit_idx,

        validation_idx,

    ) in enumerate(

        cv.split(X),

        start=1,

    ):

        print(
            f"\nFold {fold_number}/5..."
        )

        X_fit = X[
            fit_idx
        ]

        X_validation = X[
            validation_idx
        ]

        y_fit = y[
            fit_idx
        ]

        y_validation = y[
            validation_idx
        ]

        scaler = StandardScaler()

        X_fit_scaled = scaler.fit_transform(
            X_fit
        )

        X_validation_scaled = scaler.transform(
            X_validation
        )

        from sklearn.base import clone

        kernel = clone(
            kernel_template
        )

        model = GaussianProcessRegressor(

            kernel=kernel,

            alpha=1e-6,

            normalize_y=True,

            n_restarts_optimizer=1,

            random_state=RANDOM_STATE,

        )

        fold_start = time.time()

        model.fit(

            X_fit_scaled,

            y_fit,

        )

        predictions = model.predict(

            X_validation_scaled

        )

        fold_minutes = (

            time.time()
            -
            fold_start

        ) / 60.0

        if not np.isfinite(
            predictions
        ).all():

            raise RuntimeError(

                f"Non-finite predictions for "
                f"{kernel_name}, fold {fold_number}."

            )

        score = float(

            r2_score(

                y_validation,

                predictions,

            )

        )

        fold_scores.append(
            score
        )

        optimized_kernel = str(
            model.kernel_
        )

        fold_kernels.append(
            optimized_kernel
        )

        fold_rows.append(

            {

                "kernel":
                    kernel_name,

                "fold":
                    fold_number,

                "r2":
                    score,

                "n_train":
                    len(
                        fit_idx
                    ),

                "n_validation":
                    len(
                        validation_idx
                    ),

                "runtime_minutes":
                    fold_minutes,

                "optimized_kernel":
                    optimized_kernel,

            }

        )

        print(
            f" RÂ² = {score:.6f}"
        )

        print(
            f" Time = {fold_minutes:.2f} min"
        )

        print(
            f" Optimized kernel = {optimized_kernel}"
        )

    mean_r2 = float(

        np.mean(
            fold_scores
        )

    )

    sd_r2 = float(

        np.std(
            fold_scores,
            ddof=1
        )

    )

    se_r2 = float(

        sd_r2
        /
        math.sqrt(
            len(fold_scores)
        )

    )

    total_minutes = (

        time.time()
        -
        start_kernel

    ) / 60.0

    summary_rows.append(

        {

            "kernel":
                kernel_name,

            "mean_cv_r2":
                mean_r2,

            "cv_sd":
                sd_r2,

            "cv_se":
                se_r2,

            "runtime_minutes":
                total_minutes,

            "fold_1_r2":
                fold_scores[0],

            "fold_2_r2":
                fold_scores[1],

            "fold_3_r2":
                fold_scores[2],

            "fold_4_r2":
                fold_scores[3],

            "fold_5_r2":
                fold_scores[4],

        }

    )

    print(
        "\nMean CV RÂ²:",
        f"{mean_r2:.6f}"
    )

    print(
        "CV SD:",
        f"{sd_r2:.6f}"
    )

    print(
        "CV SE:",
        f"{se_r2:.6f}"
    )

    print(
        "Total kernel runtime:",
        f"{total_minutes:.2f} min"
    )

summary = (

    pd.DataFrame(
        summary_rows
    )

    .sort_values(
        "mean_cv_r2",
        ascending=False
    )

    .reset_index(
        drop=True
    )

)

fold_table = pd.DataFrame(
    fold_rows
)

print(
    "\n"
    + "=" * 100
)

print(
    "GPR KERNEL COMPARISON"
)

print(
    "=" * 100
)

print(

    summary[
        [
            "kernel",
            "mean_cv_r2",
            "cv_sd",
            "cv_se",
        ]
    ]

    .to_string(
        index=False
    )

)

best_kernel = str(

    summary.loc[
        0,
        "kernel"
    ]

)

best_cv_r2 = float(

    summary.loc[
        0,
        "mean_cv_r2"
    ]

)

best_cv_sd = float(

    summary.loc[
        0,
        "cv_sd"
    ]

)

print(
    "\nBEST GPR KERNEL:",
    best_kernel
)

print(
    "BEST GPR CV RÂ²:",
    f"{best_cv_r2:.6f}"
)

print(
    "\nFitting frozen GPR configuration "
    "on all 1121 training rows..."
)

final_scaler = StandardScaler()

X_train_scaled = (

    final_scaler
    .fit_transform(
        X
    )

)

from sklearn.base import clone

final_kernel = clone(

    KERNELS[
        best_kernel
    ]

)

final_model = GaussianProcessRegressor(

    kernel=
        final_kernel,

    alpha=
        1e-6,

    normalize_y=
        True,

    n_restarts_optimizer=
        1,

    random_state=
        RANDOM_STATE,

)

final_model.fit(

    X_train_scaled,

    y,

)

final_optimized_kernel = str(

    final_model.kernel_

)

print(
    "\nFinal optimized kernel:"
)

print(
    final_optimized_kernel
)

summary.to_csv(

    SUMMARY_OUTPUT,

    index=False

)

fold_table.to_csv(

    FOLD_OUTPUT,

    index=False

)

configuration = {

    "study":
        "Delaware turbidity GPR extension",

    "n_train":
        int(
            len(train)
        ),

    "n_test_locked":
        int(
            len(test)
        ),

    "test_used":
        False,

    "features":
        FEATURES,

    "n_features":
        len(FEATURES),

    "feature_scaling":
        "StandardScaler fitted within each CV training fold",

    "target_transform":
        "None",

    "gpr_normalize_y":
        True,

    "selection_metric":
        "R2",

    "cross_validation":
        "5-fold shuffled KFold",

    "candidate_kernels":
        list(
            KERNELS.keys()
        ),

    "best_kernel":
        best_kernel,

    "best_cv_r2":
        best_cv_r2,

    "best_cv_sd":
        best_cv_sd,

    "alpha":
        1e-6,

    "n_restarts_optimizer":
        1,

    "final_optimized_kernel":
        final_optimized_kernel,

}

with open(

    CONFIG_OUTPUT,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        configuration,

        file,

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
    FOLD_OUTPUT
)

print(
    CONFIG_OUTPUT
)

print(
    "\nGPR TRAINING/MODEL SELECTION COMPLETE."
)

print(
    "The held-out 281-row TEST SET has NOT been evaluated."
)

print(
    "Next step: evaluate the five Santos baselines + GPR "
    "once on the common held-out test set."
)


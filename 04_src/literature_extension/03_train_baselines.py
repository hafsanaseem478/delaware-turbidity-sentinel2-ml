from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    GridSearchCV,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBRegressor

DATA_FILE = Path(
    "01_data/processed/delaware_santos_features.csv"
)

FEATURE_FILE = Path(
    "05_results/metrics/31_delaware_santos_features.json"
)

SUMMARY_OUTPUT = Path(
    "05_results/tables/32_delaware_santos_cv_summary.csv"
)

CONFIG_OUTPUT = Path(
    "05_results/metrics/32_delaware_santos_best_configs.json"
)

CV_DIR = Path(
    "05_results/tables/32_cv_details"
)

SUMMARY_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

CV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "MeasurementValue"

RANDOM_STATE = 42

CV_JOBS = 4

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
) as file:

    feature_metadata = json.load(
        file
    )

FEATURES = feature_metadata[
    "final_model_features"
]

print("=" * 100)
print("DELAWARE â€” SANTOS MODEL GRID SEARCH")
print("=" * 100)

print(
    "\nFrozen predictors:"
)

for i, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )

print(
    "\nNumber of predictors:",
    len(FEATURES)
)

if len(FEATURES) != 20:

    raise RuntimeError(

        "Expected 20 frozen predictors from "
        "Santos feature selection."

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

print(
    "\nTraining rows:",
    len(train)
)

print(
    "Held-out test rows:",
    len(test)
)

print(
    "\nTEST SET STATUS: LOCKED / NOT USED"
)

missing = [

    feature

    for feature in FEATURES

    if feature not in train.columns

]

if missing:

    raise KeyError(

        "Missing frozen predictors:\n"

        +

        "\n".join(
            missing
        )

    )

X_train = (

    train[
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

)

if not np.isfinite(
    X_train.to_numpy()
).all():

    raise RuntimeError(
        "Non-finite predictor values found."
    )

if not np.isfinite(
    y_train.to_numpy()
).all():

    raise RuntimeError(
        "Non-finite target values found."
    )

cv = KFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE,

)

def make_preprocessor(
    name
):

    if name == "None":

        return "passthrough"

    if name == "Standard":

        return StandardScaler()

    if name == "YeoJohnson":

        return PowerTransformer(

            method="yeo-johnson",

            standardize=False,

        )

    if name == "YeoJohnson+Standard":

        return Pipeline(

            steps=[

                (
                    "yeo_johnson",

                    PowerTransformer(
                        method="yeo-johnson",
                        standardize=False,
                    )

                ),

                (
                    "standard",

                    StandardScaler()

                ),

            ]

        )

    raise ValueError(
        f"Unknown preprocessing: {name}"
    )

PREPROCESSING_OPTIONS = [

    "None",

    "Standard",

    "YeoJohnson",

    "YeoJohnson+Standard",

]

MODELS = {

    "XGBoost": {

        "estimator":

            XGBRegressor(

                objective=
                    "reg:squarederror",

                random_state=
                    RANDOM_STATE,

                n_jobs=
                    1,

                verbosity=
                    0,

            ),

        "grid": {

            "model__n_estimators":
                [
                    200,
                    400,
                ],

            "model__learning_rate":
                [
                    0.03,
                    0.05,
                    0.10,
                ],

            "model__max_depth":
                [
                    2,
                    3,
                    4,
                ],

            "model__min_child_weight":
                [
                    1,
                    3,
                ],

            "model__subsample":
                [
                    0.8,
                    1.0,
                ],

            "model__colsample_bytree":
                [
                    0.8,
                    1.0,
                ],

        },

    },

    "RandomForest": {

        "estimator":

            RandomForestRegressor(

                random_state=
                    RANDOM_STATE,

                n_jobs=
                    1,

            ),

        "grid": {

            "model__n_estimators":
                [
                    300,
                    600,
                ],

            "model__max_depth":
                [
                    None,
                    10,
                    20,
                ],

            "model__min_samples_leaf":
                [
                    1,
                    2,
                    4,
                ],

            "model__max_features":
                [
                    "sqrt",
                    0.7,
                    1.0,
                ],

        },

    },

    "SVR": {

        "estimator":

            SVR(
                kernel="rbf"
            ),

        "grid": {

            "model__C":
                [
                    1,
                    10,
                    100,
                ],

            "model__gamma":
                [
                    "scale",
                    0.01,
                    0.1,
                ],

            "model__epsilon":
                [
                    0.1,
                    0.5,
                    1.0,
                ],

        },

    },

    "kNN": {

        "estimator":

            KNeighborsRegressor(),

        "grid": {

            "model__n_neighbors":
                [
                    3,
                    5,
                    7,
                    11,
                    15,
                ],

            "model__weights":
                [
                    "uniform",
                    "distance",
                ],

            "model__p":
                [
                    1,
                    2,
                ],

        },

    },

    "LASSO": {

        "estimator":

            Lasso(

                max_iter=
                    100000,

                random_state=
                    RANDOM_STATE,

            ),

        "grid": {

            "model__alpha":
                [
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],

        },

    },

}

summary_rows = []

best_configs = {}

warnings.filterwarnings(

    "ignore",

    category=
        ConvergenceWarning,

)

for model_name, model_info in MODELS.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        model_name
    )

    print(
        "=" * 100
    )

    model_results = []

    for prep_name in PREPROCESSING_OPTIONS:

        print(
            f"\nPreprocessing: {prep_name}"
        )

        pipeline = Pipeline(

            steps=[

                (
                    "prep",

                    make_preprocessor(
                        prep_name
                    )

                ),

                (
                    "model",

                    model_info[
                        "estimator"
                    ]

                ),

            ]

        )

        grid_search = GridSearchCV(

            estimator=
                pipeline,

            param_grid=
                model_info[
                    "grid"
                ],

            scoring=
                "r2",

            cv=
                cv,

            n_jobs=
                CV_JOBS,

            refit=
                True,

            return_train_score=
                False,

            verbose=
                0,

            error_score=
                "raise",

        )

        start_time = time.time()

        grid_search.fit(

            X_train,

            y_train,

        )

        elapsed_minutes = (

            time.time()
            -
            start_time

        ) / 60.0

        best_index = (

            grid_search
            .best_index_

        )

        mean_r2 = float(

            grid_search
            .best_score_

        )

        sd_r2 = float(

            grid_search
            .cv_results_[
                "std_test_score"
            ][
                best_index
            ]

        )

        se_r2 = (

            sd_r2
            /
            np.sqrt(
                5
            )

        )

        print(
            f" Best CV RÂ² = {mean_r2:.6f}"
        )

        print(
            f" CV SD      = {sd_r2:.6f}"
        )

        print(
            f" CV SE      = {se_r2:.6f}"
        )

        print(
            f" Time       = {elapsed_minutes:.2f} min"
        )

        print(
            " Best parameters:"
        )

        for key, value in (

            grid_search
            .best_params_
            .items()

        ):

            print(
                f"   {key}: {value}"
            )

        cv_results = pd.DataFrame(

            grid_search
            .cv_results_

        )

        safe_model_name = (

            model_name
            .replace(
                " ",
                "_"
            )

        )

        safe_prep_name = (

            prep_name
            .replace(
                "+",
                "_plus_"
            )

        )

        detail_file = (

            CV_DIR

            /

            (
                f"{safe_model_name}"
                f"__{safe_prep_name}.csv"
            )

        )

        cv_results.to_csv(

            detail_file,

            index=False

        )

        clean_parameters = {}

        for key, value in (

            grid_search
            .best_params_
            .items()

        ):

            clean_key = key.replace(
                "model__",
                ""
            )

            if isinstance(
                value,
                np.generic
            ):

                value = value.item()

            clean_parameters[
                clean_key
            ] = value

        row = {

            "model":
                model_name,

            "preprocessing":
                prep_name,

            "best_cv_r2":
                mean_r2,

            "cv_sd":
                sd_r2,

            "cv_se":
                se_r2,

            "runtime_minutes":
                elapsed_minutes,

            "best_parameters":
                json.dumps(
                    clean_parameters
                ),

        }

        summary_rows.append(
            row
        )

        model_results.append(
            row
        )

    model_results = sorted(

        model_results,

        key=lambda row:
            row[
                "best_cv_r2"
            ],

        reverse=True,

    )

    model_best = model_results[
        0
    ]

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"BEST {model_name} CONFIGURATION"
    )

    print(
        "-" * 100
    )

    print(
        "Preprocessing:",
        model_best[
            "preprocessing"
        ]
    )

    print(
        "CV RÂ²:",
        f"{model_best['best_cv_r2']:.6f}"
    )

    print(
        "Parameters:",
        model_best[
            "best_parameters"
        ]
    )

    best_configs[
        model_name
    ] = {

        "preprocessing":
            model_best[
                "preprocessing"
            ],

        "cv_r2":
            model_best[
                "best_cv_r2"
            ],

        "cv_sd":
            model_best[
                "cv_sd"
            ],

        "cv_se":
            model_best[
                "cv_se"
            ],

        "parameters":
            json.loads(
                model_best[
                    "best_parameters"
                ]
            ),

    }

summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(

    [
        "model",
        "best_cv_r2"
    ],

    ascending=[
        True,
        False
    ],

)

summary.to_csv(

    SUMMARY_OUTPUT,

    index=False

)

best_model_rows = (

    summary

    .sort_values(
        "best_cv_r2",
        ascending=False
    )

    .groupby(
        "model",
        as_index=False
    )

    .first()

    .sort_values(
        "best_cv_r2",
        ascending=False
    )

)

print(
    "\n"
    + "=" * 100
)

print(
    "BEST TRAINING-CV CONFIGURATION PER MODEL"
)

print(
    "=" * 100
)

print(

    best_model_rows[
        [
            "model",
            "preprocessing",
            "best_cv_r2",
            "cv_sd",
        ]
    ]

    .to_string(
        index=False
    )

)

output_metadata = {

    "study":
        "Delaware Santos-style replication",

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

    "selection_metric":
        "R2",

    "cross_validation":
        "5-fold shuffled KFold",

    "random_state":
        RANDOM_STATE,

    "n_features":
        len(
            FEATURES
        ),

    "features":
        FEATURES,

    "preprocessing_options":
        PREPROCESSING_OPTIONS,

    "best_configs":
        best_configs,

}

with open(

    CONFIG_OUTPUT,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        output_metadata,

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
    CONFIG_OUTPUT
)

print(
    CV_DIR
)

print(
    "\nGridSearch complete."
)

print(
    "The 281-row held-out TEST SET has NOT been evaluated."
)

print(
    "Next: run GMDH separately, then freeze all Santos models before opening the test set."
)


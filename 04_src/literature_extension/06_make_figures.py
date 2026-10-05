from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# Project paths

ROOT = Path(__file__).resolve().parents[2]

TABLES_DIR = ROOT / "05_results" / "tables"
METRICS_DIR = ROOT / "05_results" / "metrics"
DATA_DIR = ROOT / "01_data" / "processed"
OUT_DIR = ROOT / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)


INDEX_FILE = (
    TABLES_DIR
    / "31_delaware_santos_index_path.csv"
)

BAND_FILE = (
    TABLES_DIR
    / "31_delaware_santos_band_path.csv"
)

FEATURE_METADATA_FILE = (
    METRICS_DIR
    / "31_delaware_santos_features.json"
)

FEATURE_DATA_FILE = (
    DATA_DIR
    / "delaware_santos_features.csv"
)

PREDICTION_FILE = (
    TABLES_DIR
    / "35_delaware_final_test_predictions.csv"
)

METRIC_FILE = (
    TABLES_DIR
    / "35_delaware_final_test_metrics.csv"
)


required_files = [
    INDEX_FILE,
    BAND_FILE,
    FEATURE_METADATA_FILE,
    FEATURE_DATA_FILE,
    PREDICTION_FILE,
    METRIC_FILE,
]


for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


print("Using final analysis files:")

for file in required_files:

    print(" ", file)


# Plot settings

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


# Helper functions

def r2_score_manual(observed, predicted):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    ss_res = np.sum(
        (observed - predicted) ** 2
    )

    ss_tot = np.sum(
        (observed - observed.mean()) ** 2
    )

    return 1 - ss_res / ss_tot


def rmse(observed, predicted):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.sqrt(
        np.mean(
            (observed - predicted) ** 2
        )
    )


def mae(observed, predicted):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.mean(
        np.abs(
            observed - predicted
        )
    )


def mbe(observed, predicted):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.mean(
        predicted - observed
    )


# Load data

index_path = pd.read_csv(
    INDEX_FILE
)

band_path = pd.read_csv(
    BAND_FILE
)

feature_data = pd.read_csv(
    FEATURE_DATA_FILE,
    low_memory=False
)

predictions = pd.read_csv(
    PREDICTION_FILE
)

metrics = pd.read_csv(
    METRIC_FILE
)


with open(
    FEATURE_METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    feature_metadata = json.load(
        file
    )


# Validate expected columns

selection_columns = [
    "step",
    "selected_this_step",
    "combination_mean_cv_r2",
    "combination_se_cv_r2",
]


for file_name, data in [
    ("Index path", index_path),
    ("Band path", band_path),
]:

    missing = [
        col
        for col in selection_columns
        if col not in data.columns
    ]

    if missing:

        raise ValueError(
            f"{file_name} missing columns: {missing}"
        )


prediction_columns = [
    "observed_turbidity_FNU",
    "pred_kNN",
    "pred_GPR",
]


missing = [
    col
    for col in prediction_columns
    if col not in predictions.columns
]


if missing:

    raise ValueError(
        f"Prediction file missing columns: {missing}"
    )


# Figure 1
# Spectral-index forward selection

index_plot = index_path.copy()

index_plot["label"] = [
    f"IC{i}"
    for i in range(
        1,
        len(index_plot) + 1
    )
]


best_index_position = (
    index_plot[
        "combination_mean_cv_r2"
    ]
    .idxmax()
)


fig, ax = plt.subplots(
    figsize=(10, 5.5)
)


x = np.arange(
    len(index_plot)
)


bars = ax.bar(
    x,
    index_plot[
        "combination_mean_cv_r2"
    ],
    yerr=index_plot[
        "combination_se_cv_r2"
    ],
    capsize=3,
    edgecolor="black",
    linewidth=0.8,
)


bars[
    index_plot.index.get_loc(
        best_index_position
    )
].set_hatch("//")


ax.set_xticks(x)

ax.set_xticklabels(
    index_plot["label"],
    rotation=45,
    ha="right",
)


ax.set_xlabel(
    "Spectral-index combination"
)

ax.set_ylabel(
    r"Mean 5-fold CV $R^2$"
)


ax.set_title(
    "Forward selection of spectral indices"
)


best_value = index_plot.loc[
    best_index_position,
    "combination_mean_cv_r2"
]


best_feature = index_plot.loc[
    best_index_position,
    "selected_this_step"
]


ax.annotate(
    (
        f"Selected combination\n"
        f"$R^2$ = {best_value:.3f}"
    ),
    xy=(
        index_plot.index.get_loc(
            best_index_position
        ),
        best_value
    ),
    xytext=(20, 25),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->"
    },
)


ax.text(
    0.02,
    0.97,
    (
        "Forward order:\n"
        + " → ".join(
            index_plot[
                "selected_this_step"
            ].astype(str)
        )
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=8,
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


fig.savefig(
    OUT_DIR
    / "figure_1_index_selection.png",
    bbox_inches="tight"
)

plt.close(fig)


# Figure 2
# Spectral-band forward selection

band_plot = band_path.copy()

band_plot["label"] = [
    f"BC{i}"
    for i in range(
        1,
        len(band_plot) + 1
    )
]


best_band_position = (
    band_plot[
        "combination_mean_cv_r2"
    ]
    .idxmax()
)


fig, ax = plt.subplots(
    figsize=(9, 5.2)
)


x = np.arange(
    len(band_plot)
)


bars = ax.bar(
    x,
    band_plot[
        "combination_mean_cv_r2"
    ],
    yerr=band_plot[
        "combination_se_cv_r2"
    ],
    capsize=3,
    edgecolor="black",
    linewidth=0.8,
)


bars[
    band_plot.index.get_loc(
        best_band_position
    )
].set_hatch("//")


ax.set_xticks(x)

ax.set_xticklabels(
    band_plot["label"],
    rotation=40,
    ha="right",
)


ax.set_xlabel(
    "Indices + spectral-band combination"
)

ax.set_ylabel(
    r"Mean 5-fold CV $R^2$"
)


ax.set_title(
    "Forward selection of Sentinel-2 bands"
)


best_value = band_plot.loc[
    best_band_position,
    "combination_mean_cv_r2"
]


ax.annotate(
    (
        "Selected combination\n"
        f"$R^2$ = {best_value:.3f}"
    ),
    xy=(
        band_plot.index.get_loc(
            best_band_position
        ),
        best_value
    ),
    xytext=(20, 25),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->"
    },
)


ax.text(
    0.02,
    0.97,
    (
        "Band order:\n"
        + " → ".join(
            band_plot[
                "selected_this_step"
            ]
            .astype(str)
            .str.replace(
                "_rhow",
                "",
                regex=False
            )
        )
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=8,
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


fig.savefig(
    OUT_DIR
    / "figure_2_band_selection.png",
    bbox_inches="tight"
)

plt.close(fig)


# Figure 3
# Correlation between selected predictors and turbidity

original_features = (
    feature_metadata[
        "original_features"
    ]
)


target = "MeasurementValue"


missing = [
    feature
    for feature in original_features
    if feature not in feature_data.columns
]


if missing:

    raise ValueError(
        "Missing selected predictors in feature dataset: "
        f"{missing}"
    )


if target not in feature_data.columns:

    raise ValueError(
        f"{target} missing from feature dataset."
    )


correlations = []


for feature in original_features:

    temp = feature_data[
        [
            feature,
            target
        ]
    ].copy()


    temp[feature] = pd.to_numeric(
        temp[feature],
        errors="coerce"
    )

    temp[target] = pd.to_numeric(
        temp[target],
        errors="coerce"
    )

    temp = temp.dropna()


    correlation = (
        temp[feature]
        .corr(
            temp[target],
            method="pearson"
        )
    )


    correlations.append(
        correlation
    )


correlation_table = pd.DataFrame({
    "feature": original_features,
    "correlation": correlations
})


correlation_table = (
    correlation_table
    .sort_values(
        "correlation",
        ascending=False
    )
    .reset_index(drop=True)
)


matrix = (
    correlation_table[
        "correlation"
    ]
    .to_numpy()
    .reshape(-1, 1)
)


fig, ax = plt.subplots(
    figsize=(4.6, 6.5)
)


image = ax.imshow(
    matrix,
    aspect="auto",
    vmin=-1,
    vmax=1
)


ax.set_yticks(
    np.arange(
        len(
            correlation_table
        )
    )
)


clean_names = (
    correlation_table[
        "feature"
    ]
    .str.replace(
        "_rhow",
        "",
        regex=False
    )
)


ax.set_yticklabels(
    clean_names
)


ax.set_xticks([0])

ax.set_xticklabels(
    ["Turbidity (FNU)"]
)


for i, value in enumerate(
    correlation_table[
        "correlation"
    ]
):

    ax.text(
        0,
        i,
        f"{value:.2f}",
        ha="center",
        va="center",
        fontsize=9,
    )


colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.06,
    pad=0.05
)

colorbar.set_label(
    "Pearson correlation"
)


ax.set_title(
    "Correlation of selected predictors with turbidity"
)


fig.tight_layout()


fig.savefig(
    OUT_DIR
    / "figure_3_feature_correlation.png",
    bbox_inches="tight"
)

plt.close(fig)


# Figure 4
# Observed versus predicted turbidity
# kNN and GPR with marginal distributions

metric_lookup = (
    metrics
    .set_index("model")
)


model_information = {
    "kNN": {
        "column": "pred_kNN"
    },
    "GPR": {
        "column": "pred_GPR"
    },
}


def joint_panel(
    figure,
    subplot_spec,
    model_name,
    prediction_column
):

    observed_raw = pd.to_numeric(
        predictions[
            "observed_turbidity_FNU"
        ],
        errors="coerce"
    ).to_numpy()


    predicted_raw = pd.to_numeric(
        predictions[
            prediction_column
        ],
        errors="coerce"
    ).to_numpy()


    valid = (
        np.isfinite(observed_raw)
        &
        np.isfinite(predicted_raw)
    )


    observed_raw = observed_raw[
        valid
    ]

    predicted_raw = predicted_raw[
        valid
    ]


    raw_r2 = r2_score_manual(
        observed_raw,
        predicted_raw
    )

    raw_rmse = rmse(
        observed_raw,
        predicted_raw
    )

    raw_mae = mae(
        observed_raw,
        predicted_raw
    )

    raw_mbe = mbe(
        observed_raw,
        predicted_raw
    )


    if model_name in metric_lookup.index:

        official_r2 = float(
            metric_lookup.loc[
                model_name,
                "R2"
            ]
        )

        official_rmse = float(
            metric_lookup.loc[
                model_name,
                "RMSE_FNU"
            ]
        )

        official_mae = float(
            metric_lookup.loc[
                model_name,
                "MAE_FNU"
            ]
        )

        official_mbe = float(
            metric_lookup.loc[
                model_name,
                "MBE_FNU"
            ]
        )


        if not np.isclose(
            raw_r2,
            official_r2,
            atol=1e-5
        ):

            raise RuntimeError(
                (
                    f"{model_name} R² from predictions "
                    f"({raw_r2:.6f}) does not match "
                    f"final metrics ({official_r2:.6f})."
                )
            )


        if not np.isclose(
            raw_rmse,
            official_rmse,
            atol=1e-5
        ):

            raise RuntimeError(
                (
                    f"{model_name} RMSE from predictions "
                    f"({raw_rmse:.6f}) does not match "
                    f"final metrics ({official_rmse:.6f})."
                )
            )


    positive = (
        (observed_raw > 0)
        &
        (predicted_raw > 0)
    )


    observed = observed_raw[
        positive
    ]

    predicted = predicted_raw[
        positive
    ]


    excluded = (
        len(observed_raw)
        -
        len(observed)
    )


    print(
        f"{model_name}: "
        f"{len(observed)} positive pairs plotted; "
        f"{excluded} excluded from log-scale display."
    )


    internal = GridSpecFromSubplotSpec(
        4,
        4,
        subplot_spec=subplot_spec,
        wspace=0.06,
        hspace=0.06
    )


    ax_top = figure.add_subplot(
        internal[
            0,
            0:3
        ]
    )


    ax_main = figure.add_subplot(
        internal[
            1:4,
            0:3
        ]
    )


    ax_right = figure.add_subplot(
        internal[
            1:4,
            3
        ]
    )


    ax_main.scatter(
        observed,
        predicted,
        s=20,
        alpha=0.65,
        edgecolors="none",
    )


    minimum = min(
        observed.min(),
        predicted.min()
    )


    maximum = max(
        observed.max(),
        predicted.max()
    )


    ax_main.plot(
        [
            minimum,
            maximum
        ],
        [
            minimum,
            maximum
        ],
        linestyle="--",
        linewidth=1.0,
        label="1:1 line",
    )


    log_observed = np.log10(
        observed
    )

    log_predicted = np.log10(
        predicted
    )


    slope, intercept = np.polyfit(
        log_observed,
        log_predicted,
        1
    )


    regression_x = np.logspace(
        np.log10(minimum),
        np.log10(maximum),
        200
    )


    regression_y = (
        10
        **
        (
            intercept
            +
            slope
            *
            np.log10(
                regression_x
            )
        )
    )


    ax_main.plot(
        regression_x,
        regression_y,
        linewidth=1.1,
        label="Regression",
    )


    ax_main.set_xscale(
        "log"
    )

    ax_main.set_yscale(
        "log"
    )


    ax_main.set_xlabel(
        "Observed turbidity (FNU)"
    )

    ax_main.set_ylabel(
        "Predicted turbidity (FNU)"
    )


    metric_text = (
        f"$R^2$ = {official_r2:.3f}\n"
        f"RMSE = {official_rmse:.2f} FNU\n"
        f"MAE = {official_mae:.2f} FNU\n"
        f"MBE = {official_mbe:.2f} FNU"
    )


    ax_main.text(
        0.04,
        0.96,
        metric_text,
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "boxstyle": "round,pad=0.25"
        },
    )


    ax_main.legend(
        loc="lower right",
        frameon=False
    )


    log_bins = np.logspace(
        np.log10(minimum),
        np.log10(maximum),
        20
    )


    ax_top.hist(
        observed,
        bins=log_bins,
        edgecolor="black",
        linewidth=0.5,
    )


    ax_top.set_xscale(
        "log"
    )


    ax_top.tick_params(
        axis="x",
        labelbottom=False
    )


    ax_top.set_ylabel(
        "Count"
    )


    ax_right.hist(
        predicted,
        bins=log_bins,
        orientation="horizontal",
        edgecolor="black",
        linewidth=0.5,
    )


    ax_right.set_yscale(
        "log"
    )


    ax_right.tick_params(
        axis="y",
        labelleft=False
    )


    ax_right.set_xlabel(
        "Count"
    )


    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)


    ax_top.set_title(
        f"({model_name})",
        fontsize=11
    )


fig = plt.figure(
    figsize=(11.5, 5.8)
)


outer = GridSpec(
    1,
    2,
    figure=fig,
    wspace=0.28
)


joint_panel(
    fig,
    outer[0],
    "kNN",
    model_information[
        "kNN"
    ]["column"]
)


joint_panel(
    fig,
    outer[1],
    "GPR",
    model_information[
        "GPR"
    ]["column"]
)


fig.suptitle(
    (
        "Observed and predicted turbidity "
        "for the two highest-performing models"
    ),
    y=1.01
)


fig.savefig(
    OUT_DIR
    / "figure_4_observed_vs_predicted.png",
    bbox_inches="tight"
)


plt.close(fig)


# Final report

print("\nFigures created:")

for file in sorted(
    OUT_DIR.glob(
        "figure_*.png"
    )
):

    print(
        " ",
        file.relative_to(
            ROOT
        )
    )


print(
    "\nFigure generation complete."
)
import json
import os

import altair as alt
import numpy as np
import pandas as pd

from ts2.utils.tailwind import TC


def get_slide_key(row: pd.Series) -> str:
    return f"{row['patient']}-{row['mosaic']}"


def get_slide_statistics_path(pred_root: str, row: pd.Series) -> str:
    slide_key = get_slide_key(row)
    return os.path.join(pred_root, slide_key, f"{slide_key}-slide_statistics.json")


def get_prediction_display_key(prediction_key: str) -> str:
    return prediction_key.removesuffix("_slide_tumor_probability")


def read_slide_predictions(
    pred_root: str, row: pd.Series, prediction_keys: list
) -> dict:
    slide_statistics_path = get_slide_statistics_path(pred_root, row)
    if not os.path.exists(slide_statistics_path):
        print(
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"WARNING: Missing slide statistics JSON, dropping row: {slide_statistics_path}\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
        return {get_prediction_display_key(key): pd.NA for key in prediction_keys}

    with open(slide_statistics_path) as f:
        slide_statistics = json.load(f)

    missing_keys = set(prediction_keys) - set(slide_statistics)
    if missing_keys:
        raise KeyError(
            f"Missing keys {sorted(missing_keys)} in {slide_statistics_path}"
        )

    return {
        get_prediction_display_key(key): slide_statistics[key]
        for key in prediction_keys
    }


def compute_label_correlations(
    data: pd.DataFrame,
    prediction_keys: list,
    cohort_name: str,
) -> pd.DataFrame:
    required_columns = {"label", *prediction_keys}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing columns for correlation: {sorted(missing_columns)}")

    return pd.DataFrame(
        [
            {
                "cohort": cohort_name,
                "prediction_key": prediction_key,
                "n": len(data.dropna(subset=[prediction_key, "label"])),
                "pearson_correlation": data[prediction_key].corr(data["label"]),
                "spearman_correlation": data[prediction_key].corr(data["label"], method="spearman"),
            }
            for prediction_key in prediction_keys
        ]
    )


def build_fastglioma_scatter_plot(
    data: pd.DataFrame,
    prediction_key: str,
    fg_key: str,
    label_key: str,
    color_range: list,
) -> alt.Chart:
    alt.data_transformers.disable_max_rows()

    plot_data = data[[fg_key, label_key, prediction_key, "patient", "mosaic"]].dropna(
        subset=[fg_key, label_key, prediction_key]
    )
    unit_axis = alt.Axis(
        tickSize=0,
        values=np.linspace(0, 1, 6),
        domain=False,
        labels=False,
        title="",
    )
    fg_axis = alt.Axis(tickSize=0, values=np.linspace(0, 1, 6), title="FastGlioma")
    label_axis = alt.Axis(tickWidth=0, title="Label")
    label_color = alt.Color(
        f"{label_key}:N",
        title=label_key,
        scale=alt.Scale(range=color_range),
    )

    base_chart = alt.Chart(plot_data)
    scatter = (
        base_chart.mark_point(filled=True)
        .encode(
            x=alt.X(f"{fg_key}:Q", axis=unit_axis),
            y=alt.Y(f"{prediction_key}:Q", axis=unit_axis),
            color=label_color,
            tooltip=["patient", "mosaic", label_key, fg_key, prediction_key],
        )
        .properties(width=320, height=320, title=prediction_key)
    )

    y_ticks = (
        base_chart.mark_tick()
        .encode(
            x=alt.X(f"{label_key}:N", axis=label_axis),
            y=alt.Y(
                f"{prediction_key}:Q",
                axis=alt.Axis(tickSize=0, values=np.linspace(0, 1, 6)),
            ),
            color=label_color,
            tooltip=["patient", "mosaic", label_key, prediction_key],
        )
        .properties(width=70, height=320)
    )

    x_ticks = (
        base_chart.mark_tick()
        .encode(
            x=alt.X(f"{fg_key}:Q", axis=fg_axis),
            y=alt.Y(f"{label_key}:N", axis=label_axis),
            color=label_color,
            tooltip=["patient", "mosaic", label_key, fg_key],
        )
        .properties(width=320, height=70)
    )

    return (
        (y_ticks | (scatter & x_ticks))
        .configure_axis(
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(titleFontSize=12)
    )


def load_ucsf_fastglioma_scores(
    sample_slide_map_csv: str,
    fg_score_ucsf_csv: str,
) -> pd.DataFrame:
    sample_slide_map = pd.read_csv(sample_slide_map_csv, dtype=str)
    sample_slide_map = pd.DataFrame(
        {
            "slide_id": sample_slide_map["slide_id"],
            "pbs": (
                sample_slide_map["patient_id"]
                + "-"
                + sample_slide_map["barcode"]
                + "-"
                + sample_slide_map["sample_id"]
            ),
        }
    )

    fg_score = pd.read_csv(fg_score_ucsf_csv, dtype=str).dropna()
    fg_score["pbs"] = (
        "NIO_UCSF_"
        + fg_score["patient_num"].astype(str)
        + "-"
        + fg_score["barcode"]
        + "-"
        + fg_score["sample_id"]
    )

    fg_score = fg_score.merge(sample_slide_map, on="pbs", how="left")
    fg_score = fg_score.rename({"slide_id": "mosaic"}, axis=1)
    fg_score = fg_score.drop(["pbs", "sample_id", "barcode"], axis=1).dropna()
    fg_score["patient"] = "NIO_UCSF_" + fg_score["patient_num"].astype(str)
    fg_score["mosaic"] = fg_score["mosaic"].astype(str)

    return fg_score[["patient", "mosaic", "fullsrh"]]


def add_fastglioma_scores(
    data: pd.DataFrame,
    fg_score: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()
    data["mosaic"] = data["mosaic"].astype(str)
    return data.merge(fg_score, on=["patient", "mosaic"], how="left")


def warn_missing_fastglioma_scores(data: pd.DataFrame) -> None:
    missing_fastglioma = data["fullsrh"].isna()
    if not missing_fastglioma.any():
        return

    missing_rows = data.loc[missing_fastglioma, ["patient", "mosaic"]]
    print(
        "\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        f"WARNING: Missing FastGlioma fullsrh for {len(missing_rows)} rows:\n"
        f"{missing_rows.to_string(index=False)}\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    )


def main():
    prospective_csv = "data/ucsf_retrospective_val.csv"  #"data/ucsf_all_sorted.csv"  # 
    output_prefix = os.path.splitext(os.path.basename(prospective_csv))[0]
    pred_root = "/scratch/tocho_root/tocho0/chengjia/silica_ucsf/b1a0cbe3_k1024"
    sample_slide_map_csv = (
        "../../../playgrounds/data/db_srhdg/SRHcases_ForMelike-MP-2025-03-15.csv"
    )
    fg_score_ucsf_csv = "../../../playgrounds/data/db_srhdg/scoresforsanjeev_ucsf.csv"
    prediction_keys = [
        "area_soft_slide_tumor_probability",
        "hard_slide_tumor_probability",
        "soft_slide_tumor_probability",
    ]
    prediction_display_keys = [
        get_prediction_display_key(prediction_key) for prediction_key in prediction_keys
    ]
    color_range = TC()(c="LSAR", s=5)

    data = pd.read_csv(prospective_csv)
    fg_score = load_ucsf_fastglioma_scores(sample_slide_map_csv, fg_score_ucsf_csv)
    data = add_fastglioma_scores(data, fg_score)
    warn_missing_fastglioma_scores(data)
    prediction_rows = data.apply(
        lambda row: read_slide_predictions(pred_root, row, prediction_keys),
        axis=1,
        result_type="expand",
    )
    fullsrh = data.pop("fullsrh")
    data = pd.concat((data, prediction_rows), axis=1)
    data["fullsrh"] = fullsrh
    data_with_predictions = data.dropna(subset=prediction_display_keys)
    data_with_fastglioma = data_with_predictions.dropna(subset=["fullsrh"])

    #import pdb; pdb.set_trace()
    correlations = pd.concat(
        (
            compute_label_correlations(
                data_with_predictions,
                prediction_display_keys,
                cohort_name="all_with_ours",
            ),
            compute_label_correlations(
                data_with_fastglioma,
                prediction_display_keys + ["fullsrh"],
                cohort_name="with_fastglioma",
            ),
        ),
        ignore_index=True,
    )
    print(correlations)
    for prediction_key in prediction_display_keys:
        fastglioma_scatter_plot = build_fastglioma_scatter_plot(
            data_with_fastglioma,
            prediction_key=prediction_key,
            fg_key="fullsrh",
            label_key="label",
            color_range=color_range,
        )
        fastglioma_scatter_plot.save(
            f"{output_prefix}_fastglioma_vs_{prediction_key}.html"
        )
        fastglioma_scatter_plot.save(
            f"{output_prefix}_fastglioma_vs_{prediction_key}.pdf"
        )
        fastglioma_scatter_plot.save(
            f"{output_prefix}_fastglioma_vs_{prediction_key}.png"
        )


if __name__ == "__main__":
    main()

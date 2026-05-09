import json
import os

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ts2.utils.tailwind import TC


def get_slide_key(row: pd.Series) -> str:
    return f"{row['patient']}-{row['mosaic']}"


def get_slide_statistics_path(pred_root: str, row: pd.Series) -> str:
    slide_key = get_slide_key(row)
    return os.path.join(pred_root, slide_key, f"{slide_key}-slide_statistics.json")


def get_prediction_display_key(prediction_key: str) -> str:
    return prediction_key.removesuffix("_slide_tumor_probability")


def _json_float(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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


def score_to_pred_label(score: pd.Series, num_classes: int) -> pd.Series:
    numeric_score = pd.to_numeric(score)
    pred_label = np.floor(numeric_score.to_numpy(dtype=np.float64) * num_classes)
    pred_label = np.clip(pred_label, 0, num_classes - 1)
    return pd.Series(pred_label.astype(np.int64), index=score.index)


def confusion_matrix_from_rows(rows: pd.DataFrame, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for row in rows.itertuples(index=False):
        matrix[int(row.label), int(row.pred_label)] += 1
    return matrix.tolist()


def mean_class_accuracy(rows: pd.DataFrame, num_classes: int) -> float | None:
    class_accuracies = []
    for class_idx in range(num_classes):
        in_class = rows["label"] == class_idx
        if in_class.any():
            class_accuracies.append(
                (rows.loc[in_class, "pred_label"] == rows.loc[in_class, "label"]).mean()
            )
    if not class_accuracies:
        return None
    return float(np.mean(class_accuracies))


def threshold_auroc_metrics(rows: pd.DataFrame, num_classes: int) -> dict:
    metrics = {}
    labels = rows["label"].astype(int)
    raw_score = rows["raw_score"].astype(float)
    for threshold in range(num_classes - 1):
        target = (labels > threshold).astype(int)
        negative_name = "".join(str(idx) for idx in range(threshold + 1))
        positive_name = "".join(str(idx) for idx in range(threshold + 1, num_classes))
        metric_key = f"auroc_{negative_name}_vs_{positive_name}"
        if target.nunique() < 2:
            metrics[metric_key] = None
        else:
            metrics[metric_key] = _json_float(roc_auc_score(target, raw_score))
    return metrics


def compute_prediction_page_metrics(rows: pd.DataFrame, num_classes: int) -> dict:
    if rows.empty:
        raise ValueError("No rows with predictions were available for website metrics")
    label = rows["label"].astype(int)
    pred_label = rows["pred_label"].astype(int)
    raw_score = rows["raw_score"].astype(float)

    metrics = {
        "num_datapoints": int(len(rows)),
        "confusion_matrix": confusion_matrix_from_rows(rows, num_classes),
        "pred_label_accuracy": _json_float((pred_label == label).mean()),
        "pred_label_mean_class_accuracy": _json_float(
            mean_class_accuracy(rows, num_classes)
        ),
        "pred_label_mae": _json_float((pred_label - label).abs().mean()),
        "pred_label_mse": _json_float(((pred_label - label) ** 2).mean()),
        "raw_score_pearson": _json_float(raw_score.corr(label, method="pearson")),
        "raw_score_spearman": _json_float(raw_score.corr(label, method="spearman")),
    }
    metrics.update(threshold_auroc_metrics(rows, num_classes))
    return metrics


def build_prediction_page_rows(
    data: pd.DataFrame,
    score_key: str,
    score_label: str,
    num_classes: int,
) -> pd.DataFrame:
    required_columns = {"patient", "mosaic", "label", score_key}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing columns for website rows: {sorted(missing_columns)}")

    rows = data.loc[:, ["patient", "mosaic", "label", score_key]].copy()
    rows["raw_score"] = pd.to_numeric(rows[score_key], errors="coerce")
    rows = rows.dropna(subset=["label", "raw_score"]).copy()
    rows["label"] = rows["label"].astype(int)
    rows["slide_id"] = rows.apply(get_slide_key, axis=1)
    rows["path"] = rows["slide_id"]
    rows["pred_label"] = score_to_pred_label(rows["raw_score"], num_classes)
    rows["score_label"] = score_label
    return rows


def build_prediction_page_chart(
    rows: pd.DataFrame, experiment_name: str, num_classes: int
) -> alt.Chart:
    alt.data_transformers.disable_max_rows()
    plot_rows = rows.copy()
    plot_rows["label"] = pd.to_numeric(plot_rows["label"])
    plot_rows["pred_label"] = pd.to_numeric(plot_rows["pred_label"])

    jitter = (
        pd.util.hash_pandas_object(plot_rows["slide_id"], index=False) % 1000
    ) / 999.0
    plot_rows["label_jitter"] = plot_rows["label"] + (jitter - 0.5) * 0.18

    x_axis = alt.X(
        "label:O",
        title="Ground truth",
        sort=list(range(num_classes)),
        axis=alt.Axis(ticks=False),
    )
    y_axis = alt.Y("raw_score:Q", title="Raw score", axis=alt.Axis(ticks=False))
    tooltip = [
        alt.Tooltip("slide_id:N", title="Slide"),
        alt.Tooltip("patient:N", title="Patient"),
        alt.Tooltip("mosaic:N", title="Mosaic"),
        alt.Tooltip("label:N", title="Ground truth"),
        alt.Tooltip("pred_label:N", title="Predicted"),
        alt.Tooltip("raw_score:Q", title="Raw score", format=".4f"),
        alt.Tooltip("score_label:N", title="Score"),
    ]
    box = (
        alt.Chart(plot_rows)
        .mark_boxplot(
            extent="min-max",
            size=48,
            color="#1F1F1F",
            box={"fillOpacity": 0, "stroke": "#1F1F1F"},
            median={"color": "#1F1F1F"},
            rule={"stroke": "#1F1F1F"},
            ticks=False,
        )
        .encode(x=x_axis, y=y_axis)
    )
    points = (
        alt.Chart(plot_rows)
        .mark_circle(size=42, opacity=0.7, color="#1F1F1F")
        .encode(
            x=alt.X(
                "label_jitter:Q",
                title="Ground truth",
                scale=alt.Scale(domain=[-0.5, num_classes - 0.5]),
                axis=alt.Axis(values=list(range(num_classes)), ticks=False),
            ),
            y=y_axis,
            tooltip=tooltip,
        )
    )
    return (box + points).properties(width=520, height=420, title=experiment_name)


def save_prediction_page_outputs(
    *,
    data: pd.DataFrame,
    experiment_name: str,
    score_key: str,
    score_label: str,
    output_dir: str,
    num_classes: int,
) -> None:
    rows = build_prediction_page_rows(
        data=data,
        score_key=score_key,
        score_label=score_label,
        num_classes=num_classes,
    )
    os.makedirs(output_dir, exist_ok=True)

    chart_path = os.path.join(output_dir, f"{experiment_name}_fg_score_charts.json")
    chart = build_prediction_page_chart(
        rows,
        experiment_name=experiment_name,
        num_classes=num_classes,
    )
    with open(chart_path, "w", encoding="utf-8") as f:
        json.dump(chart.to_dict(), f)

    metrics_path = os.path.join(output_dir, f"{experiment_name}_metrics.json")
    metrics = compute_prediction_page_metrics(rows, num_classes=num_classes)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, sort_keys=True, indent=4)


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
                "spearman_correlation": data[prediction_key].corr(
                    data["label"], method="spearman"
                ),
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
    data = data.merge(fg_score, on=["patient", "mosaic"], how="left")
    data["fullsrh"] = pd.to_numeric(data["fullsrh"], errors="coerce")
    return data


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
    num_classes = 4
    prospective_csv = "data/ucsf_retrospective_val.csv"  # "data/ucsf_all_sorted.csv"  #
    pred_root = "/scratch/tocho_root/tocho0/chengjia/silica_ucsf/gmm/b1a0cbe3_k16"

    sample_slide_map_csv = (
        "../../../playgrounds/data/db_srhdg/SRHcases_ForMelike-MP-2025-03-15.csv"
    )
    fg_score_ucsf_csv = "../../../playgrounds/data/db_srhdg/scoresforsanjeev_ucsf.csv"

    prediction_keys = [
        "area_soft_slide_tumor_probability",
        "hard_slide_tumor_probability",
        "soft_slide_tumor_probability",
    ]

    save_portal_output = True
    portal_output_dir = "../infil/site_res/prediction_metrics"
    silica_experiment_name = os.path.basename(os.path.normpath(pred_root))
    fastglioma_experiment_name = "fastglioma"

    os.makedirs(silica_experiment_name)
    output_prefix = os.path.join(silica_experiment_name, os.path.splitext(os.path.basename(prospective_csv))[0])

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

    if save_portal_output:
        save_prediction_page_outputs(
            data=data_with_predictions,
            experiment_name=silica_experiment_name,
            score_key="area_soft",
            score_label="Silica area soft slide tumor probability",
            output_dir=portal_output_dir,
            num_classes=num_classes,
        )
        save_prediction_page_outputs(
            data=data.dropna(subset=["fullsrh"]),
            experiment_name=fastglioma_experiment_name,
            score_key="fullsrh",
            score_label="FastGlioma fullsrh",
            output_dir=portal_output_dir,
            num_classes=num_classes,
        )

    # import pdb; pdb.set_trace()
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

import json
import logging
import os
from glob import glob
from os.path import join as opj

import altair as alt
import pandas as pd
from tqdm import tqdm

from ts2.utils.silica_scnn.eval_patch_proposal_neighborhood_embeddings import (
    build_runs_from_sets,
    extract_exp_key,
)


def infer_latest_summary_json_path(pred_path: str) -> str:
    pred_dir = os.path.dirname(pred_path)
    assert os.path.basename(pred_dir) == "predictions"
    eval_root = os.path.dirname(pred_dir)
    summary_json_paths = glob(opj(eval_root, "results", "run*", "summary.json"))
    assert summary_json_paths
    return max(summary_json_paths, key=os.path.getmtime)


def find_latest_scnn_summary_path(pred_path: str) -> str:
    return infer_latest_summary_json_path(pred_path)


def load_scnn_summary(summary_path: str) -> dict:
    assert os.path.isfile(summary_path), f"Missing SCNN summary.json: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as fd:
        summary = json.load(fd)
    assert isinstance(
        summary, dict
    ), f"Expected SCNN summary at {summary_path} to be a dict, got {type(summary)}"
    return summary


def load_saved_summaries(summary_json_paths: list[str]) -> pd.DataFrame:
    logging.info("Loading %d summary files", len(summary_json_paths))
    summaries = []
    for summary_json_path in tqdm(summary_json_paths, desc="Loading summaries"):
        summaries.append(load_scnn_summary(summary_json_path))

    assert summaries, "Expected at least one run summary to aggregate"
    return pd.DataFrame(summaries).sort_values(
        by=["name", "celldist_mode", "dgt", "dle", "nge"],
        kind="stable",
    )


def build_plot_df(
    summary_df: pd.DataFrame, exp_name_label_map: dict[str, str]
) -> pd.DataFrame:
    plot_df = summary_df[
        [
            "exp_name",
            "dgt",
            "dle",
            "nge",
            "mean_neighbor_embedding_cosine_distance",
        ]
    ].copy()
    plot_df["exp_key"] = plot_df["exp_name"].map(extract_exp_key)
    plot_df["exp"] = plot_df["exp_key"].map(exp_name_label_map)
    missing_labels = plot_df.loc[plot_df["exp"].isna(), "exp_name"].unique()
    assert len(missing_labels) == 0, "Missing exp_name legend labels for: " + ", ".join(
        sorted(missing_labels)
    )
    return plot_df.sort_values(
        by=["exp", "dgt", "dle", "nge"],
        kind="stable",
    ).reset_index(drop=True)


def build_interval_chart(plot_df: pd.DataFrame) -> alt.Chart:
    x_values = sorted(
        set(plot_df["dgt"].unique().tolist()) | set(plot_df["dle"].unique().tolist())
    )
    y_encoding = alt.Y(
        "mean_neighbor_embedding_cosine_distance:Q",
        title="mean_neighbor_embedding_cosine_distance",
        axis=alt.Axis(tickSize=0),
        scale=alt.Scale(zero=False),
    )
    tooltip = [
        "exp:N",
        "exp_name:N",
        "dgt:Q",
        "dle:Q",
        "nge:Q",
        "mean_neighbor_embedding_cosine_distance:Q",
    ]
    color_encoding = alt.Color("exp:N", title="Experiment")
    return (
        alt.Chart(plot_df)
        .encode(
            y=y_encoding,
            color=color_encoding,
            tooltip=tooltip,
        )
        .mark_rule(strokeWidth=3, opacity=0.6)
        .encode(
            x=alt.X(
                "dgt:Q",
                title="Distance bound",
                axis=alt.Axis(values=x_values, tickSize=0),
                scale=alt.Scale(zero=False),
            ),
            x2="dle:Q",
            detail=["exp:N", "dgt:Q", "dle:Q", "nge:Q"],
        )
        .properties(
            title="Mean Neighborhood Embedding Cosine Distance by [dgt, dle]",
            width=400,
            height=400,
        )
    )


def build_endpoint_chart(plot_df: pd.DataFrame) -> alt.LayerChart:
    x_values = sorted(
        set(plot_df["dgt"].unique().tolist()) | set(plot_df["dle"].unique().tolist())
    )
    x_encoding = alt.X(
        "dle:Q",
        title="Distance bound",
        axis=alt.Axis(values=x_values, tickSize=0),
    )
    y_encoding = alt.Y(
        "mean_neighbor_embedding_cosine_distance:Q",
        title="mean_neighbor_embedding_cosine_distance",
        axis=alt.Axis(tickSize=0),
        scale=alt.Scale(zero=False),
    )
    tooltip = [
        "exp:N",
        "exp_name:N",
        "dgt:Q",
        "dle:Q",
        "nge:Q",
        "mean_neighbor_embedding_cosine_distance:Q",
    ]
    color_encoding = alt.Color("exp:N", title="Experiment")
    base = alt.Chart(plot_df).encode(
        x=x_encoding,
        y=y_encoding,
        color=color_encoding,
        tooltip=tooltip,
    )
    return (
        base.mark_line(strokeWidth=3).encode(
            detail="exp:N",
            order=alt.Order("dle:Q"),
        )
        + base.mark_point(size=70, filled=True)
    ).properties(
        title="Mean Neighborhood Embedding Cosine Distance by dle",
        width=400,
        height=400,
    )


def save_chart_outputs(chart: alt.TopLevelMixin, out_path_prefix: str) -> None:
    logging.info("Saving chart to %s.[html|pdf|png]", out_path_prefix)
    chart.save(f"{out_path_prefix}.html")
    chart.save(f"{out_path_prefix}.pdf")
    chart.save(f"{out_path_prefix}.png")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/"
    ckpt = "training_124999"
    eval_key_prefix = "cellnbrring"  #"cellnbr"  # 
    default_pred_glob = f"*INF_srh7v1test_{eval_key_prefix}_*"

    neighborhood_map_csv_path_template = (
        "out/cellnbr_stats_nbr_8192_dgt{dist_min}_dle{dist_max}_nge1/"
        "sampled_neighborhood_map.csv"
    )

    exp_name_label_map = {
        "04e0bf39": "DINOv2 lr4e-3",
        "78d57cfc": "DINOv2 lr4e-3 RmBg",
        "a2706135": "DINOv2 Meta",
        "ca187b7c": "Silica FullIm iBOT lr4e-3",
        "844ffd45": "Silica Inside iBOT lr4e-3",
        "b1a0cbe3": "Silica FullIm iBOT lr1e-3",
        "4fb55301": "Silica FullIm iBOT lr5e-4",
    }
    run_sets = [
        {"exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0"},
        {"exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0"},
        {"exp_name": "a2706135_dinov2"},
        {"exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0"},
        {"exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1"},
        {"exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0"},
        {"exp_name": "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0"},
    ]
    runs = build_runs_from_sets(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=run_sets,
        neighborhood_map_csv_path_template=neighborhood_map_csv_path_template,
        default_pred_glob=default_pred_glob,
        eval_key_prefix=eval_key_prefix,
    )
    logging.info("Found %d runs to plot", len(runs))
    summary_json_paths = [
        infer_latest_summary_json_path(run["pred_path"]) for run in runs
    ]
    logging.info("Resolved %d summary paths", len(summary_json_paths))
    summary_df = load_saved_summaries(summary_json_paths)
    logging.info("Loaded summary dataframe with %d rows", len(summary_df))
    plot_df = build_plot_df(summary_df, exp_name_label_map)
    logging.info("Prepared plot dataframe with %d rows", len(plot_df))

    logging.info("Building interval chart")
    interval_chart = build_interval_chart(plot_df)
    logging.info("Building endpoint chart")
    endpoint_chart = build_endpoint_chart(plot_df)

    save_chart_outputs(interval_chart, f"nbr_cosdist_{eval_key_prefix}_interval")
    save_chart_outputs(endpoint_chart, f"nbr_cosdist_{eval_key_prefix}_line_dle")


if __name__ == "__main__":
    main()

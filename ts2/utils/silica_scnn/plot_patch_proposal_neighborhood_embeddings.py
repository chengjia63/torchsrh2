import json
import logging
import os
from os.path import join as opj
import re

import altair as alt
import pandas as pd
from tqdm import tqdm

from ts2.utils.silica_scnn.eval_patch_proposal_neighborhood_embeddings import (
    build_runs_from_sets,
)
from ts2.utils.silica_model_registry import (
    DISPLAY_NAME_BY_EXP,
    build_display_name_color_range,
)


def infer_latest_summary_json_path(pred_path: str) -> str:
    pred_dir = os.path.dirname(pred_path)
    if os.path.basename(pred_dir) != "predictions":
        raise ValueError(
            f"Expected prediction path under a predictions directory, got {pred_path}"
        )
    eval_root = os.path.dirname(pred_dir)
    results_root = opj(eval_root, "results")
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    summary_candidates = []
    for entry in os.listdir(results_root):
        entry_path = opj(results_root, entry)
        if not os.path.isdir(entry_path):
            continue
        match = re.fullmatch(r"run(\d{4})", entry)
        if match is None:
            continue
        summary_path = opj(entry_path, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        summary_candidates.append((int(match.group(1)), summary_path))

    if not summary_candidates:
        raise FileNotFoundError(
            "No completed SCNN result directories with summary.json found under "
            f"{results_root}"
        )

    return max(summary_candidates, key=lambda item: item[0])[1]


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
    summary_df: pd.DataFrame, display_name_by_exp: dict[str, str]
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
    plot_df["exp"] = plot_df["exp_name"].map(display_name_by_exp)
    missing_labels = plot_df.loc[plot_df["exp"].isna(), "exp_name"].unique()
    assert len(missing_labels) == 0, "Missing exp_name legend labels for: " + ", ".join(
        sorted(missing_labels)
    )
    return plot_df.sort_values(
        by=["exp", "dgt", "dle", "nge"],
        kind="stable",
    ).reset_index(drop=True)


def build_interval_chart(plot_df: pd.DataFrame, color_scale: alt.Scale) -> alt.Chart:
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
    color_encoding = alt.Color("exp:N", title="Experiment", scale=color_scale)
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


def build_endpoint_chart(
    plot_df: pd.DataFrame, color_scale: alt.Scale
) -> alt.LayerChart:
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
    color_encoding = alt.Color("exp:N", title="Experiment", scale=color_scale)
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
    eval_key_prefix = "cellnbr"  # "cellnbrring"  #
    default_pred_glob = f"*INF_srh7v1test_{eval_key_prefix}_*"

    neighborhood_map_csv_path_template = (
        "out/cellnbr_stats_nbr_8192_dgt{dist_min}_dle{dist_max}_nge1/"
        "sampled_neighborhood_map.csv"
    )

    run_sets = [
        {"exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0"},
        {"exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0"},
        {"exp_name": "a2706135_dinov2"},
        {"exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0"},
        {"exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1"},
        {"exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0"},
        {"exp_name": "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0"},
        {"exp_name": "326a6384_Apr10-15-07-23_sd1000_nomaskobw_lr14_tune0"},
        {"exp_name": "10d41c43_Apr11-02-05-16_sd1000_nomaskobw_lr23_tune0"},
        {"exp_name": "716f4772_Apr12-03-21-26_sd1000_maskobw_lr13_tune1"},
        {"exp_name": "28d7879f_Apr13-02-20-13_sd1000_maskobw_lr54_tune1"},
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
    plot_df = build_plot_df(summary_df, DISPLAY_NAME_BY_EXP)
    logging.info("Prepared plot dataframe with %d rows", len(plot_df))
    color_domain, color_range = build_display_name_color_range(
        [
            DISPLAY_NAME_BY_EXP[run_set["exp_name"]]
            for run_set in run_sets
            if run_set["exp_name"] in DISPLAY_NAME_BY_EXP
        ]
    )
    color_scale = alt.Scale(
        domain=color_domain,
        range=color_range,
    )

    logging.info("Building interval chart")
    interval_chart = build_interval_chart(plot_df, color_scale=color_scale)
    logging.info("Building endpoint chart")
    endpoint_chart = build_endpoint_chart(plot_df, color_scale=color_scale)

    save_chart_outputs(interval_chart, f"nbr_cosdist_{eval_key_prefix}_interval")
    save_chart_outputs(endpoint_chart, f"nbr_cosdist_{eval_key_prefix}_line_dle")


if __name__ == "__main__":
    main()

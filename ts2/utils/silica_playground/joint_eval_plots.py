import os
from os.path import join as opj
from typing import Dict, List

import altair as alt
import pandas as pd
from tqdm import tqdm

from ts2.utils.silica_sc_cls.plot_cell_inference_knn_perturbation import (
    build_metrics_df as build_cls_curve_metrics_df,
    save_chart,
)
from ts2.utils.silica_scnn.eval_patch_proposal_neighborhood_embeddings import (
    build_runs_from_sets as build_scnn_runs_from_sets,
    evaluate_run as evaluate_scnn_run,
)


def build_scnn_metrics_df(
    exp_root: str,
    ckpt: str,
    run_sets: List[Dict[str, str]],
    neighborhood_map_csv_path_template: str,
    eval_key_prefix: str,
    display_name_by_exp: Dict[str, str],
) -> pd.DataFrame:
    runs = build_scnn_runs_from_sets(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=run_sets,
        neighborhood_map_csv_path_template=neighborhood_map_csv_path_template,
        eval_key_prefix=eval_key_prefix,
    )

    rows: list[dict[str, object]] = []
    for run in tqdm(runs, desc="Evaluating SCNN neighborhood runs"):
        summary = evaluate_scnn_run(**run)
        exp_name = str(summary["exp_name"])
        rows.append(
            {
                "exp_name": exp_name,
                "model": display_name_by_exp.get(exp_name, exp_name),
                "name": str(summary["name"]),
                "eval_key": str(summary["eval_key"]),
                "dgt": int(summary["dgt"]),
                "dle": int(summary["dle"]),
                "nge": int(summary["nge"]),
                "mean_neighbor_embedding_cosine_distance": float(
                    summary["mean_neighbor_embedding_cosine_distance"]
                ),
                "neighborhood_map_csv_path": str(summary["neighborhood_map_csv_path"]),
                "pred_path": str(summary["pred_path"]),
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise ValueError("No SCNN metrics rows were collected")

    return metrics_df.sort_values(
        ["model", "dgt", "dle", "nge", "exp_name"],
        ignore_index=True,
    )


def select_cls_points(
    cls_metrics_df: pd.DataFrame,
    selected_perturbation: int,
    cls_metric_title: str,
) -> pd.DataFrame:
    required_columns = {
        "exp_name",
        "model",
        "perturbation",
        "summary_dir",
        cls_metric_title,
    }
    missing_columns = required_columns - set(cls_metrics_df.columns)
    if missing_columns:
        raise KeyError(
            "CLS metrics table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    selected_df = cls_metrics_df.loc[
        cls_metrics_df["perturbation"] == selected_perturbation
    ].copy()
    if selected_df.empty:
        available = sorted(cls_metrics_df["perturbation"].unique().tolist())
        raise ValueError(
            f"No CLS rows found for selected_perturbation={selected_perturbation}. "
            f"Available perturbations: {available}"
        )

    if selected_df["exp_name"].duplicated().any():
        dup_exp = selected_df.loc[
            selected_df["exp_name"].duplicated(), "exp_name"
        ].tolist()
        raise ValueError(
            "Expected one CLS row per experiment at the selected perturbation, "
            f"found duplicates for: {sorted(set(dup_exp))}"
        )

    return selected_df.rename(
        columns={
            "perturbation": "selected_perturbation",
            cls_metric_title: "cls_value",
            "summary_dir": "cls_summary_dir",
        }
    )[["exp_name", "model", "selected_perturbation", "cls_value", "cls_summary_dir"]]


def select_scnn_points(
    scnn_metrics_df: pd.DataFrame,
    selected_distance_lower_bound: int,
    selected_distance_upper_bound: int,
    selected_num_neighbors: int | None = None,
) -> pd.DataFrame:
    required_columns = {
        "exp_name",
        "model",
        "dgt",
        "dle",
        "nge",
        "mean_neighbor_embedding_cosine_distance",
        "neighborhood_map_csv_path",
        "pred_path",
    }
    missing_columns = required_columns - set(scnn_metrics_df.columns)
    if missing_columns:
        raise KeyError(
            "SCNN metrics table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    selected_df = scnn_metrics_df.loc[
        (scnn_metrics_df["dgt"] == selected_distance_lower_bound)
        & (scnn_metrics_df["dle"] == selected_distance_upper_bound)
    ].copy()
    if selected_num_neighbors is not None:
        selected_df = selected_df.loc[
            selected_df["nge"] == selected_num_neighbors
        ].copy()

    if selected_df.empty:
        available = sorted(scnn_metrics_df["dle"].unique().tolist())
        raise ValueError(
            "No SCNN rows found for "
            f"selected_distance_lower_bound={selected_distance_lower_bound}, "
            f"selected_distance_upper_bound={selected_distance_upper_bound} "
            f"and selected_num_neighbors={selected_num_neighbors}. "
            f"Available upper distance bounds: {available}"
        )

    if selected_df["exp_name"].duplicated().any():
        dup_df = selected_df.loc[
            selected_df["exp_name"].duplicated(keep=False),
            ["exp_name", "dgt", "dle", "nge"],
        ].sort_values(["exp_name", "dgt", "dle", "nge"])
        raise ValueError(
            "Expected one SCNN row per experiment at the selected distance bound. "
            "Filter more tightly, for example by selected_num_neighbors. "
            f"Conflicting rows:\n{dup_df.to_string(index=False)}"
        )

    return selected_df.rename(
        columns={
            "dgt": "selected_distance_lower_bound",
            "dle": "selected_distance_upper_bound",
            "mean_neighbor_embedding_cosine_distance": "scnn_value",
            "neighborhood_map_csv_path": "scnn_neighborhood_map_csv_path",
            "pred_path": "scnn_pred_path",
        }
    )[
        [
            "exp_name",
            "model",
            "selected_distance_lower_bound",
            "selected_distance_upper_bound",
            "nge",
            "scnn_value",
            "scnn_neighborhood_map_csv_path",
            "scnn_pred_path",
        ]
    ]


def merge_selected_points(
    cls_selected_df: pd.DataFrame,
    scnn_selected_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = cls_selected_df.merge(
        scnn_selected_df,
        on=["exp_name", "model"],
        how="inner",
        validate="one_to_one",
    )
    if merged_df.empty:
        raise ValueError("No experiments overlapped between selected CLS and SCNN rows")

    return merged_df.sort_values(["model", "exp_name"], ignore_index=True)


def build_chart(
    merged_df: pd.DataFrame,
    cls_axis_title: str,
    scnn_axis_title: str,
    color_range: List[str],
    width: int,
    height: int,
) -> alt.Chart:
    model_domain = list(merged_df["model"].drop_duplicates())
    color_scale = alt.Scale(
        domain=model_domain,
        range=color_range[: len(model_domain)],
    )

    base = alt.Chart(merged_df).encode(
        x=alt.X(
            "scnn_value:Q",
            title=scnn_axis_title,
            axis=alt.Axis(ticks=False),
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y(
            "cls_value:Q",
            title=cls_axis_title,
            axis=alt.Axis(ticks=False),
            scale=alt.Scale(domain=[0.4, 0.9]),
        ),
        color=alt.Color(
            "model:N",
            title=None,
            scale=color_scale,
            legend=alt.Legend(labelLimit=0),
        ),
        tooltip=[
            "model:N",
            "exp_name:N",
            alt.Tooltip("scnn_value:Q", title=scnn_axis_title, format=".4f"),
            alt.Tooltip("cls_value:Q", title=cls_axis_title, format=".4f"),
            "selected_distance_lower_bound:Q",
            "selected_distance_upper_bound:Q",
            "selected_perturbation:Q",
            "nge:Q",
            "cls_summary_dir:N",
            "scnn_neighborhood_map_csv_path:N",
            "scnn_pred_path:N",
        ],
    )
    points = base.mark_point(size=170, filled=True)
    labels = base.mark_text(align="left", dx=10, dy=-8).encode(text="model:N")
    return (
        (points + labels)
        .properties(width=width, height=height)
        .configure_title(fontSize=22)
        .configure_axis(titleFontSize=18, labelFontSize=18)
        .configure_legend(labelFontSize=18)
    )


def main() -> None:
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2"
    ckpt = "training_124999"

    selected_perturbation = 5
    selected_distance_lower_bound = 20
    selected_distance_upper_bound = 24
    selected_num_neighbors = 1

    experiment_configs = [
        {
            "exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0",
            "display_name": "DINOv2 lr4e-3",
        },
        {
            "exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0",
            "display_name": "Silica FullIm iBOT lr4e-3",
        },
        {
            "exp_name": "a2706135_dinov2",
            "display_name": "DINOv2 Meta",
        },
    ]

    cls_metric_name = "SRH7 Slide"
    cls_metric_configs = {
        "SRH7 Cell": {
            "summary_key": "instance_knn",
            "metric_key": "mca",
            "baseline": None,
        },
        "SRH7 Slide": {
            "summary_key": "mosaic_vote",
            "metric_key": "mca",
            "baseline": None,
        },
        "Tumor/Normal Cell": {
            "summary_key": "instance_knn_binary",
            "metric_key": "mca",
            "baseline": None,
        },
        "Tumor/Normal Slide": {
            "summary_key": "mosaic_vote_binary",
            "metric_key": "mca",
            "baseline": None,
        },
    }
    if cls_metric_name not in cls_metric_configs:
        raise ValueError(
            f"Unknown cls_metric_name={cls_metric_name!r}. "
            f"Available options: {sorted(cls_metric_configs)}"
        )
    cls_metric_config = cls_metric_configs[cls_metric_name]
    cls_metric_title = cls_metric_name
    cls_metric_key = cls_metric_config["metric_key"]
    cls_panels = [{"title": cls_metric_title, **cls_metric_config}]
    cls_run_key_prefix = "PERTURB"
    cls_databank_pred_glob = "*_INF_srh7v1sp1dot4m_*"
    cls_test_pred_glob = "*_INF_srh7v1tests64_PERTURB*_*"

    scnn_eval_key_prefix = "cellnbrring"
    scnn_pred_glob = f"*INF_srh7v1test_{scnn_eval_key_prefix}_*"
    scnn_neighborhood_map_csv_path_template = (
        "../silica_scnn/out/cellnbr_stats_nbr_8192_dgt{dist_min}_dle{dist_max}_nge1/"
        "sampled_neighborhood_map.csv"
    )

    cls_run_sets = [
        {
            "exp_name": cfg["exp_name"],
            "databank_pred_glob": cls_databank_pred_glob,
            "test_pred_glob": cls_test_pred_glob,
        }
        for cfg in experiment_configs
    ]
    scnn_run_sets = [
        {
            "exp_name": cfg["exp_name"],
            "pred_glob": scnn_pred_glob,
        }
        for cfg in experiment_configs
    ]
    display_name_by_exp = {
        cfg["exp_name"]: cfg["display_name"] for cfg in experiment_configs
    }
    color_range = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    width = 420
    height = 420
    out_stem = (
        "cls_vs_scnn_selected_points_"
        f"p{selected_perturbation:02d}_"
        f"d{selected_distance_lower_bound:02d}_{selected_distance_upper_bound:02d}"
    )

    formats = ["html", "png", "pdf"]

    cls_metrics_df = build_cls_curve_metrics_df(
        exp_root=exp_root,
        ckpt=ckpt,
        run_key_prefix=cls_run_key_prefix,
        run_sets=cls_run_sets,
        panels=cls_panels,
        display_name_by_exp=display_name_by_exp,
    )
    scnn_metrics_df = build_scnn_metrics_df(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=scnn_run_sets,
        neighborhood_map_csv_path_template=scnn_neighborhood_map_csv_path_template,
        eval_key_prefix=scnn_eval_key_prefix,
        display_name_by_exp=display_name_by_exp,
    )

    cls_selected_df = select_cls_points(
        cls_metrics_df=cls_metrics_df,
        selected_perturbation=selected_perturbation,
        cls_metric_title=cls_metric_title,
    )
    scnn_selected_df = select_scnn_points(
        scnn_metrics_df=scnn_metrics_df,
        selected_distance_lower_bound=selected_distance_lower_bound,
        selected_distance_upper_bound=selected_distance_upper_bound,
        selected_num_neighbors=selected_num_neighbors,
    )
    merged_df = merge_selected_points(
        cls_selected_df=cls_selected_df,
        scnn_selected_df=scnn_selected_df,
    )

    chart = build_chart(
        merged_df=merged_df,
        cls_axis_title=f"{cls_metric_title} {cls_metric_key.upper()}",
        scnn_axis_title="Mean Neighborhood Embedding Cosine Distance",
        color_range=color_range,
        width=width,
        height=height,
    ).properties(
        title=(
            "CLS vs SCNN Selected Evaluation Points "
            f"(PERTURB={selected_perturbation}, "
            f"dgt={selected_distance_lower_bound}, "
            f"dle={selected_distance_upper_bound})"
        )
    )

    out_dir = os.path.dirname(out_stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cls_metrics_df.to_csv(f"{out_stem}_cls_curve.csv", index=False)
    scnn_metrics_df.to_csv(f"{out_stem}_scnn_curve.csv", index=False)
    merged_df.to_csv(f"{out_stem}_selected_points.csv", index=False)
    save_chart(chart, out_stem=out_stem, formats=formats)

    print(f"Saved CLS curve table to {out_stem}_cls_curve.csv")
    print(f"Saved SCNN curve table to {out_stem}_scnn_curve.csv")
    print(f"Saved merged selected points to {out_stem}_selected_points.csv")
    print("Saved chart files: " + ", ".join(f"{out_stem}.{fmt}" for fmt in formats))


if __name__ == "__main__":
    main()

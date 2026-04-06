import json
import logging
import os
import re
from os.path import join as opj
from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ts2.utils.silica_sc_cls.eval_cell_inference_knn import (
    find_matching_prediction_paths,
    load_prediction,
)


def load_neighborhood_map(neighborhood_map_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(neighborhood_map_csv_path):
        raise FileNotFoundError(
            f"Neighborhood map CSV does not exist: {neighborhood_map_csv_path}"
        )

    neighborhoods_df = pd.read_csv(neighborhood_map_csv_path)
    required_columns = {
        "anchor_path",
        "neighbor_paths_json",
        "neighbor_distances_json",
    }
    missing_columns = required_columns - set(neighborhoods_df.columns)
    if missing_columns:
        raise KeyError(
            "Neighborhood map CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if neighborhoods_df.empty:
        raise ValueError(
            f"Neighborhood map CSV has no rows: {neighborhood_map_csv_path}"
        )
    return neighborhoods_df


def build_runs_from_sets(
    exp_root: str,
    ckpt: str,
    run_sets: list[dict[str, str]],
    neighborhood_map_csv_path_template: str,
    eval_key_prefix: str = "cellnbr",
) -> list[dict[str, str]]:
    if not exp_root:
        raise ValueError("Expected a non-empty exp_root")
    if not ckpt:
        raise ValueError("Expected a non-empty ckpt")
    if not neighborhood_map_csv_path_template:
        raise ValueError("Expected a non-empty neighborhood_map_csv_path_template")
    if not eval_key_prefix:
        raise ValueError("Expected a non-empty eval_key_prefix")

    runs: list[dict[str, str]] = []
    for run_set in run_sets:
        required_keys = {"exp_name", "pred_glob"}
        missing = required_keys - set(run_set)
        if missing:
            raise KeyError(f"Run set is missing required keys: {sorted(missing)}")

        exp_name = run_set["exp_name"]
        pattern = opj(
            exp_root,
            exp_name,
            "models",
            "eval",
            ckpt,
            run_set["pred_glob"],
            "predictions",
            "pred.pt",
        )
        matches = find_matching_prediction_paths(pattern)
        if not matches:
            raise ValueError(
                f"Expected at least one prediction for run set {exp_name}, "
                f"found 0 with pattern: {pattern}"
            )

        run_set_runs: list[dict[str, str]] = []
        for pred_path in matches:
            eval_key_info = extract_eval_key(
                pred_path=pred_path,
                eval_key_prefix=eval_key_prefix,
            )
            run_set_runs.append(
                {
                    "exp_name": exp_name,
                    "name": f"{exp_name}_{eval_key_info['eval_key']}",
                    "neighborhood_map_csv_path": resolve_neighborhood_map_csv_path(
                        neighborhood_map_csv_path_template=neighborhood_map_csv_path_template,
                        celldist_mode=eval_key_info["celldist_mode"],
                        dist_min=eval_key_info["dist_min"],
                        dist_max=eval_key_info["dist_max"],
                    ),
                    "pred_path": pred_path,
                    **eval_key_info,
                }
            )

        eval_key_to_pred_path = {
            run["eval_key"]: run["pred_path"] for run in run_set_runs
        }
        if len(eval_key_to_pred_path) != len(run_set_runs):
            raise ValueError(
                f"Found duplicate eval_key values for run set {exp_name}: "
                f"{[run['pred_path'] for run in run_set_runs]}"
            )
        runs.extend(sorted(run_set_runs, key=lambda run: run["name"]))
    return runs


def extract_eval_key(
    pred_path: str, eval_key_prefix: str = "cellnbr"
) -> dict[str, str]:
    if not eval_key_prefix:
        raise ValueError("Expected a non-empty eval_key_prefix")
    pattern = rf"{re.escape(eval_key_prefix)}_dgt(\d+)_dle(\d+)_nge(\d+)"
    match = re.search(pattern, pred_path)
    if match is None:
        raise ValueError(
            "Could not extract evaluation key from prediction path. "
            f"Expected '{eval_key_prefix}_dgt..._dle..._nge...' in: {pred_path}"
        )
    dgt = int(match.group(1))
    dle = int(match.group(2))
    nge = int(match.group(3))
    return {
        "eval_key": f"{eval_key_prefix}_dgt{dgt}_dle{dle}_nge{nge}",
        "celldist_mode": eval_key_prefix,
        "eval_key_digits": f"{dgt:02d}{dle:02d}",
        "dist_min": dgt,
        "dist_max": dle,
        "dgt": dgt,
        "dle": dle,
        "nge": nge,
    }


def split_eval_key_digits(eval_key_digits: str) -> tuple[int, int]:
    if len(eval_key_digits) != 4:
        raise ValueError(
            f"Expected eval_key_digits to have length 4, got: {eval_key_digits}"
        )
    left = int(eval_key_digits[:2])
    right = int(eval_key_digits[2:])
    return left, right


def resolve_neighborhood_map_csv_path(
    neighborhood_map_csv_path_template: str,
    celldist_mode: str,
    dist_min: int,
    dist_max: int,
) -> str:
    return neighborhood_map_csv_path_template.format(
        celldist_mode=celldist_mode,
        dist_min=dist_min,
        dist_max=dist_max,
    )


def extract_exp_key(exp_name: str) -> str:
    exp_key = exp_name.split("_", 1)[0]
    if not exp_key:
        raise ValueError(f"Could not extract exp_key from exp_name: {exp_name}")
    return exp_key


def build_embedding_frame(pred: Dict[str, Any]) -> pd.DataFrame:
    path_df = pd.DataFrame(
        {
            "path": pred["path"],
            "pred_label": [str(x) for x in pred["label"]],
        }
    )
    if path_df["path"].duplicated().any():
        dup_paths = path_df.loc[path_df["path"].duplicated(), "path"].tolist()
        raise ValueError(
            "Prediction file contains duplicate paths: " + ", ".join(dup_paths[:10])
        )

    embeddings = pred["embeddings"].cpu().numpy().astype(np.float32, copy=False)
    emb_df = pd.DataFrame({"path": pred["path"]})
    emb_df["embedding"] = [row for row in embeddings]
    emb_df["pred_label"] = path_df["pred_label"]
    return emb_df


def explode_neighborhood_pairs(neighborhoods_df: pd.DataFrame) -> pd.DataFrame:
    base_df = neighborhoods_df.reset_index(drop=True).copy()
    if "neighborhood_index" not in base_df.columns:
        base_df["neighborhood_index"] = np.arange(len(base_df), dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for _, row in base_df.iterrows():
        neighbor_paths = json.loads(row["neighbor_paths_json"])
        neighbor_distances = json.loads(row["neighbor_distances_json"])
        if len(neighbor_paths) != len(neighbor_distances):
            raise ValueError(
                "neighbor_paths_json and neighbor_distances_json length mismatch for "
                f"neighborhood_index={row['neighborhood_index']}"
            )
        if len(neighbor_paths) == 0:
            raise ValueError(
                f"Neighborhood has zero neighbors: neighborhood_index={row['neighborhood_index']}"
            )

        num_neighbors = int(row["num_neighbors"]) if "num_neighbors" in row else None
        if num_neighbors is not None and num_neighbors != len(neighbor_paths):
            raise ValueError(
                f"num_neighbors mismatch for neighborhood_index={row['neighborhood_index']}: "
                f"expected {num_neighbors}, got {len(neighbor_paths)}"
            )

        for neighbor_path, pixel_distance in zip(
            neighbor_paths, neighbor_distances, strict=True
        ):
            rows.append(
                {
                    "neighborhood_index": int(row["neighborhood_index"]),
                    "anchor_path": row["anchor_path"],
                    "neighbor_path": str(neighbor_path),
                    "pixel_distance": float(pixel_distance),
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise ValueError("Neighborhood map produced no anchor-neighbor rows")
    return pair_df


def attach_pair_embeddings(pair_df: pd.DataFrame, emb_df: pd.DataFrame) -> pd.DataFrame:
    anchor = emb_df.rename(
        columns={
            "path": "anchor_path",
            "embedding": "anchor_embedding",
            "pred_label": "anchor_pred_label",
        }
    )
    neighbor = emb_df.rename(
        columns={
            "path": "neighbor_path",
            "embedding": "neighbor_embedding",
            "pred_label": "neighbor_pred_label",
        }
    )

    merged = pair_df.merge(anchor, on="anchor_path", how="left", validate="many_to_one")
    merged = merged.merge(
        neighbor, on="neighbor_path", how="left", validate="many_to_one"
    )

    missing_anchor = int(merged["anchor_embedding"].isna().sum())
    missing_neighbor = int(merged["neighbor_embedding"].isna().sum())
    if missing_anchor or missing_neighbor:
        raise ValueError(
            "Could not match all neighborhood paths to embeddings: "
            f"missing anchor_path={missing_anchor}, "
            f"missing neighbor_path={missing_neighbor}"
        )
    return merged


def compute_pair_embedding_metrics(pair_df: pd.DataFrame) -> pd.DataFrame:
    emb_anchor = np.stack(pair_df["anchor_embedding"].to_list()).astype(
        np.float32, copy=False
    )
    emb_neighbor = np.stack(pair_df["neighbor_embedding"].to_list()).astype(
        np.float32, copy=False
    )
    if emb_anchor.shape != emb_neighbor.shape:
        raise ValueError(
            "Embedding shape mismatch: "
            f"anchor={tuple(emb_anchor.shape)}, neighbor={tuple(emb_neighbor.shape)}"
        )

    emb_anchor_norm = np.linalg.norm(emb_anchor, axis=1)
    emb_neighbor_norm = np.linalg.norm(emb_neighbor, axis=1)
    denom = emb_anchor_norm * emb_neighbor_norm
    if np.any(denom <= 0):
        raise ValueError("Found non-positive embedding norm while computing cosine")
    cosine_similarity = np.sum(emb_anchor * emb_neighbor, axis=1) / denom
    cosine_distance = 1.0 - cosine_similarity

    out = pair_df.copy()
    out["embedding_cosine_distance"] = cosine_distance.astype(np.float32)
    return out.drop(columns=["anchor_embedding", "neighbor_embedding"])


def summarize_neighborhood_metrics(
    neighborhoods_df: pd.DataFrame, pair_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    base_df = neighborhoods_df.reset_index(drop=True).copy()
    if "neighborhood_index" not in base_df.columns:
        base_df["neighborhood_index"] = np.arange(len(base_df), dtype=np.int64)
    agg_df = (
        pair_metrics_df.groupby("neighborhood_index", sort=True)
        .agg(
            computed_num_neighbors=("neighbor_path", "size"),
            mean_neighbor_pixel_distance=("pixel_distance", "mean"),
            mean_neighbor_embedding_cosine_distance=(
                "embedding_cosine_distance",
                "mean",
            ),
        )
        .reset_index()
    )

    out_df = base_df.merge(
        agg_df, on="neighborhood_index", how="left", validate="one_to_one"
    )
    if "num_neighbors" in out_df.columns:
        if not np.array_equal(
            out_df["num_neighbors"].to_numpy(dtype=np.int64, copy=False),
            out_df["computed_num_neighbors"].to_numpy(dtype=np.int64, copy=False),
        ):
            raise ValueError(
                "Computed neighborhood sizes do not match input num_neighbors values"
            )
    else:
        out_df["num_neighbors"] = out_df["computed_num_neighbors"]

    if (
        out_df[
            [
                "num_neighbors",
                "mean_neighbor_pixel_distance",
                "mean_neighbor_embedding_cosine_distance",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Failed to compute neighborhood metrics for all rows")
    return out_df.drop(columns=["computed_num_neighbors"])


def build_summary(
    out_df: pd.DataFrame,
    neighborhood_map_csv_path: str,
    pred_path: str,
    exp_name: str | None = None,
    name: str | None = None,
    eval_key: str | None = None,
    celldist_mode: str | None = None,
    eval_key_digits: str | None = None,
    dist_min: int | None = None,
    dist_max: int | None = None,
    dgt: int | None = None,
    dle: int | None = None,
    nge: int | None = None,
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "neighborhood_map_csv_path": neighborhood_map_csv_path,
        "pred_path": pred_path,
        "num_neighborhoods": int(len(out_df)),
        "mean_num_neighbors": float(out_df["num_neighbors"].mean()),
        "mean_neighbor_pixel_distance": float(
            out_df["mean_neighbor_pixel_distance"].mean()
        ),
        "mean_neighbor_embedding_cosine_distance": float(
            out_df["mean_neighbor_embedding_cosine_distance"].mean()
        ),
        "median_neighbor_embedding_cosine_distance": float(
            out_df["mean_neighbor_embedding_cosine_distance"].median()
        ),
    }
    if exp_name is not None:
        summary["exp_name"] = exp_name
    if name is not None:
        summary["name"] = name
    if eval_key is not None:
        summary["eval_key"] = eval_key
    if celldist_mode is not None:
        summary["celldist_mode"] = celldist_mode
    if eval_key_digits is not None:
        summary["eval_key_digits"] = eval_key_digits
    if dist_min is not None:
        summary["dist_min"] = dist_min
    if dist_max is not None:
        summary["dist_max"] = dist_max
    if dgt is not None:
        summary["dgt"] = dgt
    if dle is not None:
        summary["dle"] = dle
    if nge is not None:
        summary["nge"] = nge
    return summary


def evaluate_run(
    neighborhood_map_csv_path: str,
    pred_path: str,
    exp_name: str | None = None,
    name: str | None = None,
    eval_key: str | None = None,
    celldist_mode: str | None = None,
    eval_key_digits: str | None = None,
    dist_min: int | None = None,
    dist_max: int | None = None,
    dgt: int | None = None,
    dle: int | None = None,
    nge: int | None = None,
) -> dict[str, float | int | str]:
    neighborhoods_df = load_neighborhood_map(neighborhood_map_csv_path)
    pred = load_prediction(pred_path)
    emb_df = build_embedding_frame(pred)
    pair_df = explode_neighborhood_pairs(neighborhoods_df)
    pair_df = attach_pair_embeddings(pair_df, emb_df)
    pair_df = compute_pair_embedding_metrics(pair_df)
    out_df = summarize_neighborhood_metrics(neighborhoods_df, pair_df)
    return build_summary(
        out_df=out_df,
        neighborhood_map_csv_path=neighborhood_map_csv_path,
        pred_path=pred_path,
        exp_name=exp_name,
        name=name,
        eval_key=eval_key,
        celldist_mode=celldist_mode,
        eval_key_digits=eval_key_digits,
        dist_min=dist_min,
        dist_max=dist_max,
        dgt=dgt,
        dle=dle,
        nge=nge,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/"
    ckpt = "training_124999"
    eval_key_prefix = "cellnbrring"
    neighborhood_map_csv_path_template = (
        "out/cellnbr_stats_nbr_8192_dgt{dist_min}_dle{dist_max}_nge1/"
        "sampled_neighborhood_map.csv"
    )
    exp_name_label_map = {
        "04e0bf39": "DINOv2, LR4e-3",
        "ca187b7c": "Silica, full image iBOT, LR4e-3",
        # "3122d0c0": "DINOv2, LR4e-3",
        # "bead0872": "Silica, full image iBOT, LR4e-3",
        # "1dfffb8f": "Silica, inside iBOT, LR4e-3",
        # "1526bfe8": "Silica, full image iBOT, LR1e-3",
        # "8751a922": "Silica, inside iBOT, LR1e-3",
    }
    run_sets = [
        {
            "exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0",
            "pred_glob": f"*INF_srh7v1test_{eval_key_prefix}_*",
        },
        {
            "exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0",
            "pred_glob": f"*INF_srh7v1test_{eval_key_prefix}_*",
        },
        # {
        #    "exp_name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
        #    "pred_glob": "*INF_srh7v1test_cellnbr*",
        # },
        # {
        #    "exp_name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
        #    "pred_glob": "*INF_srh7v1test_cellnbr*",
        # },
        # {
        #    "exp_name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
        #    "pred_glob": "*INF_srh7v1test_cellnbr*",
        # },
        # {
        #    "exp_name": "1526bfe8_Mar24-15-02-22_sd1000_dev_nomaskobw_lr13_tune0",
        #    "pred_glob": "*INF_srh7v1test_cellnbr*",
        # },
        # {
        #    "exp_name": "8751a922_Mar24-15-02-22_sd1000_dev_maskobw_lr13_tune1",
        #    "pred_glob": "*INF_srh7v1test_cellnbr*",
        # },
    ]

    runs = build_runs_from_sets(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=run_sets,
        neighborhood_map_csv_path_template=neighborhood_map_csv_path_template,
        eval_key_prefix=eval_key_prefix,
    )

    summaries: list[dict[str, float | int | str]] = []
    for run in tqdm(runs, desc="Evaluating runs"):
        print(f"Evaluating {run['name']}")
        summaries.append(evaluate_run(**run))

    if not summaries:
        raise ValueError("Expected at least one run summary to aggregate")

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["name", "celldist_mode", "dgt", "dle", "nge"],
        kind="stable",
    )
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
    if len(missing_labels) > 0:
        raise ValueError(
            "Missing exp_name legend labels for: " + ", ".join(sorted(missing_labels))
        )
    plot_df = plot_df.sort_values(
        by=["exp", "dgt", "dle", "nge"],
        kind="stable",
    ).reset_index(drop=True)
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

    interval_base = alt.Chart(plot_df).encode(
        y=y_encoding,
        color=color_encoding,
        tooltip=tooltip,
    )
    interval_rules = interval_base.mark_rule(strokeWidth=3, opacity=0.6).encode(
        x=alt.X(
            "dgt:Q",
            title="Distance bound",
            axis=alt.Axis(values=x_values, tickSize=0),
            scale=alt.Scale(zero=False),
        ),
        x2="dle:Q",
        detail=["exp:N", "dgt:Q", "dle:Q", "nge:Q"],
    )
    interval_chart = interval_rules

    endpoint_base = alt.Chart(plot_df).encode(
        x=x_encoding,
        y=y_encoding,
        color=color_encoding,
        tooltip=tooltip,
    )
    endpoint_lines = endpoint_base.mark_line(strokeWidth=3).encode(
        detail="exp:N",
        order=alt.Order("dle:Q"),
    )
    endpoint_points = endpoint_base.mark_point(size=70, filled=True)
    endpoint_chart = endpoint_lines + endpoint_points

    interval_chart = interval_chart.properties(
        title="Mean Neighborhood Embedding Cosine Distance by [dgt, dle]",
        width=400,
        height=400,
    )
    endpoint_chart = endpoint_chart.properties(
        title="Mean Neighborhood Embedding Cosine Distance by dle",
        width=400,
        height=400,
    )

    interval_chart_out_path = "neighborhood_embedding_cosine_distance_interval"
    interval_chart.save(f"{interval_chart_out_path}.html")
    interval_chart.save(f"{interval_chart_out_path}.pdf")
    interval_chart.save(f"{interval_chart_out_path}.png")

    endpoint_chart_out_path = "neighborhood_embedding_cosine_distance_by_dle"
    endpoint_chart.save(f"{endpoint_chart_out_path}.html")
    endpoint_chart.save(f"{endpoint_chart_out_path}.pdf")
    endpoint_chart.save(f"{endpoint_chart_out_path}.png")


if __name__ == "__main__":
    main()

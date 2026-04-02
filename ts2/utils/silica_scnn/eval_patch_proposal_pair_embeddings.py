import logging
import json
import os
import re
from glob import glob
from os.path import join as opj
from os.path import splitext
from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ts2.utils.silica_sc_cls.eval_cell_inference_knn import (
    find_matching_prediction_paths,
    infer_results_dir_from_prediction_path,
    load_prediction,
)


def resolve_out_stem(pair_map_csv_path: str, out_stem: str | None) -> str:
    if out_stem is not None:
        return out_stem
    base, _ = splitext(pair_map_csv_path)
    return f"{base}_embedding_eval"


def load_pair_map(pair_map_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(pair_map_csv_path):
        raise FileNotFoundError(f"Pair map CSV does not exist: {pair_map_csv_path}")

    pairs_df = pd.read_csv(pair_map_csv_path)
    required_columns = {"path_a", "path_b", "distance"}
    missing_columns = required_columns - set(pairs_df.columns)
    if missing_columns:
        raise KeyError(
            f"Pair map CSV is missing required columns: {sorted(missing_columns)}"
        )
    if pairs_df.empty:
        raise ValueError(f"Pair map CSV has no rows: {pair_map_csv_path}")
    return pairs_df


def build_runs_from_sets(
    exp_root: str,
    ckpt: str,
    run_sets: list[dict[str, str]],
    pair_map_csv_path_template: str,
) -> list[dict[str, str]]:
    if not exp_root:
        raise ValueError("Expected a non-empty exp_root")
    if not ckpt:
        raise ValueError("Expected a non-empty ckpt")
    if not pair_map_csv_path_template:
        raise ValueError("Expected a non-empty pair_map_csv_path_template")

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
            eval_key_info = extract_eval_key(pred_path)
            run_set_runs.append(
                {
                    "exp_name": exp_name,
                    "name": f"{exp_name}_{eval_key_info['eval_key']}",
                    "pair_map_csv_path": resolve_pair_map_csv_path(
                        pair_map_csv_path_template=pair_map_csv_path_template,
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


def extract_eval_key(pred_path: str) -> dict[str, str]:
    match = re.search(r"celldist_([A-Za-z]+)(\d+)", pred_path)
    if match is None:
        raise ValueError(
            "Could not extract evaluation key from prediction path. "
            f"Expected 'celldist_(letters)(digits)' in: {pred_path}"
        )
    key_alpha = match.group(1)
    key_digits = match.group(2)
    dist_min, dist_max = split_eval_key_digits(key_digits)
    return {
        "eval_key": f"celldist_{key_alpha}{key_digits}",
        "celldist_mode": key_alpha,
        "eval_key_digits": key_digits,
        "dist_min": dist_min,
        "dist_max": dist_max,
    }


def split_eval_key_digits(eval_key_digits: str) -> tuple[int, int]:
    if len(eval_key_digits) != 4:
        raise ValueError(
            f"Expected eval_key_digits to have length 4, got: {eval_key_digits}"
        )
    left = int(eval_key_digits[:2])
    right = int(eval_key_digits[2:])
    return left, right


def resolve_pair_map_csv_path(
    pair_map_csv_path_template: str,
    celldist_mode: str,
    dist_min: int,
    dist_max: int,
) -> str:
    return pair_map_csv_path_template.format(
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


def attach_pair_embeddings(
    pairs_df: pd.DataFrame, emb_df: pd.DataFrame
) -> pd.DataFrame:
    left = emb_df.rename(
        columns={
            "path": "path_a",
            "embedding": "embedding_a",
            "pred_label": "pred_label_a",
        }
    )
    right = emb_df.rename(
        columns={
            "path": "path_b",
            "embedding": "embedding_b",
            "pred_label": "pred_label_b",
        }
    )

    merged = pairs_df.merge(left, on="path_a", how="left", validate="many_to_one")
    merged = merged.merge(right, on="path_b", how="left", validate="many_to_one")

    missing_a = int(merged["embedding_a"].isna().sum())
    missing_b = int(merged["embedding_b"].isna().sum())
    if missing_a or missing_b:
        raise ValueError(
            "Could not match all pair paths to embeddings: "
            f"missing path_a={missing_a}, missing path_b={missing_b}"
        )
    return merged


def compute_embedding_metrics(pairs_df: pd.DataFrame) -> pd.DataFrame:
    emb_a = np.stack(pairs_df["embedding_a"].to_list()).astype(np.float32, copy=False)
    emb_b = np.stack(pairs_df["embedding_b"].to_list()).astype(np.float32, copy=False)
    if emb_a.shape != emb_b.shape:
        raise ValueError(
            f"Embedding shape mismatch: path_a={tuple(emb_a.shape)}, path_b={tuple(emb_b.shape)}"
        )

    delta = emb_a - emb_b
    euclidean = np.linalg.norm(delta, axis=1)

    emb_a_norm = np.linalg.norm(emb_a, axis=1)
    emb_b_norm = np.linalg.norm(emb_b, axis=1)
    denom = emb_a_norm * emb_b_norm
    if np.any(denom <= 0):
        raise ValueError("Found non-positive embedding norm while computing cosine")
    cosine_similarity = np.sum(emb_a * emb_b, axis=1) / denom

    out = pairs_df.copy()
    out["embedding_euclidean_distance"] = euclidean.astype(np.float32)
    out["embedding_cosine_similarity"] = cosine_similarity.astype(np.float32)
    return out.drop(columns=["embedding_a", "embedding_b"])


def build_summary(
    out_df: pd.DataFrame,
    pair_map_csv_path: str,
    pred_path: str,
    exp_name: str | None = None,
    name: str | None = None,
    eval_key: str | None = None,
    celldist_mode: str | None = None,
    eval_key_digits: str | None = None,
    dist_min: int | None = None,
    dist_max: int | None = None,
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "pair_map_csv_path": pair_map_csv_path,
        "pred_path": pred_path,
        "num_pairs": int(len(out_df)),
        "mean_pixel_distance": float(out_df["distance"].mean()),
        "mean_embedding_euclidean_distance": float(
            out_df["embedding_euclidean_distance"].mean()
        ),
        "mean_embedding_cosine_similarity": float(
            out_df["embedding_cosine_similarity"].mean()
        ),
        "median_embedding_cosine_similarity": float(
            out_df["embedding_cosine_similarity"].median()
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
    return summary


def save_summary(summary: dict[str, float | int | str], out_stem: str) -> str:
    out_path = f"{out_stem}.json"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    return out_path


def save_pairs_df(out_df: pd.DataFrame, out_stem: str) -> str:
    out_path = f"{out_stem}.csv"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path


def evaluate_run(
    pair_map_csv_path: str,
    pred_path: str,
    out_stem: str | None = None,
    exp_name: str | None = None,
    name: str | None = None,
    eval_key: str | None = None,
    celldist_mode: str | None = None,
    eval_key_digits: str | None = None,
    dist_min: int | None = None,
    dist_max: int | None = None,
) -> dict[str, float | int | str]:
    pairs_df = load_pair_map(pair_map_csv_path)
    pred = load_prediction(pred_path)
    emb_df = build_embedding_frame(pred)
    out_df = attach_pair_embeddings(pairs_df, emb_df)
    out_df = compute_embedding_metrics(out_df)
    summary = build_summary(
        out_df=out_df,
        pair_map_csv_path=pair_map_csv_path,
        pred_path=pred_path,
        exp_name=exp_name,
        name=name,
        eval_key=eval_key,
        celldist_mode=celldist_mode,
        eval_key_digits=eval_key_digits,
        dist_min=dist_min,
        dist_max=dist_max,
    )
    # if out_stem is None:
    #    out_dir = infer_results_dir_from_prediction_path(
    #        pred_path, run_dir_prefix="pair"
    #    )
    #    os.makedirs(out_dir, exist_ok=True)
    #    out_stem = opj(out_dir, "pair_embedding_eval")
    #
    # csv_out_path = save_pairs_df(out_df=out_df, out_stem=out_stem)
    # summary_out_path = save_summary(summary=summary, out_stem=out_stem)
    #
    # print(f"Saved pair embedding eval CSV to {csv_out_path}")
    # print(f"Saved pair embedding eval summary to {summary_out_path}")
    # print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/"
    ckpt = "training_124999"
    pair_map_csv_path_template = (
        "out/celldist_stats_{celldist_mode}_8192_{dist_min}_{dist_max}/"
        "sampled_pair_map.csv"
    )
    exp_name_label_map = {
        "3122d0c0": "DINOv2, LR4e-3",
        "bead0872": "Silica, full image iBOT, LR4e-3",
        "1dfffb8f": "Silica, inside iBOT, LR4e-3",
        "1526bfe8": "Silica, full image iBOT, LR1e-3",
        "8751a922": "Silica, inside iBOT, LR1e-3",
    }
    run_sets = [
        {
            "exp_name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
            "pred_glob": "*INF_srh7v1test_celldist_nn*",
        },
        {
            "exp_name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
            "pred_glob": "*INF_srh7v1test_celldist_nn*",
        },
        {
            "exp_name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
            "pred_glob": "*INF_srh7v1test_celldist_nn*",
        },
        {
            "exp_name": "1526bfe8_Mar24-15-02-22_sd1000_dev_nomaskobw_lr13_tune0",
            "pred_glob": "*INF_srh7v1test_celldist_nn*",
        },
        {
            "exp_name": "8751a922_Mar24-15-02-22_sd1000_dev_maskobw_lr13_tune1",
            "pred_glob": "*INF_srh7v1test_celldist_nn*",
        },
    ]
    run_sets = [
        {
            "exp_name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
            "pred_glob": "*INF_srh7v1test_celldist_ap*",
        },
        {
            "exp_name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
            "pred_glob": "*INF_srh7v1test_celldist_ap*",
        },
        {
            "exp_name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
            "pred_glob": "*INF_srh7v1test_celldist_ap*",
        },
        {
            "exp_name": "1526bfe8_Mar24-15-02-22_sd1000_dev_nomaskobw_lr13_tune0",
            "pred_glob": "*INF_srh7v1test_celldist_ap*",
        },
        {
            "exp_name": "8751a922_Mar24-15-02-22_sd1000_dev_maskobw_lr13_tune1",
            "pred_glob": "*INF_srh7v1test_celldist_ap*",
        },
    ]
    runs = build_runs_from_sets(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=run_sets,
        pair_map_csv_path_template=pair_map_csv_path_template,
    )

    summaries: list[dict[str, float | int | str]] = []
    for run in tqdm(runs, desc="Evaluating runs"):
        print(f"Evaluating {run['name']}")
        summaries.append(evaluate_run(**run))

    if not summaries:
        raise ValueError("Expected at least one run summary to aggregate")

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["name", "celldist_mode", "dist_min", "dist_max"],
        kind="stable",
    )
    plot_df = summary_df[
        [
            "exp_name",
            "celldist_mode",
            "dist_min",
            "dist_max",
            "mean_embedding_cosine_similarity",
        ]
    ].copy()
    plot_df["exp_key"] = plot_df["exp_name"].map(extract_exp_key)
    plot_df["exp"] = plot_df["exp_key"].map(exp_name_label_map)
    missing_labels = plot_df.loc[plot_df["exp"].isna(), "exp_name"].unique()
    if len(missing_labels) > 0:
        raise ValueError(
            "Missing exp_name legend labels for: " + ", ".join(sorted(missing_labels))
        )
    plot_df = plot_df.loc[(plot_df["dist_max"] - plot_df["dist_min"]) == 4]
    if plot_df.empty:
        raise ValueError("No rows matched plotting filter: dist_max - dist_min == 4")
    x_values = sorted(plot_df["dist_max"].unique().tolist())
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "dist_max:Q",
                title="dist_max",
                axis=alt.Axis(values=x_values, tickSize=0),
            ),
            y=alt.Y(
                "mean_embedding_cosine_similarity:Q",
                title="mean_embedding_cosine_similarity",
                axis=alt.Axis(tickSize=0),
            ),
            color=alt.Color("exp:N", title="Experiment"),
            detail="exp:N",
            tooltip=[
                "exp:N",
                "exp_name:N",
                "celldist_mode:N",
                "dist_min:Q",
                "dist_max:Q",
                "mean_embedding_cosine_similarity:Q",
            ],
        )
        .properties(
            title="Mean Embedding Cosine Similarity by Distance Window",
            width=700,
            height=400,
        )
    )
    chart_out_path = "pair_embedding_cosine_similarity_by_dist_max"
    chart.save(f"{chart_out_path}.html")
    chart.save(f"{chart_out_path}.pdf")
    chart.save(f"{chart_out_path}.png")

    # import pdb; pdb.set_trace()
    # print(summary_df.to_string(index=False))
    # print(plot_df.to_string(index=False))
    # print(f"Saved Altair chart to {chart_out_path}")


if __name__ == "__main__":
    main()

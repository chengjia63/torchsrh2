import json
import logging
import os
import re
from os.path import join as opj
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ts2.utils.silica_sc_cls.eval_cell_inference_knn import (
    find_matching_prediction_paths,
    infer_results_dir_from_prediction_path,
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
    assert (
        not neighborhoods_df.empty
    ), f"Neighborhood map CSV has no rows: {neighborhood_map_csv_path}"
    return neighborhoods_df


def build_runs_from_sets(
    exp_root: str,
    ckpt: str,
    run_sets: list[dict[str, str]],
    neighborhood_map_csv_path_template: str,
    default_pred_glob: str | None = None,
    eval_key_prefix: str = "cellnbr",
) -> list[dict[str, str]]:
    assert exp_root, "Expected a non-empty exp_root"
    assert ckpt, "Expected a non-empty ckpt"
    assert (
        neighborhood_map_csv_path_template
    ), "Expected a non-empty neighborhood_map_csv_path_template"
    assert eval_key_prefix, "Expected a non-empty eval_key_prefix"

    runs: list[dict[str, str]] = []
    for run_set in run_sets:
        required_keys = {"exp_name"}
        missing = required_keys - set(run_set)
        if missing:
            raise KeyError(f"Run set is missing required keys: {sorted(missing)}")

        exp_name = run_set["exp_name"]
        pred_glob = run_set.get("pred_glob", default_pred_glob)
        assert pred_glob, f"Expected pred_glob for run set {exp_name}"
        pattern = opj(
            exp_root,
            exp_name,
            "models",
            "eval",
            ckpt,
            pred_glob,
            "predictions",
            "pred.pt",
        )
        matches = find_matching_prediction_paths(pattern)
        assert matches, (
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
        assert len(eval_key_to_pred_path) == len(run_set_runs), (
            f"Found duplicate eval_key values for run set {exp_name}: "
            f"{[run['pred_path'] for run in run_set_runs]}"
        )
        runs.extend(sorted(run_set_runs, key=lambda run: run["name"]))
    return runs


def extract_eval_key(
    pred_path: str, eval_key_prefix: str = "cellnbr"
) -> dict[str, str]:
    assert eval_key_prefix, "Expected a non-empty eval_key_prefix"
    pattern = rf"{re.escape(eval_key_prefix)}_dgt(\d+)_dle(\d+)_nge(\d+)"
    match = re.search(pattern, pred_path)
    assert match is not None, (
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
    assert (
        len(eval_key_digits) == 4
    ), f"Expected eval_key_digits to have length 4, got: {eval_key_digits}"
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
    assert exp_key, f"Could not extract exp_key from exp_name: {exp_name}"
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
        assert False, "Prediction file contains duplicate paths: " + ", ".join(
            dup_paths[:10]
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
        assert len(neighbor_paths) == len(neighbor_distances), (
            "neighbor_paths_json and neighbor_distances_json length mismatch for "
            f"neighborhood_index={row['neighborhood_index']}"
        )
        assert (
            len(neighbor_paths) > 0
        ), f"Neighborhood has zero neighbors: neighborhood_index={row['neighborhood_index']}"

        num_neighbors = int(row["num_neighbors"]) if "num_neighbors" in row else None
        assert num_neighbors is None or num_neighbors == len(neighbor_paths), (
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
    assert not pair_df.empty, "Neighborhood map produced no anchor-neighbor rows"
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
    assert not (missing_anchor or missing_neighbor), (
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
    assert emb_anchor.shape == emb_neighbor.shape, (
        "Embedding shape mismatch: "
        f"anchor={tuple(emb_anchor.shape)}, neighbor={tuple(emb_neighbor.shape)}"
    )

    emb_anchor_norm = np.linalg.norm(emb_anchor, axis=1)
    emb_neighbor_norm = np.linalg.norm(emb_neighbor, axis=1)
    denom = emb_anchor_norm * emb_neighbor_norm
    assert not np.any(
        denom <= 0
    ), "Found non-positive embedding norm while computing cosine"
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
        assert np.array_equal(
            out_df["num_neighbors"].to_numpy(dtype=np.int64, copy=False),
            out_df["computed_num_neighbors"].to_numpy(dtype=np.int64, copy=False),
        ), "Computed neighborhood sizes do not match input num_neighbors values"
    else:
        out_df["num_neighbors"] = out_df["computed_num_neighbors"]

    assert not (
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
    ), "Failed to compute neighborhood metrics for all rows"
    return out_df.drop(columns=["computed_num_neighbors"])


def build_summary(
    out_df: pd.DataFrame,
    metadata: dict[str, float | int | str | None],
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
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
    summary.update(metadata)
    return summary


def save_evaluation_outputs(
    out_df: pd.DataFrame,
    summary: dict[str, float | int | str | None],
    pred_path: str,
) -> str:
    out_dir = infer_results_dir_from_prediction_path(pred_path)
    os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(opj(out_dir, "out_df.csv"), index=False)
    out_df.to_json(opj(out_dir, "out_df.json"), orient="records", indent=2)
    pd.DataFrame([summary]).to_csv(opj(out_dir, "summary.csv"), index=False)
    with open(opj(out_dir, "summary.json"), "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2)

    return out_dir


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
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    neighborhoods_df = load_neighborhood_map(neighborhood_map_csv_path)
    pred = load_prediction(pred_path)
    emb_df = build_embedding_frame(pred)
    pair_df = explode_neighborhood_pairs(neighborhoods_df)
    pair_df = attach_pair_embeddings(pair_df, emb_df)
    pair_df = compute_pair_embedding_metrics(pair_df)
    out_df = summarize_neighborhood_metrics(neighborhoods_df, pair_df)
    summary = build_summary(
        out_df=out_df,
        metadata={
            "neighborhood_map_csv_path": neighborhood_map_csv_path,
            "pred_path": pred_path,
            "exp_name": exp_name,
            "name": name,
            "eval_key": eval_key,
            "celldist_mode": celldist_mode,
            "eval_key_digits": eval_key_digits,
            "dist_min": dist_min,
            "dist_max": dist_max,
            "dgt": dgt,
            "dle": dle,
            "nge": nge,
        },
    )
    return out_df, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/"
    ckpt = "training_124999"
    eval_key_prefix = "cellnbr"  #"cellnbrring"  # 
    default_pred_glob = f"*INF_srh7v1test_{eval_key_prefix}_*"

    neighborhood_map_csv_path_template = (
        "out/cellnbr_stats_nbr_8192_dgt{dist_min}_dle{dist_max}_nge1/"
        "sampled_neighborhood_map.csv"
    )

    run_sets = [
        # {"exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0"},
        # {"exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0"},
        # {"exp_name": "a2706135_dinov2"},
        # {"exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0"},
        # {"exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1"},
        # {"exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0"},
        # {"exp_name": "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0"},
        # {"exp_name": "326a6384_Apr10-15-07-23_sd1000_nomaskobw_lr14_tune0"},
        # {"exp_name": "10d41c43_Apr11-02-05-16_sd1000_nomaskobw_lr23_tune0"},
        # {"exp_name": "716f4772_Apr12-03-21-26_sd1000_maskobw_lr13_tune1"},
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

    for run in tqdm(runs, desc="Evaluating runs"):
        print(f"Evaluating {run['name']}")
        out_df, summary = evaluate_run(**run)
        save_evaluation_outputs(
            out_df=out_df, summary=summary, pred_path=run["pred_path"]
        )


if __name__ == "__main__":
    main()

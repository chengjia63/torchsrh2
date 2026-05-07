import argparse
import json
import logging
import os
import shutil
from os.path import join as opj

import pandas as pd
import torch
from tqdm.auto import tqdm


def _slide_key_from_embedding_path(path: str) -> str:
    slide_key = os.path.basename(os.path.dirname(path))
    if slide_key:
        return slide_key
    return os.path.splitext(os.path.basename(path))[0]


def _parse_global_cell_coords(cell_path: str) -> tuple[int, int]:
    patch_name, cell_coord = cell_path.split("#", 1)
    patch_coord = patch_name.rsplit("-", 1)[1]
    patch_top_str, patch_left_str = patch_coord.split("_", 1)
    cell_r_str, cell_c_str = cell_coord.split("_", 1)
    return (
        int(round(float(patch_top_str))) + int(round(float(cell_r_str))),
        int(round(float(patch_left_str))) + int(round(float(cell_c_str))),
    )


def _minmax_scores(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    min_value = values.min()
    max_value = values.max()
    span = max_value - min_value
    if span <= 0:
        return torch.zeros_like(values)
    return (values - min_value) / span


def _score_to_hex(score: float) -> str:
    color_stops = (
        (0.0, (26, 152, 80)),
        (0.32, (217, 239, 139)),
        (0.58, (253, 219, 199)),
        (0.78, (239, 138, 98)),
        (1.0, (178, 24, 43)),
    )
    clamped_score = min(max(float(score), 0.0), 1.0)
    left_stop = color_stops[0]
    right_stop = color_stops[-1]
    for stop in color_stops[1:]:
        if clamped_score <= stop[0]:
            right_stop = stop
            break
        left_stop = stop

    left_position, left_color = left_stop
    right_position, right_color = right_stop
    ratio = (
        0.0
        if right_position == left_position
        else (clamped_score - left_position) / (right_position - left_position)
    )
    rgb = [int(round(left + (right - left) * ratio)) for left, right in zip(left_color, right_color)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _load_embedding_cell_paths(embedding_path: str) -> list[str]:
    data = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if "path" not in data:
        raise KeyError(f"Embedding file is missing required `path`: {embedding_path}")
    return [str(path) for path in data["path"]]


def _build_cell_paths_by_slide(embedding_paths: list[str]) -> dict[str, list[str]]:
    cell_paths_by_slide: dict[str, list[str]] = {}
    for embedding_path in embedding_paths:
        slide_key = _slide_key_from_embedding_path(embedding_path)
        if slide_key in cell_paths_by_slide:
            raise ValueError(f"Duplicate slide key in predictions: {slide_key}")
        cell_paths_by_slide[slide_key] = _load_embedding_cell_paths(embedding_path)
    return cell_paths_by_slide


def _build_portal_cells_payload(
    *,
    cell_paths: list[str],
    attention: torch.Tensor,
    cell_score: torch.Tensor,
    cluster: torch.Tensor,
    cluster_contribution: torch.Tensor,
) -> tuple[dict, dict]:
    if len(cell_paths) != int(attention.shape[0]):
        raise ValueError(
            f"Cell path count does not match attention length: "
            f"{len(cell_paths)} paths vs {int(attention.shape[0])} attention values"
        )
    if len(cell_paths) != int(cell_score.shape[0]):
        raise ValueError(
            f"Cell path count does not match cell score length: "
            f"{len(cell_paths)} paths vs {int(cell_score.shape[0])} cell scores"
        )
    if len(cell_paths) != int(cluster.shape[0]):
        raise ValueError(
            f"Cell path count does not match cluster length: "
            f"{len(cell_paths)} paths vs {int(cluster.shape[0])} cluster values"
        )
    if len(cell_paths) != int(cluster_contribution.shape[0]):
        raise ValueError(
            f"Cell path count does not match cluster contribution length: "
            f"{len(cell_paths)} paths vs {int(cluster_contribution.shape[0])} cluster contribution values"
        )

    coords = [_parse_global_cell_coords(path) for path in cell_paths]
    y = [coord[0] for coord in coords]
    x = [coord[1] for coord in coords]
    normalized_attention = _minmax_scores(attention)
    attention_values = attention.float()
    cell_score_values = cell_score.float()
    display_cell_score = torch.sigmoid(cell_score_values)
    display_cell_score_values = display_cell_score.tolist()
    cell_score_display = [
        int(round(min(max(float(score), 0.0), 1.0) * 99.0))
        for score in display_cell_score_values
    ]
    cluster_values = cluster.long().tolist()
    cluster_contribution_display = torch.clamp(
        torch.floor(cluster_contribution.float() * 100.0),
        min=0,
        max=99,
    ).long().tolist()

    cell_count = len(cell_paths)
    num_clusters = max(cluster_values) + 1 if cluster_values else 1
    image_width = max(x) + 1 if x else 1
    image_height = max(y) + 1 if y else 1
    mean_cell_score = float(cell_score_values.mean()) if x else 0.0
    mean_attention_score = float(normalized_attention.mean()) if x else 0.0
    mean_raw_attention = float(attention_values.mean()) if x else 0.0

    cells_payload = {
        "cell_count": cell_count,
        "num_clusters": num_clusters,
        "x": x,
        "y": y,
        "normal_score": [round(1.0 - float(score), 6) for score in display_cell_score_values],
        "tumor_score": [round(float(score), 6) for score in display_cell_score_values],
        "tumor_score_display": cell_score_display,
        "dominant_cluster": cluster_values,
        "dominant_cluster_display": cluster_contribution_display,
        "detection_score": [1.0 for _ in range(cell_count)],
        "cell_type": ["attention" for _ in range(cell_count)],
        "dot_color": [_score_to_hex(float(score)) for score in display_cell_score_values],
        "cell_score": [round(float(score), 6) for score in cell_score_values],
        "attn_score": [round(float(score), 6) for score in normalized_attention],
        "raw_attn_score": [round(float(score), 10) for score in attention_values],
    }

    hard_slide_score = (
        sum(score > 0.5 for score in display_cell_score_values) / cell_count if cell_count else 0.0
    )
    soft_slide_score = sum(display_cell_score_values) / cell_count if cell_count else 0.0
    slide_statistics = {
        "cell_count": cell_count,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "tumor_probability_threshold": 0.5,
        "hard_slide_tumor_probability": round(hard_slide_score, 6),
        "percent_cells_tumor_probability_gt_0_5": round(hard_slide_score * 100.0, 6),
        "soft_slide_tumor_probability": round(soft_slide_score, 6),
        "area_soft_slide_tumor_probability": "NA",
        "area_pixel_count": int(image_width * image_height),
        "mean_cell_score": round(mean_cell_score, 6),
        "mean_attn_score": round(mean_attention_score, 6),
        "mean_raw_attn_score": round(mean_raw_attention, 10),
    }
    return cells_payload, slide_statistics


def _save_slide_predictions_json(rows, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = opj(output_dir, "slide_predictions.json")
    pd.DataFrame(rows).to_json(output_path, orient="records", indent=2)
    logging.info("Saved %d slide predictions to %s", len(rows), output_path)


def _save_portal_predictions(
    rows: list[dict],
    full: dict,
    cell_paths_by_slide: dict[str, list[str]],
    output_dir: str,
    exp_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if len(rows) != len(full["attention"]):
        raise ValueError(
            f"Prediction row count does not match attention count: "
            f"{len(rows)} rows vs {len(full['attention'])} attention tensors"
        )
    if len(rows) != len(full["cell_score"]):
        raise ValueError(
            f"Prediction row count does not match cell score count: "
            f"{len(rows)} rows vs {len(full['cell_score'])} cell score tensors"
        )
    if len(rows) != len(full["cluster"]):
        raise ValueError(
            f"Prediction row count does not match cluster count: "
            f"{len(rows)} rows vs {len(full['cluster'])} cluster tensors"
        )
    if len(rows) != len(full["cluster_contribution"]):
        raise ValueError(
            f"Prediction row count does not match cluster contribution count: "
            f"{len(rows)} rows vs {len(full['cluster_contribution'])} cluster contribution tensors"
        )

    saved_count = 0
    for row, attention, cell_score, cluster, cluster_contribution in tqdm(
        zip(
            rows,
            full["attention"],
            full["cell_score"],
            full["cluster"],
            full["cluster_contribution"],
        ),
        total=len(rows),
        desc="Saving slide portals",
        unit="slide",
        dynamic_ncols=True,
    ):
        slide_key = row["path"]
        if slide_key not in cell_paths_by_slide:
            raise KeyError(f"Missing cell paths for slide {slide_key}")
        cells_payload, slide_statistics = _build_portal_cells_payload(
            cell_paths=cell_paths_by_slide[slide_key],
            attention=attention,
            cell_score=cell_score,
            cluster=cluster,
            cluster_contribution=cluster_contribution,
        )

        portal_dir = opj(output_dir, slide_key, "portal")
        os.makedirs(portal_dir, exist_ok=True)

        cells_path = opj(portal_dir, "cells.json")
        with open(cells_path, "w", encoding="utf-8") as fd:
            json.dump(cells_payload, fd, separators=(",", ":"))

        manifest = {
            "slide_id": slide_key,
            "image_width": slide_statistics["image_width"],
            "image_height": slide_statistics["image_height"],
            "cell_count": cells_payload["cell_count"],
            "num_clusters": cells_payload["num_clusters"],
            "tile_size": 256,
            "overlap": 0,
            "cells": {
                "path": os.path.basename(cells_path),
                "score_label": "Cell score",
                "dot_color_label": "Low-green to high-red cell score",
            },
            "slide_statistics": {
                **slide_statistics,
                "label": row["label"],
                "pred_label": row["pred_label"],
                "raw_score": row["raw_score"],
            },
        }
        with open(opj(portal_dir, "slide_manifest.json"), "w", encoding="utf-8") as fd:
            json.dump(manifest, fd, indent=2, sort_keys=True)

        saved_count += 1

    logging.info("Saved %d slide portal directories under %s", saved_count, output_dir)
    print(f"Portal saved to: {output_dir}")
    print(f"Softlink command: ln -s {output_dir} {exp_name}")


def _exp_name_from_eval_dir(eval_output_dir: str) -> str:
    # eval_output_dir is {train_dir}/evals/{eval_name}; grandparent basename is the run name
    train_dir = os.path.dirname(os.path.dirname(os.path.abspath(eval_output_dir)))
    return os.path.basename(train_dir)


def generate_portal(eval_output_dir: str, site_res_dir: str) -> None:
    eval_output_dir = os.path.abspath(eval_output_dir)
    exp_name = _exp_name_from_eval_dir(eval_output_dir)
    portal_output_dir = opj(site_res_dir, "debug_out", exp_name)
    prediction_metrics_dir = opj(site_res_dir, "prediction_metrics")

    logging.info("exp_name: %s", exp_name)
    logging.info("portal_output_dir: %s", portal_output_dir)

    rows = pd.read_csv(opj(eval_output_dir, "slide_predictions.csv")).to_dict("records")
    full = torch.load(
        opj(eval_output_dir, "slide_tensors.pt"), map_location="cpu", weights_only=False
    )

    os.makedirs(prediction_metrics_dir, exist_ok=True)
    for src_name, dst_name in (
        ("gt_vs_ours_raw_score.json", f"{exp_name}_fg_score_charts.json"),
        ("metrics.json", f"{exp_name}_metrics.json"),
    ):
        shutil.copy2(opj(eval_output_dir, src_name), opj(prediction_metrics_dir, dst_name))
        logging.info("Copied %s to %s", src_name, prediction_metrics_dir)

    _save_slide_predictions_json(rows, output_dir=portal_output_dir)
    _save_portal_predictions(
        rows,
        full,
        cell_paths_by_slide=_build_cell_paths_by_slide(full["path"]),
        output_dir=portal_output_dir,
        exp_name=exp_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate portal assets from a saved eval dir")
    parser.add_argument("eval_output_dir", help="Output directory produced by eval.py")
    parser.add_argument("site_res_dir", help="Site resources root; portal and metrics go into subfolders here")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s|%(asctime)s|%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    generate_portal(args.eval_output_dir, args.site_res_dir)


if __name__ == "__main__":
    main()

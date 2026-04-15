import json
import logging
import os
from os.path import join as opj

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch

from ts2.utils.silica_scnn.eval_patch_proposal_neighborhood_embeddings import (
    attach_pair_embeddings,
    build_embedding_frame,
    build_runs_from_sets,
    compute_pair_embedding_metrics,
    extract_exp_key,
    explode_neighborhood_pairs,
    load_neighborhood_map,
    resolve_neighborhood_map_csv_path,
)
from ts2.utils.silica_scnn.get_cell_neighborhood_by_distances import (
    build_inference_dataset,
    build_dataset_path_to_index,
)
from ts2.utils.silica_sc_cls.eval_cell_inference_knn import load_prediction
from ts2.utils.srh_viz import (
    prepare_three_channel_viz_image,
    prepare_two_channel_viz_image,
)

mpl.rcParams["svg.fonttype"] = "none"


def save_figure_png_and_svg(fig: plt.Figure, png_path: str) -> None:
    if not png_path.endswith(".png"):
        raise ValueError(f"Expected png_path to end with .png, got {png_path}")
    svg_path = png_path[:-4] + ".svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def build_comparison_suffix(
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
) -> str:
    return (
        f"{extract_exp_key(baseline_run['exp_name'])}_vs_"
        f"{extract_exp_key(ours_run['exp_name'])}"
    )


def evaluate_pair_and_neighborhood_metrics(
    neighborhood_map_csv_path: str,
    pred_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neighborhoods_df = load_neighborhood_map(neighborhood_map_csv_path)
    pred = load_prediction(pred_path)
    emb_df = build_embedding_frame(pred)
    pair_df = explode_neighborhood_pairs(neighborhoods_df)
    pair_df = attach_pair_embeddings(pair_df, emb_df)
    pair_metrics_df = compute_pair_embedding_metrics(pair_df)

    neighborhood_metrics_df = (
        pair_metrics_df.groupby(["neighborhood_index", "anchor_path"], sort=True)
        .agg(
            num_neighbors=("neighbor_path", "size"),
            mean_neighbor_pixel_distance=("pixel_distance", "mean"),
            mean_neighbor_embedding_cosine_distance=(
                "embedding_cosine_distance",
                "mean",
            ),
        )
        .reset_index()
    )
    return pair_metrics_df, neighborhood_metrics_df


def build_path_to_embedding(pred_path: str) -> dict[str, np.ndarray]:
    pred = load_prediction(pred_path)
    emb_df = build_embedding_frame(pred)
    path_to_embedding = {
        str(path): np.asarray(embedding, dtype=np.float32)
        for path, embedding in zip(
            emb_df["path"].tolist(),
            emb_df["embedding"].tolist(),
            strict=True,
        )
    }
    if not path_to_embedding:
        raise ValueError(f"No embeddings found in prediction file: {pred_path}")
    return path_to_embedding


def compute_mean_all_pairs_cosine_distance(
    member_paths: list[str],
    path_to_embedding: dict[str, np.ndarray],
) -> float:
    if len(member_paths) < 2:
        raise ValueError(
            f"Expected at least 2 members to compute all-pairs distance, got {len(member_paths)}"
        )

    missing_paths = [path for path in member_paths if path not in path_to_embedding]
    if missing_paths:
        raise KeyError(
            "Could not find embeddings for neighborhood member paths: "
            + ", ".join(missing_paths[:10])
        )

    emb = np.stack([path_to_embedding[path] for path in member_paths]).astype(
        np.float32, copy=False
    )
    norms = np.linalg.norm(emb, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Found non-positive embedding norm while computing all-pairs cosine")
    cosine_similarity = (emb @ emb.T) / np.outer(norms, norms)
    cosine_distance = 1.0 - cosine_similarity
    upper_idx = np.triu_indices(len(member_paths), k=1)
    pair_values = cosine_distance[upper_idx]
    if pair_values.size == 0:
        raise ValueError("No upper-triangle pair values found for all-pairs cosine distance")
    return float(pair_values.mean())


def compute_ordered_pair_cosine_distances(
    member_paths: list[str],
    path_to_embedding: dict[str, np.ndarray],
) -> np.ndarray:
    if len(member_paths) < 2:
        raise ValueError(
            f"Expected at least 2 members to compute ordered pair distances, got {len(member_paths)}"
        )

    missing_paths = [path for path in member_paths if path not in path_to_embedding]
    if missing_paths:
        raise KeyError(
            "Could not find embeddings for paths: " + ", ".join(missing_paths[:10])
        )

    emb = np.stack([path_to_embedding[path] for path in member_paths]).astype(
        np.float32, copy=False
    )
    norms = np.linalg.norm(emb, axis=1)
    if np.any(norms <= 0):
        raise ValueError(
            "Found non-positive embedding norm while computing ordered-pair cosine"
        )
    cosine_similarity = (emb @ emb.T) / np.outer(norms, norms)
    cosine_distance = 1.0 - cosine_similarity
    non_diag_mask = ~np.eye(len(member_paths), dtype=bool)
    pair_values = cosine_distance[non_diag_mask]
    if pair_values.size == 0:
        raise ValueError("No ordered pair values found after removing diagonal")
    return pair_values.astype(np.float32, copy=False)


def select_single_run(
    runs: list[dict[str, str]],
    exp_name: str,
    selected_eval_key: str | None,
    selected_neighborhood_map_csv_path: str | None,
) -> dict[str, str]:
    matching_runs = [run for run in runs if run["exp_name"] == exp_name]
    assert matching_runs, f"No runs found for exp_name={exp_name}"

    if selected_eval_key is not None:
        matching_runs = [
            run for run in matching_runs if run["eval_key"] == selected_eval_key
        ]
        assert (
            matching_runs
        ), f"No runs found for exp_name={exp_name} and eval_key={selected_eval_key}"

    if selected_neighborhood_map_csv_path is not None:
        matching_runs = [
            run
            for run in matching_runs
            if run["neighborhood_map_csv_path"] == selected_neighborhood_map_csv_path
        ]
        assert matching_runs, (
            "No runs found for "
            f"exp_name={exp_name} and "
            f"neighborhood_map_csv_path={selected_neighborhood_map_csv_path}"
        )

    if len(matching_runs) != 1:
        available_eval_keys = sorted({run["eval_key"] for run in matching_runs})
        available_map_paths = sorted(
            {run["neighborhood_map_csv_path"] for run in matching_runs}
        )
        raise ValueError(
            "Expected exactly one matching run after filtering, found "
            f"{len(matching_runs)} for exp_name={exp_name}. "
            f"Set selected_eval_key or selected_neighborhood_map_csv_path. "
            f"Available eval_keys={available_eval_keys}. "
            f"Available neighborhood_map_csv_path values={available_map_paths}"
        )
    return matching_runs[0]


def build_topk_difference_tables(
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
    top_k: int,
    min_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        baseline_run["neighborhood_map_csv_path"]
        != ours_run["neighborhood_map_csv_path"]
    ):
        raise ValueError(
            "Baseline and ours must use the same neighborhood_map_csv_path, got "
            f"{baseline_run['neighborhood_map_csv_path']} vs "
            f"{ours_run['neighborhood_map_csv_path']}"
        )
    if baseline_run["eval_key"] != ours_run["eval_key"]:
        raise ValueError(
            "Baseline and ours must use the same eval_key, got "
            f"{baseline_run['eval_key']} vs {ours_run['eval_key']}"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")

    baseline_pair_df, baseline_nbr_df = evaluate_pair_and_neighborhood_metrics(
        neighborhood_map_csv_path=baseline_run["neighborhood_map_csv_path"],
        pred_path=baseline_run["pred_path"],
    )
    ours_pair_df, ours_nbr_df = evaluate_pair_and_neighborhood_metrics(
        neighborhood_map_csv_path=ours_run["neighborhood_map_csv_path"],
        pred_path=ours_run["pred_path"],
    )

    baseline_nbr_df = baseline_nbr_df.rename(
        columns={
            "num_neighbors": "baseline_num_neighbors",
            "mean_neighbor_pixel_distance": "baseline_mean_neighbor_pixel_distance",
            "mean_neighbor_embedding_cosine_distance": "baseline_mean_neighbor_embedding_cosine_distance",
        }
    )
    ours_nbr_df = ours_nbr_df.rename(
        columns={
            "num_neighbors": "ours_num_neighbors",
            "mean_neighbor_pixel_distance": "ours_mean_neighbor_pixel_distance",
            "mean_neighbor_embedding_cosine_distance": "ours_mean_neighbor_embedding_cosine_distance",
        }
    )

    comparison_df = baseline_nbr_df.merge(
        ours_nbr_df,
        on=["neighborhood_index", "anchor_path"],
        how="inner",
        validate="one_to_one",
    )
    assert len(comparison_df) == len(baseline_nbr_df) == len(ours_nbr_df), (
        "Neighborhood comparison lost rows: "
        f"baseline={len(baseline_nbr_df)}, ours={len(ours_nbr_df)}, "
        f"merged={len(comparison_df)}"
    )
    assert np.array_equal(
        comparison_df["baseline_num_neighbors"].to_numpy(dtype=np.int64, copy=False),
        comparison_df["ours_num_neighbors"].to_numpy(dtype=np.int64, copy=False),
    ), "Baseline and ours produced different neighbor counts for the same neighborhood"
    comparison_df["num_neighbors"] = comparison_df["baseline_num_neighbors"]
    assert np.allclose(
        comparison_df["baseline_mean_neighbor_pixel_distance"].to_numpy(
            dtype=np.float64, copy=False
        ),
        comparison_df["ours_mean_neighbor_pixel_distance"].to_numpy(
            dtype=np.float64, copy=False
        ),
        atol=0.0,
        rtol=0.0,
    ), "Baseline and ours produced different pixel-distance neighborhoods"

    comparison_df["delta_mean_neighbor_embedding_cosine_distance"] = (
        comparison_df["ours_mean_neighbor_embedding_cosine_distance"]
        - comparison_df["baseline_mean_neighbor_embedding_cosine_distance"]
    )
    comparison_df["baseline_exp_name"] = baseline_run["exp_name"]
    comparison_df["ours_exp_name"] = ours_run["exp_name"]
    comparison_df["eval_key"] = baseline_run["eval_key"]
    comparison_df["neighborhood_map_csv_path"] = baseline_run[
        "neighborhood_map_csv_path"
    ]
    comparison_df = comparison_df[
        comparison_df["num_neighbors"] >= min_neighbors
    ].copy()
    if comparison_df.empty:
        raise ValueError(
            "No neighborhoods remain after applying min_neighbors filter: "
            f"min_neighbors={min_neighbors}"
        )

    topk_df = (
        comparison_df.sort_values(
            by=[
                "delta_mean_neighbor_embedding_cosine_distance",
                "ours_mean_neighbor_embedding_cosine_distance",
                "baseline_mean_neighbor_embedding_cosine_distance",
            ],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(top_k)
        .reset_index(drop=True)
    )
    top_indices = topk_df["neighborhood_index"].tolist()

    baseline_pair_df = baseline_pair_df.rename(
        columns={
            "pixel_distance": "baseline_pixel_distance",
            "anchor_pred_label": "baseline_anchor_pred_label",
            "neighbor_pred_label": "baseline_neighbor_pred_label",
            "embedding_cosine_distance": "baseline_embedding_cosine_distance",
        }
    )
    ours_pair_df = ours_pair_df.rename(
        columns={
            "pixel_distance": "ours_pixel_distance",
            "anchor_pred_label": "ours_anchor_pred_label",
            "neighbor_pred_label": "ours_neighbor_pred_label",
            "embedding_cosine_distance": "ours_embedding_cosine_distance",
        }
    )
    pair_delta_df = baseline_pair_df.merge(
        ours_pair_df,
        on=["neighborhood_index", "anchor_path", "neighbor_path"],
        how="inner",
        validate="one_to_one",
    )
    assert len(pair_delta_df) == len(baseline_pair_df) == len(ours_pair_df), (
        "Pairwise comparison lost rows: "
        f"baseline={len(baseline_pair_df)}, ours={len(ours_pair_df)}, "
        f"merged={len(pair_delta_df)}"
    )
    assert np.allclose(
        pair_delta_df["baseline_pixel_distance"].to_numpy(dtype=np.float64, copy=False),
        pair_delta_df["ours_pixel_distance"].to_numpy(dtype=np.float64, copy=False),
        atol=0.0,
        rtol=0.0,
    ), "Baseline and ours produced different pixel distances for the same anchor-neighbor pairs"
    pair_delta_df["delta_embedding_cosine_distance"] = (
        pair_delta_df["ours_embedding_cosine_distance"]
        - pair_delta_df["baseline_embedding_cosine_distance"]
    )

    topk_pair_df = (
        topk_df[
            [
                "neighborhood_index",
                "anchor_path",
                "num_neighbors",
                "baseline_mean_neighbor_embedding_cosine_distance",
                "ours_mean_neighbor_embedding_cosine_distance",
                "delta_mean_neighbor_embedding_cosine_distance",
            ]
        ]
        .merge(
            pair_delta_df[pair_delta_df["neighborhood_index"].isin(top_indices)],
            on=["neighborhood_index", "anchor_path"],
            how="left",
            validate="one_to_many",
        )
        .sort_values(
            by=[
                "delta_mean_neighbor_embedding_cosine_distance",
                "neighborhood_index",
                "delta_embedding_cosine_distance",
                "neighbor_path",
            ],
            ascending=[False, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return topk_df, topk_pair_df


def infer_neighborhood_instances_csv_path(neighborhood_map_csv_path: str) -> str:
    map_dir = os.path.dirname(neighborhood_map_csv_path)
    instances_csv_path = opj(map_dir, "sampled_neighborhood_instances.csv")
    if not os.path.isfile(instances_csv_path):
        raise FileNotFoundError(
            "Expected sibling sampled_neighborhood_instances.csv next to "
            f"{neighborhood_map_csv_path}, but did not find {instances_csv_path}"
        )
    return instances_csv_path


def build_topk_neighborhood_rows(
    neighborhood_map_csv_path: str,
    topk_df: pd.DataFrame,
) -> pd.DataFrame:
    neighborhoods_df = load_neighborhood_map(neighborhood_map_csv_path).reset_index(
        drop=True
    )
    if "neighborhood_index" not in neighborhoods_df.columns:
        neighborhoods_df["neighborhood_index"] = np.arange(
            len(neighborhoods_df), dtype=np.int64
        )

    selected_df = neighborhoods_df.merge(
        topk_df[
            [
                "neighborhood_index",
                "anchor_path",
                "delta_mean_neighbor_embedding_cosine_distance",
                "baseline_mean_neighbor_embedding_cosine_distance",
                "ours_mean_neighbor_embedding_cosine_distance",
            ]
        ],
        on=["neighborhood_index", "anchor_path"],
        how="inner",
        validate="one_to_one",
    )
    assert len(selected_df) == len(topk_df), (
        "Failed to recover all top-k neighborhoods from the neighborhood map CSV: "
        f"topk={len(topk_df)}, selected={len(selected_df)}"
    )
    return selected_df.sort_values(
        by="delta_mean_neighbor_embedding_cosine_distance",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def attach_ordered_neighbor_cosine_distances(
    neighborhoods_df: pd.DataFrame,
    topk_pair_df: pd.DataFrame,
) -> pd.DataFrame:
    pair_lookup = topk_pair_df.set_index(
        ["neighborhood_index", "anchor_path", "neighbor_path"], drop=False
    )
    rows: list[dict[str, object]] = []
    for _, row in neighborhoods_df.iterrows():
        neighbor_paths = json.loads(row["neighbor_paths_json"])
        baseline_distances: list[float] = []
        ours_distances: list[float] = []
        for neighbor_path in neighbor_paths:
            key = (row["neighborhood_index"], row["anchor_path"], neighbor_path)
            if key not in pair_lookup.index:
                raise KeyError(
                    "Missing pairwise cosine distance for "
                    f"neighborhood_index={row['neighborhood_index']}, "
                    f"anchor_path={row['anchor_path']}, neighbor_path={neighbor_path}"
                )
            pair_row = pair_lookup.loc[key]
            if isinstance(pair_row, pd.DataFrame):
                raise ValueError(
                    "Found duplicate pair rows for "
                    f"neighborhood_index={row['neighborhood_index']}, "
                    f"anchor_path={row['anchor_path']}, neighbor_path={neighbor_path}"
                )
            baseline_distances.append(
                float(pair_row["baseline_embedding_cosine_distance"])
            )
            ours_distances.append(float(pair_row["ours_embedding_cosine_distance"]))

        row_dict = row.to_dict()
        row_dict["baseline_neighbor_embedding_cosine_distances_json"] = json.dumps(
            baseline_distances, separators=(",", ":")
        )
        row_dict["ours_neighbor_embedding_cosine_distances_json"] = json.dumps(
            ours_distances, separators=(",", ":")
        )
        rows.append(row_dict)

    out_df = pd.DataFrame(rows)
    assert len(out_df) == len(neighborhoods_df), (
        "Failed to attach ordered neighbor cosine distances to all neighborhoods: "
        f"input={len(neighborhoods_df)}, output={len(out_df)}"
    )
    return out_df


def sanitize_path_for_filename(cell_path: str) -> str:
    safe = cell_path.replace("/", "_").replace("#", "__").replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe


def center_crop_or_pad_image(
    image_np: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if target_height < 1 or target_width < 1:
        raise ValueError(
            "target_height and target_width must be >= 1, got "
            f"{target_height}, {target_width}"
        )

    src_height, src_width = image_np.shape[:2]
    if src_height > target_height:
        top = (src_height - target_height) // 2
        image_np = image_np[top : top + target_height, :, :]
        src_height = image_np.shape[0]
    if src_width > target_width:
        left = (src_width - target_width) // 2
        image_np = image_np[:, left : left + target_width, :]
        src_width = image_np.shape[1]

    out = np.zeros(
        (target_height, target_width, image_np.shape[2]), dtype=image_np.dtype
    )
    top = (target_height - src_height) // 2
    left = (target_width - src_width) // 2
    out[top : top + src_height, left : left + src_width, :] = image_np
    return out


def read_anchor_field_of_view_image(
    dataset,
    path_to_index: dict[str, int],
    anchor_path: str,
    anchor_crop_size: int,
) -> np.ndarray:
    if anchor_path not in path_to_index:
        raise KeyError(f"Anchor path not found in dataset: {anchor_path}")
    if anchor_crop_size < 1:
        raise ValueError(f"anchor_crop_size must be >= 1, got {anchor_crop_size}")

    inst = dataset.instances_.iloc[path_to_index[anchor_path]]
    if "proposal" not in inst:
        raise KeyError(f"Dataset instance is missing proposal for {anchor_path}")
    if "tensor_shape" not in inst:
        raise KeyError(f"Dataset instance is missing tensor_shape for {anchor_path}")
    if "mmap_idx" not in inst:
        raise KeyError(f"Dataset instance is missing mmap_idx for {anchor_path}")
    if (
        "institution" not in inst
        or dataset.nion_col not in inst
        or "mosaic" not in inst
    ):
        raise KeyError(
            f"Dataset instance is missing patch location fields for {anchor_path}"
        )

    proposal_row, proposal_col = int(inst["proposal"][0]), int(inst["proposal"][1])
    patch_mm_path = opj(
        dataset.data_root_,
        inst["institution"],
        inst[dataset.nion_col],
        inst["mosaic"],
        "patches",
        f"{inst[dataset.nion_col]}-{inst['mosaic']}-patches.dat",
    )
    tensor_shape = tuple(int(x) for x in inst["tensor_shape"])
    mmap_idx = int(inst["mmap_idx"])
    mm_dtype = getattr(dataset.process_read_im_, "dtype_", "uint16")

    if not os.path.isfile(patch_mm_path):
        raise FileNotFoundError(f"Patch memmap does not exist: {patch_mm_path}")

    fd = np.memmap(
        patch_mm_path,
        dtype=mm_dtype,
        mode="r",
        shape=tensor_shape,
    )
    patch_np = np.array(fd[[mmap_idx], ...])[0]
    fd._mmap.close()
    del fd

    patch_height, patch_width = patch_np.shape[:2]
    half = anchor_crop_size // 2
    row_start = proposal_row - half
    col_start = proposal_col - half
    row_end = row_start + anchor_crop_size
    col_end = col_start + anchor_crop_size

    src_row_start = max(row_start, 0)
    src_col_start = max(col_start, 0)
    src_row_end = min(row_end, patch_height)
    src_col_end = min(col_end, patch_width)

    crop_np = np.zeros(
        (anchor_crop_size, anchor_crop_size, patch_np.shape[2]), dtype=patch_np.dtype
    )
    dst_row_start = src_row_start - row_start
    dst_col_start = src_col_start - col_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_end = dst_col_start + (src_col_end - src_col_start)
    crop_np[
        dst_row_start:dst_row_end,
        dst_col_start:dst_col_end,
        :,
    ] = patch_np[src_row_start:src_row_end, src_col_start:src_col_end, :]

    return prepare_two_channel_viz_image(torch.from_numpy(crop_np)).numpy()


def save_single_neighborhood_image(
    dataset,
    path_to_index: dict[str, int],
    row: pd.Series,
    out_path: str,
    draw_center_x: bool,
    anchor_crop_size: int,
) -> None:
    if anchor_crop_size < 1:
        raise ValueError(f"anchor_crop_size must be >= 1, got {anchor_crop_size}")

    anchor_and_neighbors = [row["anchor_path"]] + json.loads(row["neighbor_paths_json"])
    neighbor_distances = json.loads(row["neighbor_distances_json"])
    baseline_neighbor_cosine_distances = json.loads(
        row["baseline_neighbor_embedding_cosine_distances_json"]
    )
    ours_neighbor_cosine_distances = json.loads(
        row["ours_neighbor_embedding_cosine_distances_json"]
    )
    if not (
        len(neighbor_distances)
        == len(baseline_neighbor_cosine_distances)
        == len(ours_neighbor_cosine_distances)
        == len(anchor_and_neighbors) - 1
    ):
        raise ValueError(
            "Neighbor metric lengths do not match for "
            f"anchor_path={row['anchor_path']}"
        )
    num_cols = len(anchor_and_neighbors)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_cols,
        figsize=(2.5 * num_cols, 2.5),
        squeeze=False,
    )

    for col_idx, cell_path in enumerate(anchor_and_neighbors):
        ax = axes[0, col_idx]
        if cell_path not in path_to_index:
            raise KeyError(f"Neighborhood path not found in dataset: {cell_path}")

        image = dataset[path_to_index[cell_path]]["image"].squeeze()
        image_np = prepare_three_channel_viz_image(image).numpy()
        if col_idx == 0:
            image_np = read_anchor_field_of_view_image(
                dataset=dataset,
                path_to_index=path_to_index,
                anchor_path=cell_path,
                anchor_crop_size=anchor_crop_size,
            )
        ax.imshow(image_np)
        if draw_center_x:
            center_row = (image_np.shape[0] - 1) / 2.0
            center_col = (image_np.shape[1] - 1) / 2.0
            arm = min(image_np.shape[0], image_np.shape[1]) * 0.12
            ax.plot(
                [center_col - arm, center_col + arm],
                [center_row - arm, center_row + arm],
                color="yellow",
                linewidth=2.0,
            )
            ax.plot(
                [center_col - arm, center_col + arm],
                [center_row + arm, center_row - arm],
                color="yellow",
                linewidth=2.0,
            )
        ax.axis("off")
        if col_idx == 0:
            ax.set_title(
                "anchor\n"
                f"n={row['num_neighbors']}\n"
                f"dinov2={row['baseline_mean_neighbor_embedding_cosine_distance']:.3f} "
                f"silica={row['ours_mean_neighbor_embedding_cosine_distance']:.3f}",
                fontsize=8,
            )
        else:
            ax.set_title(
                f"n{col_idx}\n"
                f"px={neighbor_distances[col_idx - 1]:.2f}\n"
                f"dinov2={baseline_neighbor_cosine_distances[col_idx - 1]:.3f} "
                f"silica={ours_neighbor_cosine_distances[col_idx - 1]:.3f}",
                fontsize=8,
            )

    fig.tight_layout()
    save_figure_png_and_svg(fig, out_path)
    plt.close(fig)


def save_topk_images(
    topk_df: pd.DataFrame,
    topk_pair_df: pd.DataFrame,
    neighborhood_map_csv_path: str,
    out_dir: str,
    anchor_crop_size: int,
) -> None:
    if topk_df.empty:
        raise ValueError("Cannot save images for an empty top-k dataframe")

    instances_csv_path = infer_neighborhood_instances_csv_path(
        neighborhood_map_csv_path
    )
    selected_neighborhoods_df = build_topk_neighborhood_rows(
        neighborhood_map_csv_path=neighborhood_map_csv_path,
        topk_df=topk_df,
    )
    selected_neighborhoods_df = attach_ordered_neighbor_cosine_distances(
        neighborhoods_df=selected_neighborhoods_df,
        topk_pair_df=topk_pair_df,
    )
    dataset = build_inference_dataset(cell_instances_path=instances_csv_path)
    path_to_index = build_dataset_path_to_index(dataset)
    image_out_dir = opj(out_dir, "topk_neighborhood_images")
    os.makedirs(image_out_dir, exist_ok=True)
    plain_dir = opj(image_out_dir, "plain")
    center_x_dir = opj(image_out_dir, "with_center_x")
    metrics_csv_path = opj(image_out_dir, "plot_order_neighbor_metrics.csv")
    os.makedirs(plain_dir, exist_ok=True)
    os.makedirs(center_x_dir, exist_ok=True)
    selected_neighborhoods_df.to_csv(metrics_csv_path, index=False)

    for rank, (_, row) in enumerate(selected_neighborhoods_df.iterrows(), start=1):
        base_name = (
            f"rank{rank:03d}_nbr{int(row['neighborhood_index']):05d}_"
            f"{sanitize_path_for_filename(str(row['anchor_path']))}"
        )
        plain_path = opj(plain_dir, f"{base_name}.png")
        center_x_path = opj(center_x_dir, f"{base_name}.png")
        save_single_neighborhood_image(
            dataset=dataset,
            path_to_index=path_to_index,
            row=row,
            out_path=plain_path,
            draw_center_x=False,
            anchor_crop_size=anchor_crop_size,
        )
        save_single_neighborhood_image(
            dataset=dataset,
            path_to_index=path_to_index,
            row=row,
            out_path=center_x_path,
            draw_center_x=True,
            anchor_crop_size=anchor_crop_size,
        )

    logging.info(
        "Saved %d neighborhood images to %s and %s",
        len(selected_neighborhoods_df),
        plain_dir,
        center_x_dir,
    )
    logging.info(
        "Saved plot-order neighbor cosine distances to %s",
        metrics_csv_path,
    )


def save_outputs(
    topk_df: pd.DataFrame,
    topk_pair_df: pd.DataFrame,
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    comparison_suffix = build_comparison_suffix(
        baseline_run=baseline_run,
        ours_run=ours_run,
    )
    file_stem = (
        f"topk_neighborhood_diff_{baseline_run['eval_key']}_{comparison_suffix}"
    )
    topk_csv_path = opj(out_dir, f"{file_stem}.csv")
    topk_pair_csv_path = opj(out_dir, f"{file_stem}_pairs.csv")
    meta_json_path = opj(out_dir, f"{file_stem}.json")

    topk_df.to_csv(topk_csv_path, index=False)
    topk_pair_df.to_csv(topk_pair_csv_path, index=False)
    with open(meta_json_path, "w", encoding="utf-8") as fd:
        json.dump(
            {
                "baseline_run": baseline_run["name"],
                "ours_run": ours_run["name"],
                "eval_key": baseline_run["eval_key"],
                "neighborhood_map_csv_path": baseline_run["neighborhood_map_csv_path"],
                "num_topk_neighborhoods": int(len(topk_df)),
                "top_anchor_paths": topk_df["anchor_path"].tolist(),
            },
            fd,
            indent=2,
        )

    logging.info("Saved neighborhood summary to %s", topk_csv_path)
    logging.info("Saved pair-level details to %s", topk_pair_csv_path)


def save_neighborhood_ranking_curve(
    topk_df: pd.DataFrame,
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
    out_dir: str,
) -> None:
    if topk_df.empty:
        raise ValueError("Cannot save a ranking curve for an empty top-k dataframe")

    rank_df = (
        topk_df.sort_values(
            by=[
                "delta_mean_neighbor_embedding_cosine_distance",
                "ours_mean_neighbor_embedding_cosine_distance",
                "baseline_mean_neighbor_embedding_cosine_distance",
                "neighborhood_index",
                "anchor_path",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=np.int64)
    rank_df["percentile"] = (
        rank_df["rank"].to_numpy(dtype=np.float64, copy=False) / len(rank_df) * 100.0
    )
    comparison_suffix = build_comparison_suffix(
        baseline_run=baseline_run,
        ours_run=ours_run,
    )
    rank_csv_path = opj(
        out_dir,
        (
            f"topk_neighborhood_anchor_pair_rank_curve_{baseline_run['eval_key']}_"
            f"{comparison_suffix}.csv"
        ),
    )
    rank_png_path = opj(
        out_dir,
        (
            f"topk_neighborhood_anchor_pair_rank_curve_delta_{baseline_run['eval_key']}_"
            f"{comparison_suffix}.png"
        ),
    )

    rank_df.to_csv(rank_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        rank_df["percentile"],
        rank_df["delta_mean_neighbor_embedding_cosine_distance"],
        color="tab:red",
        linewidth=2.5,
        label="silica - dinov2",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Neighborhood percentile")
    ax.set_ylabel("Delta mean anchor-pair embedding cosine distance")
    ax.set_title(
        "Neighborhood Anchor-Pair Delta Curve\n"
        f"{baseline_run['eval_key']} sorted by silica - dinov2, N={len(rank_df)}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure_png_and_svg(fig, rank_png_path)
    plt.close(fig)

    logging.info("Saved neighborhood delta ranking curve to %s", rank_png_path)
    logging.info("Saved neighborhood ranking curve data to %s", rank_csv_path)


def save_neighborhood_all_pairs_delta_curve(
    topk_df: pd.DataFrame,
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
    out_dir: str,
) -> None:
    if topk_df.empty:
        raise ValueError("Cannot save all-pairs neighborhood curve for an empty top-k dataframe")

    selected_neighborhoods_df = build_topk_neighborhood_rows(
        neighborhood_map_csv_path=baseline_run["neighborhood_map_csv_path"],
        topk_df=topk_df,
    )
    dinov2_path_to_embedding = build_path_to_embedding(baseline_run["pred_path"])
    silica_path_to_embedding = build_path_to_embedding(ours_run["pred_path"])

    rows: list[dict[str, object]] = []
    for _, row in selected_neighborhoods_df.iterrows():
        member_paths = [str(row["anchor_path"]), *json.loads(row["neighbor_paths_json"])]
        dinov2_mean = compute_mean_all_pairs_cosine_distance(
            member_paths=member_paths,
            path_to_embedding=dinov2_path_to_embedding,
        )
        silica_mean = compute_mean_all_pairs_cosine_distance(
            member_paths=member_paths,
            path_to_embedding=silica_path_to_embedding,
        )
        rows.append(
            {
                "neighborhood_index": int(row["neighborhood_index"]),
                "anchor_path": str(row["anchor_path"]),
                "num_neighbors": int(row["num_neighbors"]),
                "num_neighborhood_members": int(len(member_paths)),
                "dinov2_mean_all_pairs_embedding_cosine_distance": dinov2_mean,
                "silica_mean_all_pairs_embedding_cosine_distance": silica_mean,
                "delta_mean_all_pairs_embedding_cosine_distance": silica_mean - dinov2_mean,
            }
        )

    all_pairs_df = pd.DataFrame(rows).sort_values(
        by="delta_mean_all_pairs_embedding_cosine_distance",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    all_pairs_df["rank"] = np.arange(1, len(all_pairs_df) + 1, dtype=np.int64)
    all_pairs_df["percentile"] = (
        all_pairs_df["rank"].to_numpy(dtype=np.float64, copy=False)
        / len(all_pairs_df)
        * 100.0
    )
    comparison_suffix = build_comparison_suffix(
        baseline_run=baseline_run,
        ours_run=ours_run,
    )

    curve_csv_path = opj(
        out_dir,
        (
            f"topk_neighborhood_all_pair_rank_curve_{baseline_run['eval_key']}_"
            f"{comparison_suffix}.csv"
        ),
    )
    curve_png_path = opj(
        out_dir,
        (
            f"topk_neighborhood_all_pair_rank_curve_delta_{baseline_run['eval_key']}_"
            f"{comparison_suffix}.png"
        ),
    )
    all_pairs_df.to_csv(curve_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        all_pairs_df["percentile"],
        all_pairs_df["delta_mean_all_pairs_embedding_cosine_distance"],
        color="tab:red",
        linewidth=2.5,
        label="silica - dinov2",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Neighborhood percentile")
    ax.set_ylabel("Delta mean all-pair embedding cosine distance")
    ax.set_title(
        "Neighborhood All-Pair Delta Curve\n"
        f"{baseline_run['eval_key']} sorted by silica - dinov2, N={len(all_pairs_df)}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure_png_and_svg(fig, curve_png_path)
    plt.close(fig)

    logging.info("Saved neighborhood all-pairs delta curve to %s", curve_png_path)
    logging.info("Saved neighborhood all-pairs curve data to %s", curve_csv_path)


def save_pair_level_plots(
    topk_pair_df: pd.DataFrame,
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
    out_dir: str,
) -> None:
    if topk_pair_df.empty:
        raise ValueError("Cannot save pair-level plots for an empty pair dataframe")

    pair_df = (
        topk_pair_df.sort_values(
            by=[
                "delta_embedding_cosine_distance",
                "ours_embedding_cosine_distance",
                "baseline_embedding_cosine_distance",
                "neighborhood_index",
                "neighbor_path",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    comparison_suffix = build_comparison_suffix(
        baseline_run=baseline_run,
        ours_run=ours_run,
    )

    pair_curve_csv_path = opj(
        out_dir,
        (
            f"topk_pair_rank_curve_{baseline_run['eval_key']}_"
            f"{comparison_suffix}.csv"
        ),
    )

    pair_df.to_csv(pair_curve_csv_path, index=False)
    logging.info("Saved pair plot data to %s", pair_curve_csv_path)


def save_anchor_anchor_pair_curves(
    topk_df: pd.DataFrame,
    baseline_run: dict[str, str],
    ours_run: dict[str, str],
    out_dir: str,
) -> None:
    if topk_df.empty:
        raise ValueError("Cannot save anchor-anchor curves for an empty top-k dataframe")

    anchor_paths = [str(path) for path in topk_df["anchor_path"].tolist()]
    if len(set(anchor_paths)) != len(anchor_paths):
        raise ValueError("Expected unique anchor_path values in topk_df")
    if len(anchor_paths) < 2:
        raise ValueError(
            f"Need at least 2 anchors to compute anchor-anchor pairs, got {len(anchor_paths)}"
        )

    dinov2_path_to_embedding = build_path_to_embedding(baseline_run["pred_path"])
    silica_path_to_embedding = build_path_to_embedding(ours_run["pred_path"])
    dinov2_distances = np.sort(
        compute_ordered_pair_cosine_distances(
            member_paths=anchor_paths,
            path_to_embedding=dinov2_path_to_embedding,
        )
    )[::-1]
    silica_distances = np.sort(
        compute_ordered_pair_cosine_distances(
            member_paths=anchor_paths,
            path_to_embedding=silica_path_to_embedding,
        )
    )[::-1]
    assert len(dinov2_distances) == len(silica_distances), (
        "Expected same number of anchor-anchor ordered pairs for both models, got "
        f"{len(dinov2_distances)} vs {len(silica_distances)}"
    )

    percentile = np.arange(1, len(dinov2_distances) + 1, dtype=np.float64)
    percentile = percentile / len(dinov2_distances) * 100.0
    curve_df = pd.DataFrame(
        {
            "percentile": percentile,
            "dinov2_anchor_anchor_embedding_cosine_distance": dinov2_distances,
            "silica_anchor_anchor_embedding_cosine_distance": silica_distances,
        }
    )
    curve_df["dinov2_mean_anchor_anchor_embedding_cosine_distance"] = float(
        dinov2_distances.mean()
    )
    curve_df["silica_mean_anchor_anchor_embedding_cosine_distance"] = float(
        silica_distances.mean()
    )

    curve_csv_path = opj(
        out_dir,
        (
            f"topk_anchor_anchor_pair_percentile_curve_{baseline_run['eval_key']}_"
            f"{build_comparison_suffix(baseline_run=baseline_run, ours_run=ours_run)}.csv"
        ),
    )
    curve_png_path = opj(
        out_dir,
        (
            f"topk_anchor_anchor_pair_percentile_curve_{baseline_run['eval_key']}_"
            f"{build_comparison_suffix(baseline_run=baseline_run, ours_run=ours_run)}.png"
        ),
    )
    curve_df.to_csv(curve_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        curve_df["percentile"],
        curve_df["dinov2_anchor_anchor_embedding_cosine_distance"],
        color="tab:blue",
        linewidth=2.0,
        label="dinov2",
    )
    ax.plot(
        curve_df["percentile"],
        curve_df["silica_anchor_anchor_embedding_cosine_distance"],
        color="tab:orange",
        linewidth=2.0,
        label="silica",
    )
    ax.axhline(
        curve_df["dinov2_mean_anchor_anchor_embedding_cosine_distance"].iloc[0],
        color="tab:blue",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        label="dinov2 mean",
    )
    ax.axhline(
        curve_df["silica_mean_anchor_anchor_embedding_cosine_distance"].iloc[0],
        color="tab:orange",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        label="silica mean",
    )
    mean_x = 99.0
    dinov2_mean = float(
        curve_df["dinov2_mean_anchor_anchor_embedding_cosine_distance"].iloc[0]
    )
    silica_mean = float(
        curve_df["silica_mean_anchor_anchor_embedding_cosine_distance"].iloc[0]
    )
    ax.text(
        mean_x,
        dinov2_mean,
        f"dinov2 mean={dinov2_mean:.3f}",
        color="tab:blue",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.text(
        mean_x,
        silica_mean,
        f"silica mean={silica_mean:.3f}",
        color="tab:orange",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("Anchor-anchor ordered-pair percentile")
    ax.set_ylabel("Embedding cosine distance")
    ax.set_title(
        "Anchor-Anchor Ordered Pair Distance Curve\n"
        f"{baseline_run['eval_key']}, N_anchor={len(anchor_paths)}, N_pair={len(curve_df)}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure_png_and_svg(fig, curve_png_path)
    plt.close(fig)

    logging.info("Saved anchor-anchor pair curve to %s", curve_png_path)
    logging.info("Saved anchor-anchor pair curve data to %s", curve_csv_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/"
    ckpt = "training_124999"
    eval_key_prefix = "cellnbr"  #
    default_pred_glob = f"*INF_srh7v1test_{eval_key_prefix}_*"
    neighborhood_map_csv_path_template = (
        "out/cellnbr_stats_nbr_8192_dgt{dist_min}_dle{dist_max}_nge1/"
        "sampled_neighborhood_map.csv"
    )

    baseline_exp_name = "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0"
    ours_exp_name = "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0"
    dist_min = 12
    dist_max = 40
    min_neighbors = 3
    top_k = 128
    anchor_crop_size = 128
    out_dir = "topk_neighborhood/"

    selected_eval_key = f"{eval_key_prefix}_dgt{dist_min}_dle{dist_max}_nge1"
    selected_neighborhood_map_csv_path = resolve_neighborhood_map_csv_path(
        neighborhood_map_csv_path_template=neighborhood_map_csv_path_template,
        celldist_mode=eval_key_prefix,
        dist_min=dist_min,
        dist_max=dist_max,
    )

    run_sets = [
        {"exp_name": baseline_exp_name},
        {"exp_name": ours_exp_name},
    ]
    runs = build_runs_from_sets(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=run_sets,
        neighborhood_map_csv_path_template=neighborhood_map_csv_path_template,
        default_pred_glob=default_pred_glob,
        eval_key_prefix=eval_key_prefix,
    )

    baseline_run = select_single_run(
        runs=runs,
        exp_name=baseline_exp_name,
        selected_eval_key=selected_eval_key,
        selected_neighborhood_map_csv_path=selected_neighborhood_map_csv_path,
    )
    ours_run = select_single_run(
        runs=runs,
        exp_name=ours_exp_name,
        selected_eval_key=(
            baseline_run["eval_key"] if selected_eval_key is None else selected_eval_key
        ),
        selected_neighborhood_map_csv_path=baseline_run["neighborhood_map_csv_path"],
    )

    logging.info(
        "Comparing baseline=%s vs ours=%s for eval_key=%s",
        baseline_run["exp_name"],
        ours_run["exp_name"],
        baseline_run["eval_key"],
    )
    topk_df, topk_pair_df = build_topk_difference_tables(
        baseline_run=baseline_run,
        ours_run=ours_run,
        top_k=top_k,
        min_neighbors=min_neighbors,
    )
    save_outputs(
        topk_df=topk_df,
        topk_pair_df=topk_pair_df,
        baseline_run=baseline_run,
        ours_run=ours_run,
        out_dir=out_dir,
    )
    save_neighborhood_ranking_curve(
        topk_df=topk_df,
        baseline_run=baseline_run,
        ours_run=ours_run,
        out_dir=out_dir,
    )
    save_neighborhood_all_pairs_delta_curve(
        topk_df=topk_df,
        baseline_run=baseline_run,
        ours_run=ours_run,
        out_dir=out_dir,
    )
    save_pair_level_plots(
        topk_pair_df=topk_pair_df,
        baseline_run=baseline_run,
        ours_run=ours_run,
        out_dir=out_dir,
    )
    save_anchor_anchor_pair_curves(
        topk_df=topk_df,
        baseline_run=baseline_run,
        ours_run=ours_run,
        out_dir=out_dir,
    )
    save_topk_images(
        topk_df=topk_df,
        topk_pair_df=topk_pair_df,
        neighborhood_map_csv_path=baseline_run["neighborhood_map_csv_path"],
        out_dir=out_dir,
        anchor_crop_size=anchor_crop_size,
    )
    logging.info(
        "Top neighborhoods:\n%s",
        topk_df[
            [
                "neighborhood_index",
                "anchor_path",
                "num_neighbors",
                "delta_mean_neighbor_embedding_cosine_distance",
                "baseline_mean_neighbor_embedding_cosine_distance",
                "ours_mean_neighbor_embedding_cosine_distance",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()

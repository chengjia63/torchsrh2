import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from ts2.data.transforms import HistologyTransform
from ts2.train.main_cell_inference import SingleCellListInferenceDataset
from ts2.utils.silica_sc_cls.eval_cell_inference_knn import parse_tuple_string
from ts2.utils.srh_viz import prepare_three_channel_viz_image

dset_cf_yaml = """
data:
  xform_params:
    which_set: srh
    base_aug_params:
      laser_noise_config: null
      get_third_channel_params:
        mode: three_channels
        subtracted_base: 0.07629394531
      to_uint8: False
    strong_aug_params:
      aug_list:
      - which: center_crop_always_apply
        params:
          size: 48
      aug_prob: 1
  test_dataset:
    which: SingleCellListInferenceDataset
    params:
      data_root: /nfs/turbo/umms-tocho-snr/data/root_histology_db/srh/
      cell_instances: TO BE FILLED IN
"""


def format_value_for_path(value: float | int) -> str:
    if isinstance(value, int) or int(value) == value:
        return str(int(value))
    return str(value).replace(".", "p")


def build_job_out_dir(
    out_root: str,
    base_out_name: str,
    num_cells: int,
    distance_bounds: list[float | None] | tuple[float | None, float | None],
    min_neighbors: int,
) -> str:
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)
    folder_name = (
        f"{base_out_name}_nbr_{num_cells}_"
        f"dgt{format_value_for_path(distance_lower_bound)}_"
        f"dle{format_value_for_path(distance_upper_bound)}_"
        f"nge{format_value_for_path(min_neighbors)}"
    )
    return os.path.join(out_root, folder_name)


def parse_distance_bounds(
    distance_bounds: list[float | None] | tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    if len(distance_bounds) != 2:
        raise ValueError(
            f"distance_bounds must have exactly two elements [lower, upper], got {distance_bounds}"
        )
    return distance_bounds[0], distance_bounds[1]


def validate_job_args(
    distance_bounds: list[float | None] | tuple[float | None, float | None],
    min_neighbors: int,
    num_cells: int,
    random_seed: int,
) -> None:
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)
    if distance_lower_bound is not None and distance_lower_bound < 0:
        raise ValueError(
            f"distance_lower_bound must be >= 0, got {distance_lower_bound}"
        )
    if distance_upper_bound is not None and distance_upper_bound < 0:
        raise ValueError(
            f"distance_upper_bound must be >= 0, got {distance_upper_bound}"
        )
    if (
        distance_lower_bound is not None
        and distance_upper_bound is not None
        and distance_lower_bound > distance_upper_bound
    ):
        raise ValueError(
            "distance_lower_bound cannot be greater than distance_upper_bound: "
            f"{distance_lower_bound} > {distance_upper_bound}"
        )
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
    if num_cells < 1:
        raise ValueError(f"num_cells must be >= 1, got {num_cells}")
    if random_seed < 0:
        raise ValueError(f"random_seed must be >= 0, got {random_seed}")


def load_instances(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"patch", "proposal"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Input CSV is missing required columns: {sorted(missing_columns)}"
        )
    if df.empty:
        raise ValueError(f"Input CSV has no rows: {csv_path}")
    return df


def parse_proposals(df: pd.DataFrame) -> pd.DataFrame:
    coords: List[Tuple[int, int]] = []
    for proposal in tqdm(
        df["proposal"].tolist(),
        total=len(df),
        desc="Parsing proposals",
    ):
        coords.append(parse_tuple_string(str(proposal)))

    parsed = df.copy()
    parsed["proposal_row"] = [coord[0] for coord in coords]
    parsed["proposal_col"] = [coord[1] for coord in coords]
    return parsed


def drop_duplicate_proposals(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"patch", "proposal_row", "proposal_col"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Cannot deduplicate proposals without columns: {sorted(missing_columns)}"
        )

    deduped = df.drop_duplicates(
        subset=["patch", "proposal_row", "proposal_col"], keep="first"
    ).reset_index(drop=True)
    if deduped.empty:
        raise ValueError("No proposals remain after dropping duplicates")
    return deduped


def attach_paths(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"patch", "proposal_row", "proposal_col"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Cannot build paths without columns: {sorted(missing_columns)}")

    out = df.copy()
    out["path"] = [
        f"{patch}#{int(row)}_{int(col)}"
        for patch, row, col in zip(
            out["patch"], out["proposal_row"], out["proposal_col"]
        )
    ]
    return out


def prepare_instances(csv_path: str) -> tuple[pd.DataFrame, List[str]]:
    raw_df = load_instances(csv_path)
    original_columns = raw_df.columns.tolist()
    df = parse_proposals(raw_df)
    df = drop_duplicate_proposals(df)
    df = attach_paths(df)
    return df, original_columns


def collect_candidate_neighborhoods(
    df: pd.DataFrame,
    distance_bounds: list[float | None] | tuple[float | None, float | None],
    min_neighbors: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    total_cells = 0
    total_cells_with_any_neighbor = 0
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)

    grouped = df.groupby("patch", sort=True)
    for patch, group in tqdm(
        grouped,
        total=df["patch"].nunique(),
        desc="Collect neighborhoods",
    ):
        coords = group[["proposal_row", "proposal_col"]].to_numpy(dtype=np.float64)
        if coords.shape[0] == 0:
            continue

        proposals = [str(x) for x in group["proposal"].tolist()]
        paths = group["path"].tolist()
        total_cells += int(coords.shape[0])
        if coords.shape[0] == 1:
            continue

        deltas = coords[:, None, :] - coords[None, :, :]
        distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=-1, dtype=np.float64))
        within_threshold = np.ones(distance_matrix.shape, dtype=bool)
        if distance_lower_bound is not None:
            within_threshold &= distance_matrix > distance_lower_bound
        if distance_upper_bound is not None:
            within_threshold &= distance_matrix <= distance_upper_bound
        np.fill_diagonal(within_threshold, False)

        neighbor_counts = within_threshold.sum(axis=1)
        total_cells_with_any_neighbor += int((neighbor_counts > 0).sum())

        for anchor_idx in np.flatnonzero(neighbor_counts >= min_neighbors):
            neighbor_idx = np.flatnonzero(within_threshold[anchor_idx])
            neighbor_distances = distance_matrix[anchor_idx, neighbor_idx]
            sort_order = np.lexsort(
                (
                    coords[neighbor_idx, 1],
                    coords[neighbor_idx, 0],
                    neighbor_distances,
                )
            )
            neighbor_idx = neighbor_idx[sort_order]
            neighbor_distances = neighbor_distances[sort_order]

            rows.append(
                {
                    "patch": patch,
                    "anchor_path": paths[anchor_idx],
                    "anchor_proposal": proposals[anchor_idx],
                    "anchor_proposal_row": int(coords[anchor_idx, 0]),
                    "anchor_proposal_col": int(coords[anchor_idx, 1]),
                    "num_neighbors": int(neighbor_idx.size),
                    "neighbor_paths_json": json.dumps(
                        [paths[i] for i in neighbor_idx], separators=(",", ":")
                    ),
                    "neighbor_proposals_json": json.dumps(
                        [proposals[i] for i in neighbor_idx], separators=(",", ":")
                    ),
                    "neighbor_distances_json": json.dumps(
                        [float(x) for x in neighbor_distances], separators=(",", ":")
                    ),
                }
            )

    neighborhoods_df = pd.DataFrame(rows)
    if neighborhoods_df.empty:
        raise ValueError(
            "No cells remained after neighborhood filtering with "
            f"distance_bounds={distance_bounds}, min_neighbors={min_neighbors}"
        )

    stats = {
        "total_cells": int(total_cells),
        "total_cells_with_any_neighbor": int(total_cells_with_any_neighbor),
        "eligible_cells": int(len(neighborhoods_df)),
    }
    return neighborhoods_df, stats


def sample_neighborhoods(
    neighborhoods_df: pd.DataFrame,
    num_cells: int,
    random_seed: int,
) -> pd.DataFrame:
    if neighborhoods_df.empty:
        raise ValueError("No candidate neighborhoods were found in the dataset")

    sample_size = min(num_cells, len(neighborhoods_df))
    if sample_size == len(neighborhoods_df):
        return neighborhoods_df.reset_index(drop=True)

    return neighborhoods_df.sample(
        n=sample_size, replace=False, random_state=random_seed
    ).reset_index(drop=True)


def collect_neighborhood_instance_rows(
    df: pd.DataFrame,
    neighborhoods_df: pd.DataFrame,
    original_columns: List[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if "path" not in df.columns:
        raise KeyError("Expected `path` column in instance dataframe")
    if "anchor_path" not in neighborhoods_df.columns:
        raise KeyError("Expected `anchor_path` column in neighborhoods dataframe")

    selected_paths = list(
        dict.fromkeys(
            path
            for _, row in neighborhoods_df.iterrows()
            for path in [row["anchor_path"], *json.loads(row["neighbor_paths_json"])]
        )
    )
    if not selected_paths:
        raise ValueError("No instance rows matched the sampled neighborhoods")

    indexed_df = df.set_index("path", drop=False)
    missing_paths = sorted(set(selected_paths) - set(indexed_df.index))
    if missing_paths:
        raise ValueError(
            "Neighborhood paths were not found in the instance dataframe: "
            + ", ".join(missing_paths[:10])
        )

    selected = indexed_df.loc[selected_paths].copy().reset_index(drop=True)
    out_df = selected.loc[:, original_columns].reset_index(drop=True)
    path_to_instance_index = {path: idx for idx, path in enumerate(selected_paths)}
    return out_df, path_to_instance_index


def attach_instance_indices(
    neighborhoods_df: pd.DataFrame,
    path_to_instance_index: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for neighborhood_index, (_, row) in enumerate(neighborhoods_df.iterrows()):
        anchor_path = row["anchor_path"]
        neighbor_paths = json.loads(row["neighbor_paths_json"])
        neighbor_instance_indices = [
            path_to_instance_index[path] for path in neighbor_paths
        ]
        rows.append(
            {
                "neighborhood_index": int(neighborhood_index),
                **row.to_dict(),
                "anchor_instance_index": int(path_to_instance_index[anchor_path]),
                "neighbor_instance_indices_json": json.dumps(
                    neighbor_instance_indices, separators=(",", ":")
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("Neighborhood map is unexpectedly empty")
    return out_df


def build_filtering_summary(
    csv_path: str,
    distance_bounds: list[float | None] | tuple[float | None, float | None],
    min_neighbors: int,
    num_cells_requested: int,
    random_seed: int,
    total_cells: int,
    total_cells_with_any_neighbor: int,
    num_candidate_neighborhoods: int,
    num_sampled_neighborhoods: int,
    num_sampled_instances: int,
) -> dict[str, float | int | str]:
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)
    return {
        "csv_path": csv_path,
        "distance_lower_bound": (
            None if distance_lower_bound is None else float(distance_lower_bound)
        ),
        "distance_upper_bound": (
            None if distance_upper_bound is None else float(distance_upper_bound)
        ),
        "min_neighbors": int(min_neighbors),
        "num_cells_requested": int(num_cells_requested),
        "random_seed": int(random_seed),
        "total_cells": int(total_cells),
        "total_cells_with_any_neighbor": int(total_cells_with_any_neighbor),
        "num_candidate_neighborhoods": int(num_candidate_neighborhoods),
        "num_filtered_out_for_threshold_or_min_neighbors": int(
            total_cells - num_candidate_neighborhoods
        ),
        "num_sampled_neighborhoods": int(num_sampled_neighborhoods),
        "num_unsampled_candidate_neighborhoods": int(
            num_candidate_neighborhoods - num_sampled_neighborhoods
        ),
        "num_sampled_instances": int(num_sampled_instances),
    }


def save_filtering_summary(summary: dict[str, float | int | str], out_dir: str) -> str:
    out_path = os.path.join(out_dir, "filtering_summary.json")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    return out_path


def save_neighborhood_instances_df(instances_df: pd.DataFrame, out_dir: str) -> str:
    out_path = os.path.join(out_dir, "sampled_neighborhood_instances.csv")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    instances_df.to_csv(out_path, index=False)
    return out_path


def save_neighborhood_map_df(neighborhoods_df: pd.DataFrame, out_dir: str) -> str:
    out_path = os.path.join(out_dir, "sampled_neighborhood_map.csv")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    neighborhoods_df.to_csv(out_path, index=False)
    return out_path


def build_inference_dataset(
    cell_instances_path: str,
) -> SingleCellListInferenceDataset:
    if not os.path.isfile(cell_instances_path):
        raise FileNotFoundError(
            f"Cell instances CSV does not exist: {cell_instances_path}"
        )
    if not dset_cf_yaml.strip():
        raise ValueError(
            "dset_cf_yaml is empty. Fill in the dataset YAML in this file."
        )

    cf = OmegaConf.create(yaml.safe_load(dset_cf_yaml))
    cf.data.test_dataset.params.cell_instances = cell_instances_path
    dataset = SingleCellListInferenceDataset(
        transform=HistologyTransform(**cf.data.xform_params),
        **cf.data.test_dataset.params,
    )
    if len(dataset) == 0:
        raise ValueError("Image dataset instantiated with zero cells")
    return dataset


def build_dataset_path_to_index(
    dataset: SingleCellListInferenceDataset,
) -> dict[str, int]:
    path_to_index: dict[str, int] = {}
    for idx, inst in dataset.instances_.iterrows():
        path = f"{inst['patch']}#{int(inst['proposal'][0])}_{int(inst['proposal'][1])}"
        if path in path_to_index:
            raise ValueError(f"Duplicate dataset path encountered: {path}")
        path_to_index[path] = int(idx)

    if not path_to_index:
        raise ValueError("No dataset paths were built for image export")
    return path_to_index


def save_neighborhood_image_grid(
    dataset: SingleCellListInferenceDataset,
    neighborhoods_df: pd.DataFrame,
    out_dir: str,
    num_rows: int = 8,
    max_neighbors_per_row: int = 16,
    draw_center_x: bool = False,
) -> str:
    if neighborhoods_df.empty:
        raise ValueError("Cannot save neighborhood images for an empty dataframe")
    if num_rows < 1:
        raise ValueError(f"num_rows must be >= 1, got {num_rows}")
    if max_neighbors_per_row < 1:
        raise ValueError(
            f"max_neighbors_per_row must be >= 1, got {max_neighbors_per_row}"
        )

    path_to_index = build_dataset_path_to_index(dataset)
    visualized_df = neighborhoods_df.head(num_rows).reset_index(drop=True)
    num_cols = 1 + max_neighbors_per_row
    out_path = os.path.join(
        out_dir,
        (
            "sampled_neighborhood_images_with_center_x.png"
            if draw_center_x
            else "sampled_neighborhood_images.png"
        ),
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(2.5 * num_cols, 2.5 * num_rows),
        squeeze=False,
    )

    for row_idx in range(num_rows):
        if row_idx >= len(visualized_df):
            for col_idx in range(num_cols):
                axes[row_idx, col_idx].axis("off")
            continue

        row = visualized_df.iloc[row_idx]
        anchor_and_neighbors = [row["anchor_path"]] + json.loads(
            row["neighbor_paths_json"]
        )[:max_neighbors_per_row]
        anchor_and_neighbors += [None] * (num_cols - len(anchor_and_neighbors))
        neighbor_distances = json.loads(row["neighbor_distances_json"])[
            :max_neighbors_per_row
        ]

        for col_idx, cell_path in enumerate(anchor_and_neighbors):
            ax = axes[row_idx, col_idx]
            if cell_path is None:
                ax.axis("off")
                continue
            if cell_path not in path_to_index:
                raise KeyError(f"Neighborhood path not found in dataset: {cell_path}")

            image = dataset[path_to_index[cell_path]]["image"].squeeze()
            image_np = prepare_three_channel_viz_image(image).numpy()
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
                    f"anchor\nd={0:.2f}\nn={row['num_neighbors']}",
                    fontsize=8,
                )
            else:
                ax.set_title(
                    f"n{col_idx}\nd={neighbor_distances[col_idx - 1]:.2f}", fontsize=8
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_job(
    csv_path: str,
    out_root: str,
    base_out_name: str,
    distance_bounds: list[float | None] | tuple[float | None, float | None],
    min_neighbors: int,
    num_cells: int = 8192,
    random_seed: int = 0,
    num_visualized_anchors: int = 8,
    max_visualized_neighbors: int = 16,
) -> None:
    validate_job_args(
        distance_bounds=distance_bounds,
        min_neighbors=min_neighbors,
        num_cells=num_cells,
        random_seed=random_seed,
    )
    out_dir = build_job_out_dir(
        out_root=out_root,
        base_out_name=base_out_name,
        num_cells=num_cells,
        distance_bounds=distance_bounds,
        min_neighbors=min_neighbors,
    )
    df, original_columns = prepare_instances(csv_path)
    candidate_neighborhoods_df, stats = collect_candidate_neighborhoods(
        df=df,
        distance_bounds=distance_bounds,
        min_neighbors=min_neighbors,
    )
    sampled_neighborhoods_df = sample_neighborhoods(
        candidate_neighborhoods_df,
        num_cells=num_cells,
        random_seed=random_seed,
    )
    neighborhood_instances_df, path_to_instance_index = (
        collect_neighborhood_instance_rows(
            df=df,
            neighborhoods_df=sampled_neighborhoods_df,
            original_columns=original_columns,
        )
    )
    mapped_neighborhoods_df = attach_instance_indices(
        sampled_neighborhoods_df, path_to_instance_index=path_to_instance_index
    )
    filtering_summary = build_filtering_summary(
        csv_path=csv_path,
        distance_bounds=distance_bounds,
        min_neighbors=min_neighbors,
        num_cells_requested=num_cells,
        random_seed=random_seed,
        total_cells=stats["total_cells"],
        total_cells_with_any_neighbor=stats["total_cells_with_any_neighbor"],
        num_candidate_neighborhoods=len(candidate_neighborhoods_df),
        num_sampled_neighborhoods=len(mapped_neighborhoods_df),
        num_sampled_instances=len(neighborhood_instances_df),
    )

    neighborhood_instances_path = save_neighborhood_instances_df(
        instances_df=neighborhood_instances_df,
        out_dir=out_dir,
    )
    neighborhood_map_path = save_neighborhood_map_df(
        neighborhoods_df=mapped_neighborhoods_df,
        out_dir=out_dir,
    )
    filtering_summary_path = save_filtering_summary(
        summary=filtering_summary, out_dir=out_dir
    )
    dataset = build_inference_dataset(cell_instances_path=neighborhood_instances_path)
    neighborhood_image_path = save_neighborhood_image_grid(
        dataset=dataset,
        neighborhoods_df=mapped_neighborhoods_df,
        out_dir=out_dir,
        num_rows=num_visualized_anchors,
        max_neighbors_per_row=max_visualized_neighbors,
    )
    neighborhood_image_center_x_path = save_neighborhood_image_grid(
        dataset=dataset,
        neighborhoods_df=mapped_neighborhoods_df,
        out_dir=out_dir,
        num_rows=num_visualized_anchors,
        max_neighbors_per_row=max_visualized_neighbors,
        draw_center_x=True,
    )

    print(f"Saved filtering summary to {filtering_summary_path}")
    print(f"Saved neighborhood instances to {neighborhood_instances_path}")
    print(f"Saved neighborhood map to {neighborhood_map_path}")
    print(f"Saved neighborhood images to {neighborhood_image_path}")
    print(
        "Saved neighborhood images with center x to "
        f"{neighborhood_image_center_x_path}"
    )
    print(json.dumps(filtering_summary, indent=2, sort_keys=True))


def main() -> None:
    job_defaults = {
        "csv_path": "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_test_.csv",
        "base_out_name": "cellnbr_stats",
        "out_root": ".",
        "num_cells": 8192,
        "random_seed": 0,
        "min_neighbors": 1,
    }

    jobs = [
        {
            "distance_bounds": [4, 8],
        },
        {
            "distance_bounds": [8, 12],
        },
        # {
        #    "distance_bounds": [12, 16],
        # },
        # {
        #    "distance_bounds": [16, 20],
        # },
        # {
        #    "distance_bounds": [20, 24],
        # },
        # {
        #    "distance_bounds": [24, 28],
        # },
        # {
        #    "distance_bounds": [28, 32],
        # },
        # {
        #    "distance_bounds": [32, 36],
        # },
        # {
        #    "distance_bounds": [36, 40],
        # },
        # {
        #    "distance_bounds": [40, 44],
        # },
        # {
        #    "distance_bounds": [44, 48],
        # },
        # {
        #    "distance_bounds": [60, 64],
        # },
    ]

    for job in jobs:
        run_job(
            **job_defaults,
            **job,
        )


if __name__ == "__main__":
    main()

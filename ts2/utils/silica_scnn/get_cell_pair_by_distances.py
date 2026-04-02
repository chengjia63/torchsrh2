import json
import os
from typing import Dict, List, Tuple

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
      #- which: srh_instance_norm_always_apply
      #  params: {}
      aug_prob: 1
  test_dataset:
    which: SingleCellListInferenceDataset
    params:
      data_root: /nfs/turbo/umms-tocho-snr/data/root_histology_db/srh/
      cell_instances: TO BE FILLED IN
  loader:
    params:
      test:
        pin_memory: False
        persistent_workers: False
        prefetch_factor: 8
        num_workers: 7
        batch_size: 256
        drop_last: False
        shuffle: False
"""


def format_bound_for_path(bound: float | None) -> str:
    if bound is None:
        return "none"
    if int(bound) == bound:
        return str(int(bound))
    return str(bound).replace(".", "p")


def parse_distance_bounds(
    distance_bounds: list[float | None] | tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    if len(distance_bounds) != 2:
        raise ValueError(
            f"distance_bounds must have exactly two elements [lower, upper], got {distance_bounds}"
        )
    return distance_bounds[0], distance_bounds[1]


def get_distance_mode_tag(distance_mode: str) -> str:
    if distance_mode == "nearest_neighbor":
        return "nn"
    if distance_mode == "all_pairdistance":
        return "ap"
    raise ValueError(f"Unhandled distance_mode: {distance_mode}")


def build_job_out_dir(
    out_root: str,
    base_out_name: str,
    distance_mode: str,
    num_pairs: int,
    distance_bounds: list[float | None] | tuple[float | None, float | None],
) -> str:
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)
    distance_mode_tag = get_distance_mode_tag(distance_mode)
    folder_name = (
        f"{base_out_name}_{distance_mode_tag}_{num_pairs}_"
        f"{format_bound_for_path(distance_lower_bound)}_"
        f"{format_bound_for_path(distance_upper_bound)}"
    )
    return os.path.join(out_root, folder_name)


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


def collect_nearest_neighbor_pairs(
    df: pd.DataFrame,
    distance_bounds: (
        list[float | None] | tuple[float | None, float | None] | None
    ) = None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    num_pairs_before_filtering = 0
    distance_lower_bound = None
    distance_upper_bound = None
    if distance_bounds is not None:
        distance_lower_bound, distance_upper_bound = parse_distance_bounds(
            distance_bounds
        )
        validate_distance_bounds(distance_lower_bound, distance_upper_bound)

    grouped = df.groupby("patch", sort=True)
    for patch, group in tqdm(
        grouped,
        total=df["patch"].nunique(),
        desc="Patch groups",
    ):
        coords = group[["proposal_row", "proposal_col"]].to_numpy(dtype=np.float64)
        if coords.shape[0] < 2:
            continue

        proposals = group["proposal"].tolist()
        deltas = coords[:, None, :] - coords[None, :, :]
        distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=-1, dtype=np.float64))
        np.fill_diagonal(distance_matrix, np.inf)
        nearest_j = distance_matrix.argmin(axis=1)
        nearest_distances = distance_matrix[np.arange(coords.shape[0]), nearest_j]
        num_pairs_before_filtering += len(nearest_distances)
        for i, j, distance in zip(
            np.arange(coords.shape[0]), nearest_j, nearest_distances
        ):
            if distance_lower_bound is not None and distance < distance_lower_bound:
                continue
            if distance_upper_bound is not None and distance > distance_upper_bound:
                continue
            rows.append(
                {
                    "patch": patch,
                    "path_a": f"{patch}#{int(coords[i, 0])}_{int(coords[i, 1])}",
                    "path_b": f"{patch}#{int(coords[j, 0])}_{int(coords[j, 1])}",
                    "proposal_a": proposals[i],
                    "proposal_b": proposals[j],
                    "proposal_a_row": int(coords[i, 0]),
                    "proposal_a_col": int(coords[i, 1]),
                    "proposal_b_row": int(coords[j, 0]),
                    "proposal_b_col": int(coords[j, 1]),
                    "distance": float(distance),
                }
            )
        if not nearest_distances.size and len(group) > 1:
            raise RuntimeError(
                f"Patch {patch} has {len(group)} proposals but produced no nearest-neighbor distances"
            )

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        if distance_bounds is None:
            raise ValueError(
                "No within-patch nearest-neighbor distances were found in the dataset"
            )
        raise ValueError(
            "No pairs remained after applying distance bounds "
            f"lower={distance_lower_bound}, upper={distance_upper_bound}"
        )
    return pairs_df, int(num_pairs_before_filtering)


def collect_all_within_patch_pairs(
    df: pd.DataFrame,
    distance_bounds: (
        list[float | None] | tuple[float | None, float | None] | None
    ) = None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    num_pairs_before_filtering = 0
    distance_lower_bound = None
    distance_upper_bound = None
    if distance_bounds is not None:
        distance_lower_bound, distance_upper_bound = parse_distance_bounds(
            distance_bounds
        )
        validate_distance_bounds(distance_lower_bound, distance_upper_bound)

    grouped = df.groupby("patch", sort=True)
    for patch, group in tqdm(
        grouped,
        total=df["patch"].nunique(),
        desc="Collect pairs",
    ):
        coords = group[["proposal_row", "proposal_col"]].to_numpy(dtype=np.float64)
        if coords.shape[0] < 2:
            continue

        proposals = group["proposal"].tolist()
        upper_i, upper_j = np.triu_indices(coords.shape[0], k=1)
        num_pairs_before_filtering += len(upper_i)

        deltas = coords[:, None, :] - coords[None, :, :]
        distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=-1, dtype=np.float64))
        pair_distances = distance_matrix[upper_i, upper_j]
        if distance_lower_bound is not None:
            keep_mask = pair_distances >= distance_lower_bound
            upper_i = upper_i[keep_mask]
            upper_j = upper_j[keep_mask]
            pair_distances = pair_distances[keep_mask]
        if distance_upper_bound is not None:
            keep_mask = pair_distances <= distance_upper_bound
            upper_i = upper_i[keep_mask]
            upper_j = upper_j[keep_mask]
            pair_distances = pair_distances[keep_mask]

        for i, j, distance in zip(upper_i, upper_j, pair_distances):
            rows.append(
                {
                    "patch": patch,
                    "path_a": f"{patch}#{int(coords[i, 0])}_{int(coords[i, 1])}",
                    "path_b": f"{patch}#{int(coords[j, 0])}_{int(coords[j, 1])}",
                    "proposal_a": proposals[i],
                    "proposal_b": proposals[j],
                    "proposal_a_row": int(coords[i, 0]),
                    "proposal_a_col": int(coords[i, 1]),
                    "proposal_b_row": int(coords[j, 0]),
                    "proposal_b_col": int(coords[j, 1]),
                    "distance": float(distance),
                }
            )

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        raise ValueError(
            "No pairs remained after applying distance bounds "
            f"lower={distance_lower_bound}, upper={distance_upper_bound}"
        )
    return pairs_df, int(num_pairs_before_filtering)


def sample_pairs(
    pairs_df: pd.DataFrame, num_pairs: int, random_seed: int
) -> pd.DataFrame:
    if num_pairs < 1:
        raise ValueError(f"num_pairs must be >= 1, got {num_pairs}")
    if pairs_df.empty:
        raise ValueError("No within-patch proposal pairs were found in the dataset")

    sample_size = min(num_pairs, len(pairs_df))
    if sample_size == len(pairs_df):
        return pairs_df.reset_index(drop=True)

    return pairs_df.sample(
        n=sample_size, replace=False, random_state=random_seed
    ).reset_index(drop=True)


def validate_distance_bounds(
    distance_lower_bound: float | None, distance_upper_bound: float | None
) -> None:
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


def collect_pair_instance_rows(
    df: pd.DataFrame, pairs_df: pd.DataFrame, original_columns: List[str]
) -> pd.DataFrame:
    if "path" not in df.columns:
        raise KeyError("Expected `path` column in instance dataframe")
    if "path_a" not in pairs_df.columns or "path_b" not in pairs_df.columns:
        raise KeyError("Expected `path_a` and `path_b` columns in pairs dataframe")

    selected_paths = list(
        dict.fromkeys(pairs_df["path_a"].tolist() + pairs_df["path_b"].tolist())
    )
    selected = df[df["path"].isin(selected_paths)].copy()
    if selected.empty:
        raise ValueError("No instance rows matched the sampled pairs")
    if len(selected) != len(selected_paths):
        raise ValueError(
            f"Expected {len(selected_paths)} unique pair instance rows, found {len(selected)}"
        )

    return selected.loc[:, original_columns].reset_index(drop=True)


def build_filtering_summary(
    csv_path: str,
    distance_mode: str,
    distance_bounds: list[float | None] | tuple[float | None, float | None],
    num_pairs_requested: int,
    num_visualized_pairs_requested: int,
    random_seed: int,
    num_pairs_before_filtering: int,
    num_pairs_after_filtering: int,
    num_sampled_pairs: int,
    num_pair_instances: int,
    num_visualized_pairs: int,
) -> dict[str, float | int | str | None]:
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)
    return {
        "csv_path": csv_path,
        "distance_mode": distance_mode,
        "distance_lower_bound": (
            None if distance_lower_bound is None else float(distance_lower_bound)
        ),
        "distance_upper_bound": (
            None if distance_upper_bound is None else float(distance_upper_bound)
        ),
        "pair_sample_size_requested": int(num_pairs_requested),
        "num_visualized_pairs_requested": int(num_visualized_pairs_requested),
        "pair_sample_random_seed": int(random_seed),
        "num_pairs_before_filtering": int(num_pairs_before_filtering),
        "num_pairs_after_filtering": int(num_pairs_after_filtering),
        "num_pairs_removed_by_filtering": int(
            num_pairs_before_filtering - num_pairs_after_filtering
        ),
        "num_sampled_pairs": int(num_sampled_pairs),
        "num_unsampled_filtered_pairs": int(
            num_pairs_after_filtering - num_sampled_pairs
        ),
        "num_pair_instances": int(num_pair_instances),
        "num_visualized_pairs": int(num_visualized_pairs),
    }


def save_filtering_summary(
    summary: Dict[str, float | int | str | None], out_dir: str
) -> str:
    out_path = os.path.join(out_dir, "filtering_summary.json")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    return out_path


def save_pair_instances_df(instances_df: pd.DataFrame, out_dir: str) -> str:
    out_path = os.path.join(out_dir, "sampled_pair_instances.csv")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    instances_df.to_csv(out_path, index=False)
    return out_path


def save_pair_map_df(pairs_df: pd.DataFrame, out_dir: str) -> str:
    out_path = os.path.join(out_dir, "sampled_pair_map.csv")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pairs_df.to_csv(out_path, index=False)
    return out_path


def build_inference_dataset(cell_instances_path: str) -> SingleCellListInferenceDataset:
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


def save_pair_image_grid(
    dataset: SingleCellListInferenceDataset,
    pairs_df: pd.DataFrame,
    out_dir: str,
    draw_center_x: bool = False,
) -> str:
    if pairs_df.empty:
        raise ValueError("Cannot save pair images for an empty pair dataframe")

    path_to_index = build_dataset_path_to_index(dataset)
    out_path = os.path.join(
        out_dir,
        (
            "sampled_pair_images_with_center_x.png"
            if draw_center_x
            else "sampled_pair_images.png"
        ),
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(pairs_df),
        figsize=(3 * len(pairs_df), 6),
        squeeze=False,
    )

    for row_idx, (_, pair_row) in enumerate(pairs_df.iterrows()):
        for col_idx, path_key in enumerate(("path_a", "path_b")):
            cell_path = pair_row[path_key]
            if cell_path not in path_to_index:
                raise KeyError(f"Sampled pair path not found in dataset: {cell_path}")

            image = dataset[path_to_index[cell_path]]["image"].squeeze()
            image_np = prepare_three_channel_viz_image(image).numpy()
            ax = axes[col_idx, row_idx]
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

            proposal_label = "A" if path_key == "path_a" else "B"
            ax.set_title(
                f"{proposal_label}: {cell_path}\nd={pair_row['distance']:.2f}",
                fontsize=8,
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
    distance_mode: str = "nearest_neighbor",
    num_pairs: int = 8192,
    num_visualized_pairs: int = 16,
    random_seed: int = 0,
) -> None:
    distance_lower_bound, distance_upper_bound = parse_distance_bounds(distance_bounds)
    out_dir = build_job_out_dir(
        out_root=out_root,
        base_out_name=base_out_name,
        distance_mode=distance_mode,
        num_pairs=num_pairs,
        distance_bounds=distance_bounds,
    )
    df, original_columns = prepare_instances(csv_path)

    if distance_mode == "all_pairdistance":
        collect_fn = collect_all_within_patch_pairs
    elif distance_mode == "nearest_neighbor":
        collect_fn = collect_nearest_neighbor_pairs
    else:
        raise ValueError(f"Unhandled distance_mode: {distance_mode}")

    candidate_pairs_df, num_pairs_before_filtering = collect_fn(
        df,
        distance_bounds=distance_bounds,
    )
    num_pairs_after_filtering = int(len(candidate_pairs_df))
    print(
        f"Applying thresholding for {distance_mode}: "
        f"lower={distance_lower_bound}, upper={distance_upper_bound}, "
        f"pairs_before_filtering={num_pairs_before_filtering}, "
        f"pairs_after_filtering={num_pairs_after_filtering}"
    )
    sampled_pairs_df = sample_pairs(
        candidate_pairs_df, num_pairs=num_pairs, random_seed=random_seed
    )

    pair_instances_df = collect_pair_instance_rows(
        df=df,
        pairs_df=sampled_pairs_df,
        original_columns=original_columns,
    )
    filtering_summary = build_filtering_summary(
        csv_path=csv_path,
        distance_mode=distance_mode,
        distance_bounds=distance_bounds,
        num_pairs_requested=num_pairs,
        num_visualized_pairs_requested=num_visualized_pairs,
        random_seed=random_seed,
        num_pairs_before_filtering=num_pairs_before_filtering,
        num_pairs_after_filtering=num_pairs_after_filtering,
        num_sampled_pairs=len(sampled_pairs_df),
        num_pair_instances=len(pair_instances_df),
        num_visualized_pairs=min(num_visualized_pairs, len(sampled_pairs_df)),
    )

    pair_instances_path = save_pair_instances_df(
        instances_df=pair_instances_df, out_dir=out_dir
    )
    pair_map_path = save_pair_map_df(pairs_df=sampled_pairs_df, out_dir=out_dir)
    filtering_summary_path = save_filtering_summary(
        summary=filtering_summary, out_dir=out_dir
    )

    dataset = build_inference_dataset(cell_instances_path=pair_instances_path)
    visualized_pairs_df = sample_pairs(
        sampled_pairs_df,
        num_pairs=num_visualized_pairs,
        random_seed=random_seed,
    )
    pair_image_path = save_pair_image_grid(
        dataset=dataset,
        pairs_df=visualized_pairs_df,
        out_dir=out_dir,
    )
    pair_image_center_x_path = save_pair_image_grid(
        dataset=dataset,
        pairs_df=visualized_pairs_df,
        out_dir=out_dir,
        draw_center_x=True,
    )

    print(f"Saved filtering summary to {filtering_summary_path}")
    print(f"Saved pair instances to {pair_instances_path}")
    print(f"Saved pair map to {pair_map_path}")
    print(f"Saved pair images to {pair_image_path}")
    print(f"Saved pair images with center x to {pair_image_center_x_path}")
    print(json.dumps(filtering_summary, indent=2, sort_keys=True))


def main() -> None:
    job_defaults = {
        "csv_path": "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_test_.csv",
        "base_out_name": "celldist_stats",
        "out_root": ".",
        "distance_mode": "all_pairdistance",  # "nearest_neighbor",  #
        "num_pairs": 8192,
        "num_visualized_pairs": 16,
        "random_seed": 0,
    }

    jobs = [
        # {
        #    "distance_bounds": [0, 8],
        # },
        # {
        #    "distance_bounds": [0, 16],
        # },
        # {
        #    "distance_bounds": [0, 24],
        # },
        {
            "distance_bounds": [4, 8],
        },
        {
            "distance_bounds": [8, 12],
        },
        {
            "distance_bounds": [12, 16],
        },
        {
            "distance_bounds": [16, 20],
        },
        {
            "distance_bounds": [20, 24],
        },
        {
            "distance_bounds": [24, 28],
        },
        {
            "distance_bounds": [28, 32],
        },
        {
            "distance_bounds": [32, 36],
        },
        {
            "distance_bounds": [36, 40],
        },
        {
            "distance_bounds": [40, 44],
        },
        {
            "distance_bounds": [44, 48],
        },
    ]

    for job in jobs:
        run_job(
            **job_defaults,
            **job,
        )


if __name__ == "__main__":
    main()

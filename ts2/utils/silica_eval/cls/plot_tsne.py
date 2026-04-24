import logging
import os
from os.path import join as opj
import json
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from ts2.data.transforms import HistologyTransform
from ts2.train.main_cell_inference import SingleCellListInferenceDataset
from ts2.utils.silica_sc_cls.eval_cell_inference_knn import (
    encode_cell_path,
    extract_perturb_key,
    find_matching_prediction_paths,
    infer_results_dir_from_prediction_path,
    load_labels_from_cell_instances,
    load_prediction,
)
from ts2.utils.silica_sc_eval.fit_gmm import (
    build_tsne_axis,
    compute_tsne,
    im_to_bytestr,
)
from ts2.utils.srh_viz import get_third_channel, prepare_three_channel_viz_image
from ts2.utils.tailwind import TC


def sample_prediction_deterministically(
    pred: Dict[str, Any],
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    num_rows = pred["embeddings"].shape[0]
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if num_rows < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size} rows from prediction with only {num_rows} rows"
        )

    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(num_rows, size=sample_size, replace=False))
    sampled_indices_tensor = torch.as_tensor(sampled_indices, dtype=torch.long)

    return {
        "embeddings": pred["embeddings"][sampled_indices_tensor],
        "label": [
            (
                pred["label"][i].item()
                if isinstance(pred["label"][i], torch.Tensor)
                else pred["label"][i]
            )
            for i in sampled_indices.tolist()
        ],
        "path": [pred["path"][i] for i in sampled_indices.tolist()],
    }


def build_path_to_index_from_dataset(
    dataset: SingleCellListInferenceDataset,
) -> Dict[str, int]:
    inst_df = dataset.instances_.copy().reset_index(names="dataset_index")
    inst_df["path"] = [
        encode_cell_path(patch, proposal)
        for patch, proposal in zip(inst_df["patch"], inst_df["proposal"])
    ]
    dedup_df = inst_df.drop_duplicates(subset=["path"], keep="first")
    duplicate_count = len(inst_df) - len(dedup_df)
    if duplicate_count > 0:
        logging.warning(
            "Dataset instances contain %d duplicate path rows; using the first occurrence for each path",
            duplicate_count,
        )

    return {
        path: int(dataset_index)
        for path, dataset_index in zip(
            dedup_df["path"].tolist(), dedup_df["dataset_index"].tolist()
        )
    }


def load_viz_images_for_paths(
    sampled_paths: List[str],
    parsed_config_path: str,
    cell_instances_path: str,
) -> tuple[List[np.ndarray], List[str]]:
    dataset = build_dataset_from_parsed_config(
        parsed_config_path=parsed_config_path,
        cell_instances_path=cell_instances_path,
    )
    path_to_index = build_path_to_index_from_dataset(dataset)
    missing_paths = sorted(set(sampled_paths) - set(path_to_index))
    if missing_paths:
        raise ValueError(
            f"Could not map {len(missing_paths)} sampled paths to {cell_instances_path}. "
            f"Examples: {missing_paths[:5]}"
        )

    source_indices = [path_to_index[path] for path in sampled_paths]
    image_arrays = [
        prepare_viz_image(dataset[i]["image"]).numpy()
        for i in tqdm(source_indices, desc="Loading sampled cell images")
    ]
    if len(image_arrays) != len(sampled_paths):
        raise ValueError(
            f"Loaded {len(image_arrays)} images for {len(sampled_paths)} sampled paths"
        )
    image_strs = [im_to_bytestr(image) for image in image_arrays]
    return image_arrays, image_strs


def prepare_viz_image(image: torch.Tensor) -> torch.Tensor:
    image = image.squeeze()
    if image.ndim != 3:
        raise ValueError(
            f"Expected sampled image tensor to be rank-3 after squeeze, got shape {tuple(image.shape)}"
        )

    if image.shape[0] == 2:
        image = get_third_channel(image.float() / 65536.0).clamp(0, 1)
    elif image.shape[0] >= 3:
        image = image[:3]
    else:
        raise ValueError(
            f"Expected sampled image tensor to have at least 2 channels, got shape {tuple(image.shape)}"
        )

    return prepare_three_channel_viz_image(image)


def normalize_tsne_coordinates(embeddings_2d: np.ndarray) -> np.ndarray:
    if embeddings_2d.ndim != 2 or embeddings_2d.shape[1] != 2:
        raise ValueError(
            f"Expected TSNE embeddings with shape (N, 2), got {embeddings_2d.shape}"
        )

    mins = embeddings_2d.min(axis=0)
    maxs = embeddings_2d.max(axis=0)
    spans = maxs - mins
    if np.any(spans <= 0):
        raise ValueError(
            f"Cannot normalize degenerate TSNE coordinates with axis spans {spans.tolist()}"
        )
    return (embeddings_2d - mins) / spans


def save_tsne_tiled_image(
    out_path: str,
    normalized_embeddings_2d: np.ndarray,
    image_arrays: List[np.ndarray],
    grid_size: int = 20,
    tile_size: int = 48,
    tile_padding: int = 0,
    distance_threshold: Optional[float] = None,
    circular_mask: bool = False,
) -> None:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_padding < 0:
        raise ValueError(f"tile_padding must be non-negative, got {tile_padding}")
    if len(image_arrays) != normalized_embeddings_2d.shape[0]:
        raise ValueError(
            "Expected one image per TSNE point, got "
            f"{len(image_arrays)} images for {normalized_embeddings_2d.shape[0]} points"
        )
    if normalized_embeddings_2d.ndim != 2 or normalized_embeddings_2d.shape[1] != 2:
        raise ValueError(
            "Expected normalized TSNE embeddings with shape (N, 2), got "
            f"{normalized_embeddings_2d.shape}"
        )
    if np.any(normalized_embeddings_2d < 0) or np.any(normalized_embeddings_2d > 1):
        raise ValueError("Expected normalized TSNE coordinates to lie in [0, 1]")

    if distance_threshold is None:
        distance_threshold = 0.5 / grid_size
    if distance_threshold <= 0:
        raise ValueError(
            f"distance_threshold must be positive, got {distance_threshold}"
        )

    cell_size = tile_size + 2 * tile_padding
    canvas_size = grid_size * cell_size
    canvas = np.zeros(
        (canvas_size, canvas_size, 4),
        dtype=np.uint8,
    )
    centers = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    tile_alpha_mask = None
    if circular_mask:
        yy, xx = np.ogrid[:tile_size, :tile_size]
        center = (tile_size - 1) / 2.0
        radius = tile_size / 2.0
        tile_alpha_mask = (
            ((xx - center) ** 2 + (yy - center) ** 2) <= radius**2
        ).astype(np.uint8) * 255

    for row, center_y in enumerate(centers):
        y0 = row * cell_size + tile_padding
        y1 = y0 + tile_size
        for col, center_x in enumerate(centers):
            x0 = col * cell_size + tile_padding
            x1 = x0 + tile_size
            grid_point = np.array([center_x, center_y], dtype=np.float32)
            distances = np.linalg.norm(normalized_embeddings_2d - grid_point, axis=1)
            nearest_idx = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_idx])
            if nearest_distance > distance_threshold:
                continue

            tile = image_arrays[nearest_idx]
            if tile.shape != (tile_size, tile_size, 3):
                raise ValueError(
                    f"Expected tile shape {(tile_size, tile_size, 3)}, got {tile.shape}"
                )
            canvas[y0:y1, x0:x1, :3] = tile
            if tile_alpha_mask is None:
                canvas[y0:y1, x0:x1, 3] = 255
            else:
                canvas[y0:y1, x0:x1, 3] = tile_alpha_mask

    Image.fromarray(canvas).save(out_path)


def build_dataset_from_parsed_config(
    parsed_config_path: str,
    cell_instances_path: str,
) -> SingleCellListInferenceDataset:
    if not os.path.isfile(parsed_config_path):
        raise FileNotFoundError(f"Parsed config does not exist: {parsed_config_path}")
    if not os.path.isfile(cell_instances_path):
        raise FileNotFoundError(
            f"Cell instances CSV does not exist: {cell_instances_path}"
        )

    with open(parsed_config_path, "r", encoding="utf-8") as fd:
        cf = OmegaConf.create(json.load(fd))
    cf = OmegaConf.create(OmegaConf.to_container(cf, resolve=True))

    assert "data" in cf
    assert "xform_params" in cf.data
    assert "test_dataset" in cf.data
    assert "params" in cf.data.test_dataset

    cf.data.xform_params.strong_aug_params.aug_list = []
    transform = HistologyTransform(**cf.data.xform_params)
    cf.data.test_dataset.params.cell_instances = cell_instances_path
    return SingleCellListInferenceDataset(
        transform=transform,
        **cf.data.test_dataset.params,
    )


def infer_parsed_config_path(pred_path: str) -> str:
    run_dir = os.path.dirname(os.path.dirname(pred_path))
    parsed_config_path = os.path.join(
        run_dir,
        "config",
        "parsed_config.json",
    )
    if not os.path.isfile(parsed_config_path):
        raise FileNotFoundError(
            f"Parsed config does not exist for prediction path {pred_path}: {parsed_config_path}"
        )
    return parsed_config_path


def build_tsne_runs_from_sets(
    exp_root: str,
    ckpt: str,
    run_sets: List[Dict[str, str]],
    run_key_prefix: str,
    databank_gt_csv_path: str,
    test_gt_csv_path: str,
    db_only: bool = False,
) -> List[Dict[str, str]]:
    if not exp_root:
        raise ValueError("Expected a non-empty exp_root")
    if not ckpt:
        raise ValueError("Expected a non-empty ckpt")

    runs: List[Dict[str, str]] = []
    for run_set in run_sets:
        required_keys = {
            "exp_name",
            "databank_pred_glob",
            "test_pred_glob",
        }
        missing = required_keys - set(run_set)
        if missing:
            raise KeyError(f"Run set is missing required keys: {sorted(missing)}")

        exp_name = run_set["exp_name"]
        base_eval_dir = opj(exp_root, exp_name, "models", "eval", ckpt)
        databank_glob = opj(
            base_eval_dir,
            run_set["databank_pred_glob"],
            "predictions",
            "pred.pt",
        )
        test_glob = opj(
            base_eval_dir,
            run_set["test_pred_glob"],
            "predictions",
            "pred.pt",
        )

        databank_matches = find_matching_prediction_paths(databank_glob)
        split_runs = []

        for databank_pred_path in databank_matches:
            split_runs.append(
                {
                    "name": f"{exp_name}_databank",
                    "split_name": "databank",
                    "pred_path": databank_pred_path,
                    "gt_csv_path": databank_gt_csv_path,
                    "sort_key": ("databank", databank_pred_path),
                }
            )

        if not db_only:
            test_matches = find_matching_prediction_paths(test_glob)
            for test_pred_path in test_matches:
                perturb_key = extract_perturb_key(test_pred_path, prefix=run_key_prefix)
                split_runs.append(
                    {
                        "name": f"{exp_name}_{perturb_key}_test",
                        "split_name": "test",
                        "pred_path": test_pred_path,
                        "gt_csv_path": test_gt_csv_path,
                        "sort_key": ("test", perturb_key, test_pred_path),
                    }
                )

        pred_paths = [run["pred_path"] for run in split_runs]
        if len(set(pred_paths)) != len(pred_paths):
            raise ValueError(
                f"Expected unique prediction paths for run set {exp_name}, found duplicates: {pred_paths}"
            )

        split_runs = sorted(split_runs, key=lambda run: run["sort_key"])
        runs.extend(
            {
                "name": run["name"],
                "split_name": run["split_name"],
                "pred_path": run["pred_path"],
                "gt_csv_path": run["gt_csv_path"],
            }
            for run in split_runs
        )

    return runs


def encode_prediction_labels(
    databank_labels: List[Any],
    test_labels: List[Any],
    class_names: Optional[List[str]] = None,
) -> tuple[List[int], List[int], List[str]]:
    databank_labels = [str(x) for x in databank_labels]
    test_labels = [str(x) for x in test_labels]
    unique_labels = list(dict.fromkeys(databank_labels + test_labels))

    if class_names is not None:
        class_names = [str(x) for x in class_names]
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        missing = sorted(set(unique_labels) - set(label_to_idx))
        if missing:
            raise ValueError(
                f"Found labels not present in class_names: {missing}. "
                f"class_names={class_names}"
            )
        return (
            [label_to_idx[x] for x in databank_labels],
            [label_to_idx[x] for x in test_labels],
            class_names,
        )

    all_numeric = all(label.lstrip("+-").isdigit() for label in unique_labels)
    if all_numeric:
        databank_encoded = [int(x) for x in databank_labels]
        test_encoded = [int(x) for x in test_labels]
        num_classes = max(databank_encoded + test_encoded) + 1
        inferred_class_names = [str(i) for i in range(num_classes)]
        return databank_encoded, test_encoded, inferred_class_names

    inferred_class_names = unique_labels
    label_to_idx = {name: idx for idx, name in enumerate(inferred_class_names)}
    return (
        [label_to_idx[x] for x in databank_labels],
        [label_to_idx[x] for x in test_labels],
        inferred_class_names,
    )


def save_chart(chart: alt.Chart, out_stem: str, formats: List[str]) -> None:
    out_dir = os.path.dirname(out_stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for fmt in formats:
        out_path = f"{out_stem}.{fmt}"
        try:
            chart.save(out_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save chart to {out_path}. "
                "If you requested png/pdf, ensure Altair export dependencies such as "
                "`vl-convert-python` are installed."
            ) from exc


def save_tsne_plot(
    out_dir: str,
    split_name: str,
    sampled_pred: Dict[str, Any],
    image_arrays: List[np.ndarray],
    image_strs: List[str],
    class_names: List[str],
    color_range: List[str],
    sample_size: int = 4096,
    seed: int = 0,
    perplexity: int = 50,
    tiled_grid_size: int = 40,
    tiled_tile_size: int = 48,
    tiled_tile_padding: int = 0,
    tiled_circular_mask: bool = True,
    formats: Optional[List[str]] = None,
) -> None:
    if formats is None:
        formats = ["html", "png", "pdf"]

    logging.info(
        "Creating %s TSNE plot with %d samples (seed=%d)",
        split_name,
        sample_size,
        seed,
    )
    num_rows = sampled_pred["embeddings"].shape[0]
    if num_rows != sample_size:
        raise ValueError(
            f"Expected sampled_pred to contain {sample_size} rows, got {num_rows}"
        )
    if len(image_arrays) != num_rows:
        raise ValueError(
            f"Expected {num_rows} image arrays for sampled prediction, got {len(image_arrays)}"
        )
    if len(image_strs) != num_rows:
        raise ValueError(
            f"Expected {num_rows} image strings for sampled prediction, got {len(image_strs)}"
        )
    if num_rows < 2:
        raise ValueError(f"TSNE requires at least 2 rows, got {num_rows}")
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")
    if perplexity >= num_rows:
        raise ValueError(
            f"perplexity must be smaller than the number of rows, got perplexity={perplexity} rows={num_rows}"
        )
    embeddings_2d = compute_tsne(
        sampled_pred["embeddings"].cpu().numpy(),
        perplexity=perplexity,
        random_state=seed,
    )
    normalized_embeddings_2d = normalize_tsne_coordinates(embeddings_2d)

    plot_df = pd.DataFrame(
        {
            "path": sampled_pred["path"],
            "label": sampled_pred["label"],
            "image": image_strs,
            "tsne_x": normalized_embeddings_2d[:, 0],
            "tsne_y": normalized_embeddings_2d[:, 1],
        }
    )
    plot_df["label_name"] = [class_names[idx] for idx in plot_df["label"]]

    alt.data_transformers.disable_max_rows()
    color_domain = list(dict.fromkeys(class_names))
    tsne_unit_axis = build_tsne_axis()
    chart = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=36, opacity=0.65)
        .encode(
            x=alt.X(
                "tsne_x:Q",
                title="",
                axis=tsne_unit_axis,
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "tsne_y:Q",
                title="",
                axis=tsne_unit_axis,
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "label_name:N",
                title="Label",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=["image:N", "label_name:N", "path:N"],
        )
        .properties(
            width=600,
            height=600,
        )
        .interactive()
    )

    tsne_dir = opj(out_dir, "tsne")
    os.makedirs(tsne_dir, exist_ok=True)
    plot_df.to_csv(opj(tsne_dir, f"{split_name}_tsne_points.csv"), index=False)
    save_chart(chart, out_stem=opj(tsne_dir, f"{split_name}_tsne"), formats=formats)
    save_tsne_tiled_image(
        out_path=opj(tsne_dir, f"{split_name}_tsne_tiled.png"),
        normalized_embeddings_2d=normalized_embeddings_2d,
        image_arrays=image_arrays,
        grid_size=tiled_grid_size,
        tile_size=tiled_tile_size,
        tile_padding=tiled_tile_padding,
    )
    if tiled_circular_mask:
        save_tsne_tiled_image(
            out_path=opj(tsne_dir, f"{split_name}_tsne_tiled_circular.png"),
            normalized_embeddings_2d=normalized_embeddings_2d,
            image_arrays=image_arrays,
            grid_size=tiled_grid_size,
            tile_size=tiled_tile_size,
            tile_padding=tiled_tile_padding,
            circular_mask=True,
        )


def load_labeled_predictions(
    pred_path: str,
    gt_csv_path: Optional[str],
    gt_label_column: str,
) -> tuple[Dict[str, Any], List[Any]]:
    pred = load_prediction(pred_path)
    if gt_csv_path is None:
        logging.info(
            "No GT CSV provided for %s; using labels from prediction file",
            pred_path,
        )
        return pred, pred["label"]

    labels = load_labels_from_cell_instances(
        pred["path"],
        gt_csv_path,
        gt_label_column,
    )
    return pred, labels


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    databank_gt_csv_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_1dot4m_.csv"
    test_gt_csv_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_test_.csv"
    class_names = [
        "hgg",
        "lgg",
        "mening",
        "metast",
        "normal",
        "pituita",
        "schwan",
    ]
    sample_size = 4096
    seed = 0
    perplexity = 50
    formats = ["html", "png", "pdf"]
    run_dir_prefix = "tsne"
    db_only = True
    tiled_grid_size = 40
    tiled_tile_size = 48
    tiled_tile_padding = 2
    tiled_circular_mask = True
    color_range = TC()(c="RALTFSV")

    run_sets = [
        # {
        #    "exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "a2706135_dinov2",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        {
            "exp_name": "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
    ]

    runs = build_tsne_runs_from_sets(
        exp_root="/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/",
        ckpt="training_124999",
        run_sets=run_sets,
        run_key_prefix="PERTURB",
        databank_gt_csv_path=databank_gt_csv_path,
        test_gt_csv_path=test_gt_csv_path,
        db_only=db_only,
    )

    for cfg in runs:
        pred, gt_labels = load_labeled_predictions(
            cfg["pred_path"],
            cfg["gt_csv_path"],
            gt_label_column="label",
        )
        encoded_labels, _, encoded_class_names = encode_prediction_labels(
            gt_labels,
            gt_labels,
            class_names=class_names,
        )
        pred["label"] = torch.as_tensor(encoded_labels, dtype=torch.long)
        parsed_config_path = infer_parsed_config_path(cfg["pred_path"])
        sampled_pred = sample_prediction_deterministically(
            pred,
            sample_size=sample_size,
            seed=seed,
        )
        image_arrays, image_strs = load_viz_images_for_paths(
            sampled_pred["path"],
            parsed_config_path=parsed_config_path,
            cell_instances_path=cfg["gt_csv_path"],
        )

        out_dir = infer_results_dir_from_prediction_path(
            cfg["pred_path"], run_dir_prefix=run_dir_prefix
        )
        save_tsne_plot(
            out_dir=out_dir,
            split_name=cfg["split_name"],
            sampled_pred=sampled_pred,
            image_arrays=image_arrays,
            image_strs=image_strs,
            class_names=encoded_class_names,
            color_range=color_range,
            sample_size=sample_size,
            seed=seed,
            perplexity=perplexity,
            tiled_grid_size=tiled_grid_size,
            tiled_tile_size=tiled_tile_size,
            tiled_tile_padding=tiled_tile_padding,
            tiled_circular_mask=tiled_circular_mask,
            formats=formats,
        )


if __name__ == "__main__":
    main()

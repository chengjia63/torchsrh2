import base64
import html
import logging
import os
from io import BytesIO
from typing import Callable, Sequence

import altair as alt
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pyvis.network import Network
from sklearn.mixture import GaussianMixture

from ts2.train.main_cell_inference import SingleCellListInferenceDataset
from ts2.utils.silica_eval.clustering.data import (
    build_dataset,
    load_embedding_table,
    sample_global_cells,
)
from ts2.utils.silica_eval.clustering.embeddings import prepare_embeddings
from ts2.utils.silica_eval.clustering.models import load_gmm_models
from ts2.utils.silica_eval.clustering.utils import configure_logging
from ts2.utils.srh_viz import prepare_three_channel_viz_image

logger = logging.getLogger(__name__)

GRAPH_HEIGHT = "900px"
GRAPH_WIDTH = "100%"
POSITION_SCALE = 1000.0
NODE_SIZE_BASE = 7.0
NODE_SIZE_SCALE = 0.45
NODE_SIZE_MIN = 9.0
NODE_SIZE_MAX = 24.0
SIZE_SLIDER_MIN = 0.4
SIZE_SLIDER_MAX = 2.5
SIZE_SLIDER_STEP = 0.05
EDGE_WIDTH = 1.0
LAYOUT_SPACING_ITERATIONS = 200
LAYOUT_MIN_NODE_GAP = 0.025
LAYOUT_NODE_GAP_SCALE = 1.8
LAYOUT_MARGIN = 0.05
LAYOUT_COMPONENT_GAP = 0.85
GRAPH_EDGE_LENGTH_MIN = 55.0
GRAPH_EDGE_LENGTH_MAX = 260.0
GRAPH_EDGE_LENGTH_POWER = 0.85
GRAPH_PHYSICS_STABILIZATION_ITERATIONS = 500
GAUSSIAN_DISTANCE_BATCH_SIZE = 16
CLUSTER_SAMPLE_GRID_CLOSEST_N = 4
CLUSTER_SAMPLE_GRID_RANDOM_N = 4
CLUSTER_SAMPLE_GRID_TILE_SIZE = 48
CLUSTER_SAMPLE_GRID_RANDOM_SEED = 0
INTERACTIVE_HEATMAP_MAX_SIZE_PX = 900
INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX = 160

DistanceMetricFn = Callable[[GaussianMixture], np.ndarray]
DistanceThresholdSpec = float | Sequence[float]


def ensure_merge_gmm_output_dir(out_dir: str) -> str:
    merge_dir = os.path.join(out_dir, "merge_gmm")
    os.makedirs(merge_dir, exist_ok=True)
    logger.info("Ensured merge GMM output directory: %s", merge_dir)
    return merge_dir


def ensure_merge_gmm_output_subdir(merge_dir: str, subdir_name: str) -> str:
    output_dir = os.path.join(merge_dir, subdir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_gmm_partition(
    db_embs_norm: torch.Tensor,
    gmm_model_dir: str,
    k: int,
) -> tuple[np.ndarray, GaussianMixture]:
    db_embs_np = db_embs_norm.cpu().numpy()
    logger.info(
        "Loading GMM with k=%d and predicting normalized embeddings with shape=%s",
        k,
        db_embs_np.shape,
    )
    gmm = load_gmm_models(out_dir=gmm_model_dir, k_range=[k])[0]
    labels = gmm.predict(db_embs_np).astype(np.int32)
    logger.info("GMM assignment vector shape: %s", labels.shape)
    return labels, gmm


def compute_cosine_mean_distance(gmm: GaussianMixture) -> np.ndarray:
    means = gmm.means_.astype(np.float64)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    normalized_means = means / np.maximum(norms, 1e-12)
    cosine_similarity = normalized_means @ normalized_means.T
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    distance = 1.0 - cosine_similarity
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)


def _component_diag_covariances(gmm: GaussianMixture) -> np.ndarray:
    if gmm.covariance_type == "diag":
        return gmm.covariances_.astype(np.float64)
    if gmm.covariance_type == "spherical":
        return np.repeat(
            gmm.covariances_[:, None].astype(np.float64),
            gmm.means_.shape[1],
            axis=1,
        )
    raise NotImplementedError(
        "Gaussian distance metrics are implemented for diag and spherical GMM covariances."
    )


def compute_wasserstein_2_distance(gmm: GaussianMixture) -> np.ndarray:
    means = gmm.means_.astype(np.float64)
    variances = np.maximum(_component_diag_covariances(gmm), 1e-12)
    n_clusters = means.shape[0]
    distance_squared = np.empty((n_clusters, n_clusters), dtype=np.float64)
    variance_sqrt = np.sqrt(variances)
    for start_idx in range(0, n_clusters, GAUSSIAN_DISTANCE_BATCH_SIZE):
        stop_idx = min(start_idx + GAUSSIAN_DISTANCE_BATCH_SIZE, n_clusters)
        mean_distance_squared = (
            (means[start_idx:stop_idx, None, :] - means[None, :, :]) ** 2
        ).sum(axis=2)
        covariance_distance_squared = (
            (variance_sqrt[start_idx:stop_idx, None, :] - variance_sqrt[None, :, :])
            ** 2
        ).sum(axis=2)
        distance_squared[start_idx:stop_idx] = (
            mean_distance_squared + covariance_distance_squared
        )
    np.fill_diagonal(distance_squared, 0.0)
    return np.sqrt(np.maximum(distance_squared, 0.0)).astype(np.float32)


def compute_symmetric_kl_divergence(gmm: GaussianMixture) -> np.ndarray:
    means = gmm.means_.astype(np.float64)
    variances = np.maximum(_component_diag_covariances(gmm), 1e-12)
    inverse_variances = 1.0 / variances
    n_clusters = means.shape[0]
    divergence = np.empty((n_clusters, n_clusters), dtype=np.float64)
    for start_idx in range(0, n_clusters, GAUSSIAN_DISTANCE_BATCH_SIZE):
        stop_idx = min(start_idx + GAUSSIAN_DISTANCE_BATCH_SIZE, n_clusters)
        mean_delta_squared = (
            means[start_idx:stop_idx, None, :] - means[None, :, :]
        ) ** 2
        divergence[start_idx:stop_idx] = 0.25 * (
            (variances[start_idx:stop_idx, None, :] * inverse_variances[None, :, :])
            + (variances[None, :, :] * inverse_variances[start_idx:stop_idx, None, :])
            + mean_delta_squared
            * (
                inverse_variances[start_idx:stop_idx, None, :]
                + inverse_variances[None, :, :]
            )
            - 2.0
        ).sum(axis=2)
    np.fill_diagonal(divergence, 0.0)
    return np.maximum(divergence, 0.0).astype(np.float32)


DISTANCE_METRICS: dict[str, DistanceMetricFn] = {
    "cosine_mean": compute_cosine_mean_distance,
    "wasserstein_2": compute_wasserstein_2_distance,
    "symmetric_kl": compute_symmetric_kl_divergence,
}


def save_cluster_distance_matrix(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
) -> None:
    n_clusters = distance.shape[0]
    assert distance.shape == (
        n_clusters,
        n_clusters,
    ), f"Expected square distance matrix, got shape {distance.shape}"
    distance_dir = ensure_merge_gmm_output_subdir(merge_dir, "distance")
    out_path = os.path.join(distance_dir, f"cluster_distance_{metric_name}.csv")
    pd.DataFrame(distance).to_csv(out_path, index_label="cluster")
    assert os.path.exists(out_path)


def save_cluster_distance_image(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
    suffix: str = "",
) -> None:
    distance_dir = ensure_merge_gmm_output_subdir(merge_dir, "distance")
    out_path = os.path.join(
        distance_dir,
        f"cluster_distance_{metric_name}{suffix}.png",
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(distance, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(f"{metric_name} cluster distance")
    fig.colorbar(image, ax=ax, label="Distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    assert os.path.exists(out_path)


def compute_binary_threshold_matrix(
    distance: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    return (distance <= distance_threshold).astype(np.uint8)


def _format_heatmap_value(value: float) -> str:
    return f"{value:.6g}"


def _build_cluster_centroid_records(
    node_table: pd.DataFrame | None,
) -> dict[int, dict[str, str]]:
    if node_table is None:
        return {}

    records: dict[int, dict[str, str]] = {}
    for _, row in node_table.iterrows():
        cluster = int(row["cluster"])
        records[cluster] = {
            "image": str(row["image"]),
            "label": str(row["representative_label"]),
            "patient": str(row["representative_patient"]),
            "path": str(row["representative_path"]),
            "size": str(int(row["cluster_size"])),
            "centroid_distance": _format_heatmap_value(float(row["centroid_distance"])),
        }
    return records


def _save_heatmap_centroid_assets(
    output_dir: str,
    centroid_records: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    records_with_paths: dict[int, dict[str, str]] = {}
    data_uri_prefix = "data:image/png;base64,"
    for cluster, record in centroid_records.items():
        updated_record = dict(record)
        image_data_uri = record["image"]
        if image_data_uri:
            assert image_data_uri.startswith(data_uri_prefix), (
                "Expected centroid image as PNG data URI for cluster "
                f"{cluster}, got {image_data_uri[:32]}"
            )
            image_filename = f"cluster_{int(cluster)}.png"
            image_path = os.path.join(assets_dir, image_filename)
            with open(image_path, "wb") as fd:
                fd.write(base64.b64decode(image_data_uri[len(data_uri_prefix) :]))
            assert os.path.exists(image_path)
            updated_record["image"] = os.path.join("assets", image_filename)
        records_with_paths[int(cluster)] = updated_record
    return records_with_paths


def _build_interactive_heatmap_table(
    matrix: np.ndarray,
    cluster_labels: Sequence[int],
    centroid_records: dict[int, dict[str, str]],
    value_label: str,
) -> pd.DataFrame:
    n_clusters = matrix.shape[0]
    rows: list[dict[str, object]] = []
    for row_idx in range(n_clusters):
        row_label = int(cluster_labels[row_idx])
        for col_idx in range(n_clusters):
            col_label = int(cluster_labels[col_idx])
            value = float(matrix[row_idx, col_idx])
            row: dict[str, object] = {
                "cell_id": f"{row_idx}:{col_idx}",
                "row_cluster": row_label,
                "column_cluster": col_label,
                "value": value,
                "x_image": "",
                "y_image": "",
                "x_label_text": f"X cluster {col_label}",
                "y_label_text": f"Y cluster {row_label}",
                "x_tooltip_text": f"X cluster {col_label}",
                "y_tooltip_text": f"Y cluster {row_label}",
                "preview_text": (
                    f"X,Y pair: ({col_label}, {row_label})\n"
                    f"X: cluster {col_label}\n"
                    f"Y: cluster {row_label}\n"
                    f"{value_label}: {_format_heatmap_value(value)}"
                ),
            }
            if col_label in centroid_records:
                x_record = centroid_records[col_label]
                row["x_image"] = x_record["image"]
                row["x_tooltip_text"] = (
                    f"X cluster {col_label}\n"
                    f"Label: {x_record['label']}\n"
                    f"Patient: {x_record['patient']}\n"
                    f"Size: {x_record['size']}\n"
                    f"Centroid distance: {x_record['centroid_distance']}"
                )
                row["preview_text"] = (
                    f"{row['preview_text']}\n"
                    f"X size: {x_record['size']}\n"
                    f"X centroid distance: {x_record['centroid_distance']}\n"
                    f"X label: {x_record['label']}\n"
                    f"X patient: {x_record['patient']}"
                )
            if row_label in centroid_records:
                y_record = centroid_records[row_label]
                row["y_image"] = y_record["image"]
                row["y_tooltip_text"] = (
                    f"Y cluster {row_label}\n"
                    f"Label: {y_record['label']}\n"
                    f"Patient: {y_record['patient']}\n"
                    f"Size: {y_record['size']}\n"
                    f"Centroid distance: {y_record['centroid_distance']}"
                )
                row["preview_text"] = (
                    f"{row['preview_text']}\n"
                    f"Y size: {y_record['size']}\n"
                    f"Y centroid distance: {y_record['centroid_distance']}\n"
                    f"Y label: {y_record['label']}\n"
                    f"Y patient: {y_record['patient']}"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _save_interactive_heatmap_html(
    output_dir: str,
    title: str,
    value_label: str,
    matrix: np.ndarray,
    node_table: pd.DataFrame | None,
    ordered_clusters: Sequence[int] | None,
    threshold_colors: bool,
) -> None:
    n_clusters = matrix.shape[0]
    assert matrix.shape == (
        n_clusters,
        n_clusters,
    ), f"Expected square heatmap matrix, got shape {matrix.shape}"
    if ordered_clusters is None:
        cluster_labels = list(range(n_clusters))
    else:
        cluster_labels = [int(cluster) for cluster in ordered_clusters]
    assert (
        len(cluster_labels) == n_clusters
    ), f"Expected {n_clusters} cluster labels, got {len(cluster_labels)}"

    centroid_records = _build_cluster_centroid_records(node_table=node_table)
    if node_table is not None:
        missing_clusters = sorted(set(cluster_labels).difference(centroid_records))
        assert (
            not missing_clusters
        ), f"Missing centroid records for clusters: {missing_clusters}"

    os.makedirs(output_dir, exist_ok=True)
    centroid_records = _save_heatmap_centroid_assets(
        output_dir=output_dir,
        centroid_records=centroid_records,
    )
    plot_table = _build_interactive_heatmap_table(
        matrix=matrix,
        cluster_labels=cluster_labels,
        centroid_records=centroid_records,
        value_label=value_label,
    )

    hover = alt.selection_point(
        fields=["cell_id"],
        on="mouseover",
        empty=False,
        clear="mouseout",
    )
    color = (
        alt.Color(
            "value:Q",
            scale=alt.Scale(domain=[0, 1], range=["#ffffff", "#111827"]),
            legend=alt.Legend(title=value_label),
        )
        if threshold_colors
        else alt.Color(
            "value:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=alt.Legend(title=value_label),
        )
    )
    axis = alt.Axis(labelAngle=0, title="Cluster")
    tooltip = [
        alt.Tooltip("row_cluster:O", title="Row cluster"),
        alt.Tooltip("column_cluster:O", title="Column cluster"),
        alt.Tooltip("value:Q", title=value_label, format=".6g"),
    ]
    heatmap = (
        alt.Chart(plot_table)
        .mark_rect()
        .encode(
            x=alt.X("column_cluster:O", sort=cluster_labels, axis=axis),
            y=alt.Y("row_cluster:O", sort=cluster_labels, axis=axis),
            color=color,
            stroke=alt.condition(hover, alt.value("#0f172a"), alt.value(None)),
            strokeWidth=alt.condition(hover, alt.value(2), alt.value(0)),
            tooltip=tooltip,
        )
        .add_params(hover)
        .properties(
            width=INTERACTIVE_HEATMAP_MAX_SIZE_PX,
            height=INTERACTIVE_HEATMAP_MAX_SIZE_PX,
            title=title,
        )
    )
    preview_gap = 32
    preview_width = 2 * INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX + preview_gap
    preview_base = alt.Chart(plot_table).transform_filter(hover)
    x_preview_image = preview_base.mark_image(
        width=INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX,
        height=INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX,
    ).encode(
        x=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX / 2),
        y=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX / 2),
        url="x_image:N",
        tooltip=alt.Tooltip("x_tooltip_text:N", title="X centroid"),
    )
    y_preview_image = preview_base.mark_image(
        width=INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX,
        height=INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX,
    ).encode(
        x=alt.value(
            INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX
            + preview_gap
            + (INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX / 2)
        ),
        y=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX / 2),
        url="y_image:N",
        tooltip=alt.Tooltip("y_tooltip_text:N", title="Y centroid"),
    )
    x_preview_label = preview_base.mark_text(
        align="center",
        baseline="top",
        fontSize=12,
    ).encode(
        x=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX / 2),
        y=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX + 10),
        text="x_label_text:N",
    )
    y_preview_label = preview_base.mark_text(
        align="center",
        baseline="top",
        fontSize=12,
    ).encode(
        x=alt.value(
            INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX
            + preview_gap
            + (INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX / 2)
        ),
        y=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX + 10),
        text="y_label_text:N",
    )
    preview_text = preview_base.mark_text(
        align="left",
        baseline="top",
        lineBreak="\n",
        fontSize=12,
    ).encode(
        x=alt.value(0),
        y=alt.value(INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX + 38),
        text="preview_text:N",
    )
    preview = (
        x_preview_image
        + y_preview_image
        + x_preview_label
        + y_preview_label
        + preview_text
    ).properties(
        width=preview_width,
        height=INTERACTIVE_HEATMAP_PREVIEW_SIZE_PX + 220,
        title="Centroid pair",
    )
    alt.data_transformers.disable_max_rows()
    index_path = os.path.join(output_dir, "index.html")
    (heatmap | preview).save(index_path)
    assert os.path.exists(index_path)


def save_binary_threshold_matrix(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
    distance_threshold: float,
    suffix: str = "",
    ordered_clusters: Sequence[int] | None = None,
) -> None:
    thresholded = compute_binary_threshold_matrix(
        distance=distance,
        distance_threshold=distance_threshold,
    )
    threshold_dir = ensure_merge_gmm_output_subdir(merge_dir, "threshold")
    out_path = os.path.join(
        threshold_dir,
        f"cluster_threshold_{metric_name}{suffix}.csv",
    )
    if ordered_clusters is None:
        pd.DataFrame(thresholded).to_csv(out_path, index_label="cluster")
    else:
        cluster_labels = [int(cluster) for cluster in ordered_clusters]
        pd.DataFrame(
            thresholded,
            index=cluster_labels,
            columns=cluster_labels,
        ).to_csv(out_path, index_label="cluster")
    assert os.path.exists(out_path)


def save_binary_threshold_image(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
    distance_threshold: float,
    suffix: str = "",
    node_table: pd.DataFrame | None = None,
    ordered_clusters: Sequence[int] | None = None,
) -> None:
    thresholded = compute_binary_threshold_matrix(
        distance=distance,
        distance_threshold=distance_threshold,
    )
    threshold_dir = ensure_merge_gmm_output_subdir(merge_dir, "threshold")
    out_path = os.path.join(
        threshold_dir,
        f"cluster_threshold_{metric_name}{suffix}.png",
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(
        thresholded,
        cmap="gray_r",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(f"{metric_name} threshold <= {distance_threshold:.6g}")
    fig.colorbar(image, ax=ax, label="Kept edge")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    assert os.path.exists(out_path)
    interactive_dir = os.path.join(
        threshold_dir,
        f"cluster_threshold_{metric_name}{suffix}",
    )
    _save_interactive_heatmap_html(
        output_dir=interactive_dir,
        title=f"{metric_name} threshold <= {distance_threshold:.6g}",
        value_label="Kept edge",
        matrix=thresholded.astype(np.float32),
        node_table=node_table,
        ordered_clusters=ordered_clusters,
        threshold_colors=True,
    )


def compute_similarity_from_distance(
    metric_name: str,
    distance: np.ndarray,
) -> np.ndarray:
    assert np.isfinite(distance).all(), "Expected finite distances for similarity plot"
    if metric_name == "cosine_mean":
        return (1.0 - distance).astype(np.float32)
    if metric_name == "symmetric_kl":
        return (-distance).astype(np.float32)

    upper_triangle = np.triu_indices(distance.shape[0], k=1)
    pair_distances = distance[upper_triangle]
    positive_distances = pair_distances[pair_distances > 0.0]
    if len(positive_distances) == 0:
        return np.ones_like(distance, dtype=np.float32)

    scale = float(np.median(positive_distances))
    similarity = np.exp(-distance.astype(np.float64) / scale)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def save_cluster_similarity_image(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
    suffix: str = "",
    node_table: pd.DataFrame | None = None,
    ordered_clusters: Sequence[int] | None = None,
) -> None:
    similarity = compute_similarity_from_distance(
        metric_name=metric_name,
        distance=distance,
    )
    similarity_dir = ensure_merge_gmm_output_subdir(merge_dir, "similarity")
    out_path = os.path.join(
        similarity_dir,
        f"cluster_similarity_{metric_name}{suffix}.png",
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(similarity, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(f"{metric_name} cluster similarity")
    fig.colorbar(image, ax=ax, label="Similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    assert os.path.exists(out_path)
    interactive_dir = os.path.join(
        similarity_dir,
        f"cluster_similarity_{metric_name}{suffix}",
    )
    _save_interactive_heatmap_html(
        output_dir=interactive_dir,
        title=f"{metric_name} cluster similarity",
        value_label="Similarity",
        matrix=similarity,
        node_table=node_table,
        ordered_clusters=ordered_clusters,
        threshold_colors=False,
    )


def save_cluster_similarity_matrix(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
    suffix: str = "",
    ordered_clusters: Sequence[int] | None = None,
) -> None:
    similarity = compute_similarity_from_distance(
        metric_name=metric_name,
        distance=distance,
    )
    similarity_dir = ensure_merge_gmm_output_subdir(merge_dir, "similarity")
    out_path = os.path.join(
        similarity_dir,
        f"cluster_similarity_{metric_name}{suffix}.csv",
    )
    if ordered_clusters is None:
        pd.DataFrame(similarity).to_csv(out_path, index_label="cluster")
    else:
        cluster_labels = [int(cluster) for cluster in ordered_clusters]
        pd.DataFrame(
            similarity,
            index=cluster_labels,
            columns=cluster_labels,
        ).to_csv(out_path, index_label="cluster")
    assert os.path.exists(out_path)


def save_cluster_distance_histogram(
    merge_dir: str,
    metric_name: str,
    distance: np.ndarray,
) -> None:
    upper_triangle = np.triu_indices(distance.shape[0], k=1)
    pair_distances = distance[upper_triangle]
    pair_distances = pair_distances[np.isfinite(pair_distances)]
    histogram_dir = ensure_merge_gmm_output_subdir(merge_dir, "histogram")
    out_path = os.path.join(
        histogram_dir,
        f"cluster_distance_{metric_name}_histogram.png",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        pair_distances, bins=100, color="#2563eb", edgecolor="#ffffff", linewidth=0.3
    )
    ax.set_xlabel("Cluster distance")
    ax.set_ylabel("Cluster pair count")
    ax.set_title(f"{metric_name} cluster distance distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    assert os.path.exists(out_path)


def compute_distance_edges(
    distance: np.ndarray,
    distance_threshold: float,
) -> pd.DataFrame:
    n_clusters = distance.shape[0]
    finite_distances = distance[np.isfinite(distance)]
    max_distance = float(finite_distances.max())
    if max_distance <= 0.0:
        max_distance = 1.0

    edge_rows = []
    for cluster_a in range(n_clusters):
        for cluster_b in range(cluster_a + 1, n_clusters):
            edge_distance = float(distance[cluster_a, cluster_b])
            edge_rows.append(
                {
                    "cluster_a": int(cluster_a),
                    "cluster_b": int(cluster_b),
                    "distance": edge_distance,
                    "distance_norm": float(np.clip(edge_distance / max_distance, 0, 1)),
                    "is_kept": edge_distance <= distance_threshold,
                }
            )
    return pd.DataFrame(
        edge_rows,
        columns=["cluster_a", "cluster_b", "distance", "distance_norm", "is_kept"],
    )


def _cluster_distance_to_color(distance_norm: float) -> str:
    anchors: tuple[tuple[int, int, int], ...] = (
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    )
    value = float(np.clip(distance_norm, 0.0, 1.0))
    scaled = value * float(len(anchors) - 1)
    left_idx = int(np.floor(scaled))
    right_idx = min(left_idx + 1, len(anchors) - 1)
    fraction = scaled - float(left_idx)
    left = np.asarray(anchors[left_idx], dtype=np.float32)
    right = np.asarray(anchors[right_idx], dtype=np.float32)
    rgb = np.round(left + fraction * (right - left)).astype(np.int32)
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _find_connected_components(
    n_nodes: int,
    edge_table: pd.DataFrame,
) -> list[list[int]]:
    adjacency = [set() for _ in range(n_nodes)]
    for _, row in edge_table.iterrows():
        cluster_a = int(row["cluster_a"])
        cluster_b = int(row["cluster_b"])
        adjacency[cluster_a].add(cluster_b)
        adjacency[cluster_b].add(cluster_a)

    components = []
    seen = np.zeros(n_nodes, dtype=bool)
    for start_node in range(n_nodes):
        if seen[start_node]:
            continue
        stack = [start_node]
        seen[start_node] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _split_component_into_cliques(
    component: Sequence[int],
    adjacency: Sequence[set[int]],
) -> list[list[int]]:
    remaining = set(int(node) for node in component)
    cliques = []
    while remaining:
        seed = max(
            remaining, key=lambda node: (len(adjacency[node] & remaining), -node)
        )
        clique = [seed]
        remaining.remove(seed)
        candidates = sorted(
            remaining,
            key=lambda node: (len(adjacency[node] & remaining), -node),
            reverse=True,
        )
        for candidate in candidates:
            if all(candidate in adjacency[node] for node in clique):
                clique.append(candidate)
                remaining.remove(candidate)
        cliques.append(sorted(clique))
    return cliques


def compute_distance_groups(
    n_clusters: int,
    edge_table: pd.DataFrame,
    dense_connected_components: bool,
) -> list[list[int]]:
    kept_edges = edge_table.loc[edge_table["is_kept"]].reset_index(drop=True)
    components = _find_connected_components(n_nodes=n_clusters, edge_table=kept_edges)
    if not dense_connected_components:
        return components

    adjacency = [set() for _ in range(n_clusters)]
    for _, row in kept_edges.iterrows():
        cluster_a = int(row["cluster_a"])
        cluster_b = int(row["cluster_b"])
        adjacency[cluster_a].add(cluster_b)
        adjacency[cluster_b].add(cluster_a)
    return [
        clique
        for component in components
        for clique in _split_component_into_cliques(component, adjacency)
    ]


def _build_kept_adjacency(
    n_clusters: int,
    edge_table: pd.DataFrame,
) -> list[dict[int, float]]:
    adjacency: list[dict[int, float]] = [dict() for _ in range(n_clusters)]
    kept_edges = edge_table.loc[edge_table["is_kept"]].reset_index(drop=True)
    for _, row in kept_edges.iterrows():
        cluster_a = int(row["cluster_a"])
        cluster_b = int(row["cluster_b"])
        edge_distance = float(row["distance"])
        adjacency[cluster_a][cluster_b] = edge_distance
        adjacency[cluster_b][cluster_a] = edge_distance
    return adjacency


def _graph_component_order_key(
    component: Sequence[int],
    adjacency: Sequence[dict[int, float]],
) -> tuple[int, int, float, int]:
    component_set = set(int(cluster) for cluster in component)
    n_edges = sum(
        1
        for cluster in component_set
        for neighbor in adjacency[int(cluster)]
        if int(cluster) < int(neighbor) and int(neighbor) in component_set
    )
    edge_distance_sum = sum(
        float(distance)
        for cluster in component_set
        for neighbor, distance in adjacency[int(cluster)].items()
        if int(cluster) < int(neighbor) and int(neighbor) in component_set
    )
    mean_edge_distance = edge_distance_sum / float(max(n_edges, 1))
    return (
        0 if n_edges > 0 else 1,
        -len(component_set),
        mean_edge_distance,
        min(component_set),
    )


def _reverse_cuthill_mckee_component_order(
    component: Sequence[int],
    adjacency: Sequence[dict[int, float]],
) -> list[int]:
    component_set = set(int(cluster) for cluster in component)
    if len(component_set) <= 1:
        return sorted(component_set)

    def degree(cluster: int) -> int:
        return len(component_set.intersection(adjacency[cluster]))

    start = min(component_set, key=lambda cluster: (degree(cluster), cluster))
    seen = {start}
    queue = [start]
    bfs_order = []
    while queue:
        cluster = queue.pop(0)
        bfs_order.append(cluster)
        neighbors = [
            neighbor
            for neighbor in adjacency[cluster]
            if neighbor in component_set and neighbor not in seen
        ]
        neighbors = sorted(
            neighbors,
            key=lambda neighbor: (
                degree(neighbor),
                float(adjacency[cluster][neighbor]),
                int(neighbor),
            ),
        )
        for neighbor in neighbors:
            seen.add(neighbor)
            queue.append(neighbor)

    remaining = sorted(component_set.difference(seen))
    return list(reversed(bfs_order)) + remaining


def compute_reordered_distance_matrix(
    distance: np.ndarray,
    cluster_sizes: np.ndarray,
    edge_table: pd.DataFrame,
    dense_connected_components: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_clusters = distance.shape[0]
    adjacency = _build_kept_adjacency(
        n_clusters=n_clusters,
        edge_table=edge_table,
    )
    groups = compute_distance_groups(
        n_clusters=n_clusters,
        edge_table=edge_table,
        dense_connected_components=dense_connected_components,
    )
    groups = sorted(
        groups,
        key=lambda group: _graph_component_order_key(
            component=group,
            adjacency=adjacency,
        ),
    )

    ordered_clusters = []
    order_rows = []
    for group_idx, group in enumerate(groups):
        group_order = _reverse_cuthill_mckee_component_order(
            component=group,
            adjacency=adjacency,
        )
        for cluster in group_order:
            order_rows.append(
                {
                    "cluster": int(cluster),
                    "distance_group": int(group_idx),
                    "position": int(len(ordered_clusters)),
                    "cluster_size": int(cluster_sizes[int(cluster)]),
                }
            )
            ordered_clusters.append(cluster)

    order = np.asarray(ordered_clusters, dtype=np.int32)
    reordered_distance = distance[np.ix_(order, order)]
    return reordered_distance, pd.DataFrame(order_rows)


def save_reordered_distance_matrix(
    merge_dir: str,
    metric_name: str,
    reordered_distance: np.ndarray,
    order_table: pd.DataFrame,
    suffix: str = "",
) -> None:
    distance_dir = ensure_merge_gmm_output_subdir(merge_dir, "distance")
    order_dir = ensure_merge_gmm_output_subdir(merge_dir, "order")
    distance_path = os.path.join(
        distance_dir,
        f"cluster_distance_{metric_name}_reordered{suffix}.csv",
    )
    order_path = os.path.join(
        order_dir,
        f"cluster_distance_{metric_name}_order{suffix}.csv",
    )
    ordered_clusters = order_table["cluster"].astype(int).tolist()
    pd.DataFrame(
        reordered_distance,
        index=ordered_clusters,
        columns=ordered_clusters,
    ).to_csv(distance_path, index_label="cluster")
    order_table.to_csv(order_path, index=False)
    assert os.path.exists(distance_path)
    assert os.path.exists(order_path)


def _load_cluster_sample_image(
    dataset: SingleCellListInferenceDataset,
    source_index: int,
) -> np.ndarray:
    image = dataset[source_index]["image"]
    image_array = prepare_three_channel_viz_image(image.squeeze()).numpy()
    assert (
        image_array.ndim == 3 and image_array.shape[2] == 3
    ), f"Expected RGB sample image, got shape {image_array.shape}"
    return image_array


def _image_array_to_data_uri(image_array: np.ndarray, max_size: int = 128) -> str:
    assert (
        image_array.dtype == np.uint8
    ), f"Expected uint8 graph node image, got dtype {image_array.dtype}"
    buffer = BytesIO()
    image = Image.fromarray(image_array)
    image.thumbnail((max_size, max_size))
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _select_cluster_sample_source_indices(
    source_indices: Sequence[int],
    gmm_labels: np.ndarray,
    db_embs_norm: torch.Tensor,
    gmm: GaussianMixture,
    n_clusters: int,
) -> dict[int, list[int | None]]:
    selected: dict[int, list[int | None]] = {}
    gmm_means = torch.as_tensor(
        gmm.means_,
        dtype=db_embs_norm.dtype,
        device=db_embs_norm.device,
    )
    for cluster_idx in range(n_clusters):
        positions = np.flatnonzero(gmm_labels == cluster_idx)
        if len(positions) == 0:
            selected[cluster_idx] = [
                None
                for _ in range(
                    CLUSTER_SAMPLE_GRID_CLOSEST_N + CLUSTER_SAMPLE_GRID_RANDOM_N
                )
            ]
            continue

        cluster_embs = db_embs_norm[positions]
        distances = (
            (cluster_embs - gmm_means[cluster_idx : cluster_idx + 1]) ** 2
        ).sum(dim=1)
        closest_local_order = torch.argsort(distances).cpu().numpy()
        closest_positions = positions[
            closest_local_order[:CLUSTER_SAMPLE_GRID_CLOSEST_N]
        ]

        rng = np.random.default_rng(CLUSTER_SAMPLE_GRID_RANDOM_SEED + cluster_idx)
        random_count = min(CLUSTER_SAMPLE_GRID_RANDOM_N, len(positions))
        random_positions = rng.choice(positions, size=random_count, replace=False)

        closest_source_indices: list[int | None] = [
            int(source_indices[int(position)]) for position in closest_positions
        ]
        random_source_indices: list[int | None] = [
            int(source_indices[int(position)]) for position in random_positions
        ]
        selected[cluster_idx] = (
            closest_source_indices
            + [None] * (CLUSTER_SAMPLE_GRID_CLOSEST_N - len(closest_source_indices))
            + random_source_indices
            + [None] * (CLUSTER_SAMPLE_GRID_RANDOM_N - len(random_source_indices))
        )
    return selected


def build_cluster_sample_image_table(
    source_indices: Sequence[int],
    gmm_labels: np.ndarray,
    db_embs_norm: torch.Tensor,
    gmm: GaussianMixture,
    dataset: SingleCellListInferenceDataset,
    n_clusters: int,
) -> dict[int, list[np.ndarray | None]]:
    sample_source_indices = _select_cluster_sample_source_indices(
        source_indices=source_indices,
        gmm_labels=gmm_labels,
        db_embs_norm=db_embs_norm,
        gmm=gmm,
        n_clusters=n_clusters,
    )
    sample_images: dict[int, list[np.ndarray | None]] = {}
    for cluster_idx, cluster_source_indices in sample_source_indices.items():
        sample_images[cluster_idx] = [
            (
                None
                if source_index is None
                else _load_cluster_sample_image(
                    dataset=dataset,
                    source_index=int(source_index),
                )
            )
            for source_index in cluster_source_indices
        ]
    return sample_images


def _resize_cluster_sample_tile(image_array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_array)
    image = image.resize(
        (CLUSTER_SAMPLE_GRID_TILE_SIZE, CLUSTER_SAMPLE_GRID_TILE_SIZE),
        resample=Image.BILINEAR,
    )
    return np.asarray(image, dtype=np.uint8)


def save_reordered_cluster_sample_grid(
    merge_dir: str,
    metric_name: str,
    ordered_clusters: Sequence[int],
    cluster_sample_images: dict[int, list[np.ndarray | None]],
    suffix: str = "",
) -> None:
    sample_dir = ensure_merge_gmm_output_subdir(merge_dir, "samples")
    out_path = os.path.join(
        sample_dir,
        f"cluster_samples_{metric_name}{suffix}.png",
    )
    n_rows = CLUSTER_SAMPLE_GRID_CLOSEST_N + CLUSTER_SAMPLE_GRID_RANDOM_N
    n_cols = len(ordered_clusters)
    tile_size = CLUSTER_SAMPLE_GRID_TILE_SIZE
    canvas = np.full(
        (n_rows * tile_size, n_cols * tile_size, 3),
        fill_value=255,
        dtype=np.uint8,
    )
    for col_idx, cluster in enumerate(ordered_clusters):
        images = cluster_sample_images[int(cluster)]
        for row_idx, image_array in enumerate(images):
            if image_array is None:
                continue
            tile = _resize_cluster_sample_tile(image_array=image_array)
            row_start = row_idx * tile_size
            col_start = col_idx * tile_size
            canvas[
                row_start : row_start + tile_size,
                col_start : col_start + tile_size,
            ] = tile
    Image.fromarray(canvas).save(out_path)
    assert os.path.exists(out_path)


def build_cluster_node_table(
    db_data: pd.DataFrame,
    source_indices: Sequence[int],
    gmm_labels: np.ndarray,
    db_embs_norm: torch.Tensor,
    dataset: SingleCellListInferenceDataset,
    n_clusters: int,
) -> pd.DataFrame:
    assignment_table = db_data[["path", "label", "patient"]].copy()
    assignment_table["source_index"] = list(source_indices)
    assignment_table["cluster_position"] = np.arange(len(assignment_table))
    assignment_table["cluster"] = gmm_labels

    rows = []
    for cluster_idx in range(n_clusters):
        group = assignment_table.loc[assignment_table["cluster"] == cluster_idx]
        if len(group) == 0:
            rows.append(
                {
                    "cluster": int(cluster_idx),
                    "representative_source_index": -1,
                    "representative_path": "",
                    "representative_label": "",
                    "representative_patient": "",
                    "cluster_size": 0,
                    "centroid_distance": float("nan"),
                    "image": "",
                }
            )
            continue

        positions = group["cluster_position"].astype(int).to_numpy()
        cluster_embs = db_embs_norm[positions]
        center = cluster_embs.mean(dim=0, keepdim=True)
        distances = ((cluster_embs - center) ** 2).sum(dim=1)
        closest_row = group.iloc[int(torch.argmin(distances).item())]
        image_array = _load_cluster_sample_image(
            dataset=dataset,
            source_index=int(closest_row["source_index"]),
        )
        rows.append(
            {
                "cluster": int(cluster_idx),
                "representative_source_index": int(closest_row["source_index"]),
                "representative_path": str(closest_row["path"]),
                "representative_label": str(closest_row["label"]),
                "representative_patient": str(closest_row["patient"]),
                "cluster_size": int(len(group)),
                "centroid_distance": float(distances.min().detach().cpu().item()),
                "image": _image_array_to_data_uri(image_array),
            }
        )
    return pd.DataFrame(rows)


def compute_graph_layout(distance: np.ndarray) -> np.ndarray:
    n_clusters = distance.shape[0]
    assert distance.shape == (
        n_clusters,
        n_clusters,
    ), f"Expected square distance matrix, got shape {distance.shape}"
    if n_clusters == 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float32)

    finite_distances = distance[np.isfinite(distance)]
    max_distance = float(finite_distances.max())
    if max_distance <= 0.0:
        angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
        return np.stack(
            [0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)],
            axis=1,
        ).astype(np.float32)

    affinity = 1.0 - np.clip(distance.astype(np.float64) / max_distance, 0.0, 1.0)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = affinity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
    laplacian = np.eye(n_clusters, dtype=np.float64) - normalized
    _, vectors = np.linalg.eigh(laplacian)
    coords = vectors[:, 1:3]
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((n_clusters, 1))], axis=1)

    coord_min = coords.min(axis=0, keepdims=True)
    coord_max = coords.max(axis=0, keepdims=True)
    coord_range = np.maximum(coord_max - coord_min, 1e-12)
    return (0.05 + 0.9 * (coords - coord_min) / coord_range).astype(np.float32)


def _compute_circular_layout(n_nodes: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)


def _orient_layout_axes(coords: np.ndarray) -> np.ndarray:
    result = coords.copy()
    for axis_idx in range(result.shape[1]):
        anchor_idx = int(np.argmax(np.abs(result[:, axis_idx])))
        if result[anchor_idx, axis_idx] < 0.0:
            result[:, axis_idx] *= -1.0
    return result


def _compute_component_distance_layout(
    distance: np.ndarray,
    component: Sequence[int],
) -> np.ndarray:
    clusters = [int(cluster) for cluster in component]
    n_nodes = len(clusters)
    if n_nodes == 1:
        return np.zeros((1, 2), dtype=np.float64)

    component_distance = distance[np.ix_(clusters, clusters)].astype(np.float64)
    assert np.isfinite(component_distance).all(), (
        "Expected finite component distances for graph layout, got non-finite values "
        f"for clusters {clusters}"
    )
    max_distance = float(component_distance.max())
    if max_distance <= 1e-12:
        return _compute_circular_layout(n_nodes=n_nodes)

    scaled_distance = component_distance / max_distance
    squared_distance = scaled_distance**2
    centering = np.eye(n_nodes, dtype=np.float64) - (
        np.ones((n_nodes, n_nodes), dtype=np.float64) / float(n_nodes)
    )
    gram = -0.5 * centering @ squared_distance @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive_order = [idx for idx in order if eigenvalues[idx] > 1e-12]
    if len(positive_order) == 0:
        return _compute_circular_layout(n_nodes=n_nodes)

    selected = positive_order[:2]
    coords = eigenvectors[:, selected] * np.sqrt(eigenvalues[selected])[None, :]
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((n_nodes, 1))], axis=1)
    return _orient_layout_axes(coords=coords)


def _normalize_local_component_layout(
    coords: np.ndarray, box_size: float
) -> np.ndarray:
    if len(coords) == 1:
        return np.zeros_like(coords)

    centered = coords - coords.mean(axis=0, keepdims=True)
    span = float(np.max(np.ptp(centered, axis=0)))
    if span <= 1e-12:
        centered = _compute_circular_layout(n_nodes=len(coords))
        span = float(np.max(np.ptp(centered, axis=0)))
    return centered * ((0.8 * box_size) / max(span, 1e-12))


def _pack_component_layouts(
    component_layouts: Sequence[tuple[list[int], np.ndarray, float]],
    n_clusters: int,
) -> np.ndarray:
    coords = np.zeros((n_clusters, 2), dtype=np.float64)
    target_row_width = np.sqrt(
        sum(box_size * box_size for _, _, box_size in component_layouts)
    )
    target_row_width = max(target_row_width, 1.0)

    rows: list[list[tuple[list[int], np.ndarray, float]]] = []
    current_row: list[tuple[list[int], np.ndarray, float]] = []
    current_width = 0.0
    for component_layout in component_layouts:
        box_size = component_layout[2]
        added_width = box_size if not current_row else box_size + LAYOUT_COMPONENT_GAP
        if current_row and current_width + added_width > target_row_width:
            rows.append(current_row)
            current_row = []
            current_width = 0.0
            added_width = box_size
        current_row.append(component_layout)
        current_width += added_width
    if current_row:
        rows.append(current_row)

    y_offset = 0.0
    packed_ranges: list[tuple[float, float, float]] = []
    for row in rows:
        row_height = max(box_size for _, _, box_size in row)
        row_width = sum(box_size for _, _, box_size in row)
        row_width += LAYOUT_COMPONENT_GAP * float(max(len(row) - 1, 0))
        packed_ranges.append((row_width, row_height, y_offset))
        y_offset += row_height + LAYOUT_COMPONENT_GAP

    total_width = max(row_width for row_width, _, _ in packed_ranges)
    for row, (row_width, row_height, row_y_offset) in zip(rows, packed_ranges):
        x_offset = 0.5 * (total_width - row_width)
        for clusters, local_coords, box_size in row:
            local = _normalize_local_component_layout(
                coords=local_coords,
                box_size=box_size,
            )
            center = np.asarray(
                [x_offset + 0.5 * box_size, row_y_offset + 0.5 * row_height],
                dtype=np.float64,
            )
            for local_idx, cluster in enumerate(clusters):
                coords[int(cluster)] = center + local[local_idx]
            x_offset += box_size + LAYOUT_COMPONENT_GAP

    coord_min = coords.min(axis=0, keepdims=True)
    coord_max = coords.max(axis=0, keepdims=True)
    coord_range = coord_max - coord_min
    normalized = np.full(coords.shape, 0.5, dtype=np.float64)
    valid_axes = (coord_range[0] > 1e-12).astype(bool)
    normalized[:, valid_axes] = (
        LAYOUT_MARGIN
        + (1.0 - 2.0 * LAYOUT_MARGIN)
        * (coords[:, valid_axes] - coord_min[:, valid_axes])
        / coord_range[:, valid_axes]
    )
    return normalized.astype(np.float32)


def compute_threshold_graph_layout(
    distance: np.ndarray,
    edge_table: pd.DataFrame,
    cluster_sizes: np.ndarray,
    dense_connected_components: bool,
) -> np.ndarray:
    n_clusters = distance.shape[0]
    assert distance.shape == (
        n_clusters,
        n_clusters,
    ), f"Expected square distance matrix, got shape {distance.shape}"
    assert len(cluster_sizes) == n_clusters, (
        f"Expected {n_clusters} cluster sizes for graph layout, got "
        f"{len(cluster_sizes)}"
    )
    if n_clusters == 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float32)

    adjacency = _build_kept_adjacency(n_clusters=n_clusters, edge_table=edge_table)
    components = compute_distance_groups(
        n_clusters=n_clusters,
        edge_table=edge_table,
        dense_connected_components=dense_connected_components,
    )
    components = sorted(
        components,
        key=lambda component: _graph_component_order_key(
            component=component,
            adjacency=adjacency,
        ),
    )

    component_layouts: list[tuple[list[int], np.ndarray, float]] = []
    for component in components:
        clusters = [int(cluster) for cluster in component]
        local_coords = _compute_component_distance_layout(
            distance=distance,
            component=clusters,
        )
        component_sizes = cluster_sizes[np.asarray(clusters, dtype=np.int32)]
        component_scale = np.sqrt(float(len(clusters)))
        node_scale = np.sqrt(float(max(int(component_sizes.max()), 1))) / 18.0
        box_size = max(component_scale + node_scale, 1.0)
        component_layouts.append((clusters, local_coords, box_size))

    return _pack_component_layouts(
        component_layouts=component_layouts,
        n_clusters=n_clusters,
    )


def _compute_graph_node_size(cluster_size: int) -> float:
    return float(
        np.clip(
            NODE_SIZE_BASE + NODE_SIZE_SCALE * np.sqrt(float(cluster_size)),
            NODE_SIZE_MIN,
            NODE_SIZE_MAX,
        )
    )


def _compute_graph_edge_length(
    edge_distance: float,
    min_distance: float,
    max_distance: float,
) -> float:
    if max_distance - min_distance <= 1e-12:
        return GRAPH_EDGE_LENGTH_MIN
    distance_norm = np.clip(
        (edge_distance - min_distance) / (max_distance - min_distance),
        0.0,
        1.0,
    )
    return float(
        GRAPH_EDGE_LENGTH_MIN
        + (GRAPH_EDGE_LENGTH_MAX - GRAPH_EDGE_LENGTH_MIN)
        * (distance_norm**GRAPH_EDGE_LENGTH_POWER)
    )


def relax_graph_layout_overlaps(node_table: pd.DataFrame) -> pd.DataFrame:
    result = node_table.copy()
    coords = result[["x", "y"]].to_numpy(dtype=np.float64)
    cluster_sizes = result["cluster_size"].astype(int).to_numpy()
    node_sizes = np.asarray(
        [_compute_graph_node_size(int(cluster_size)) for cluster_size in cluster_sizes],
        dtype=np.float64,
    )
    target_distances = np.maximum(
        LAYOUT_MIN_NODE_GAP,
        LAYOUT_NODE_GAP_SCALE
        * (node_sizes[:, None] + node_sizes[None, :])
        / POSITION_SCALE,
    )
    for _ in range(LAYOUT_SPACING_ITERATIONS):
        max_overlap = 0.0
        for node_a in range(len(coords)):
            for node_b in range(node_a + 1, len(coords)):
                delta = coords[node_a] - coords[node_b]
                distance = float(np.linalg.norm(delta))
                target_distance = float(target_distances[node_a, node_b])
                if distance >= target_distance:
                    continue
                if distance < 1e-12:
                    angle = 2.399963229728653 * float(node_a + node_b + 1)
                    direction = np.asarray([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / distance
                overlap = target_distance - distance
                shift = 0.5 * overlap * direction
                coords[node_a] += shift
                coords[node_b] -= shift
                max_overlap = max(max_overlap, overlap)
        if max_overlap < 1e-4:
            break

    coords -= coords.mean(axis=0, keepdims=True) - 0.5
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    return result


def inject_graph_controls(
    output_path: str,
    initial_distance_threshold: float,
    max_distance: float,
) -> None:
    with open(output_path) as fd:
        graph_html = fd.read()

    slider_step = max(max_distance / 200.0, 1e-6)
    controls_html = """
<style>
  #graph-control-panel {{
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 9999;
    width: 260px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d1d5db;
    border-radius: 6px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    color: #111827;
    font: 12px Arial, sans-serif;
  }}
  #graph-control-panel label {{
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 6px;
  }}
  #graph-control-panel input {{
    width: 100%;
  }}
  #graph-control-panel input[type="checkbox"] {{
    width: auto;
  }}
  .graph-control-section {{
    margin-bottom: 10px;
  }}
  .graph-control-section:last-child {{
    margin-bottom: 0;
  }}
</style>
<div id="graph-control-panel">
  <div class="graph-control-section">
    <label>
      <span>Max distance</span>
      <span id="edge-threshold-value">{initial_threshold:.4f}</span>
    </label>
    <input
      id="edge-threshold-slider"
      type="range"
      min="0"
      max="{threshold_max}"
      step="{threshold_step}"
      value="{initial_threshold}"
    />
    <div id="visible-edge-count"></div>
  </div>
  <div class="graph-control-section">
    <label>
      <span>Show edge distance</span>
      <input id="edge-label-checkbox" type="checkbox" />
    </label>
  </div>
  <div class="graph-control-section">
    <label>
      <span>Fully connected regions only</span>
      <input id="fully-connected-checkbox" type="checkbox" />
    </label>
  </div>
  <div class="graph-control-section">
    <label>
      <span>Node image size</span>
      <span id="node-size-value">1.00x</span>
    </label>
    <input
      id="node-size-slider"
      type="range"
      min="{size_min}"
      max="{size_max}"
      step="{size_step}"
      value="1.0"
    />
  </div>
</div>
<script>
(function() {{
  function setupGraphControls() {{
    if (
      typeof nodes === "undefined" ||
      typeof edges === "undefined" ||
      typeof network === "undefined" ||
      !document.getElementById("node-size-slider") ||
      !document.getElementById("edge-threshold-slider") ||
      !document.getElementById("edge-label-checkbox") ||
      !document.getElementById("fully-connected-checkbox")
    ) {{
      window.setTimeout(setupGraphControls, 100);
      return;
    }}
    var nodeSlider = document.getElementById("node-size-slider");
    var nodeValueLabel = document.getElementById("node-size-value");
    var edgeSlider = document.getElementById("edge-threshold-slider");
    var edgeValueLabel = document.getElementById("edge-threshold-value");
    var edgeLabelCheckbox = document.getElementById("edge-label-checkbox");
    var fullyConnectedCheckbox = document.getElementById("fully-connected-checkbox");
    var visibleEdgeCount = document.getElementById("visible-edge-count");
    var baseSizes = {{}};
    nodes.getIds().forEach(function(id) {{
      var node = nodes.get(id);
      baseSizes[id] = node.size || {fallback_size};
    }});
    var baseEdges = edges.get().map(function(edge) {{
      var cleanEdge = Object.assign({{}}, edge);
      delete cleanEdge.label;
      return cleanEdge;
    }});
    function freezeGraphLayout() {{
      if (freezeGraphLayout.didFreeze) {{
        return;
      }}
      freezeGraphLayout.didFreeze = true;
      network.stopSimulation();
      network.setOptions({{physics: {{enabled: false}}}});
      network.redraw();
    }}
    network.once("stabilizationIterationsDone", freezeGraphLayout);
    network.once("stabilized", freezeGraphLayout);
    function applyNodeScale() {{
      var scale = parseFloat(nodeSlider.value);
      nodeValueLabel.textContent = scale.toFixed(2) + "x";
      nodes.update(nodes.getIds().map(function(id) {{
        return {{id: id, size: baseSizes[id] * scale}};
      }}));
      network.redraw();
    }}
    function edgeKey(nodeA, nodeB) {{
      var left = String(nodeA);
      var right = String(nodeB);
      if (left < right) {{
        return left + "||" + right;
      }}
      return right + "||" + left;
    }}
    function nodeDegree(adjacency, nodeId) {{
      return Object.keys(adjacency[nodeId] || {{}}).length;
    }}
    function computeFullyConnectedEdgeIds(threshold) {{
      var adjacency = {{}};
      var edgeByPair = {{}};
      baseEdges.forEach(function(edge) {{
        var distance = parseFloat(edge.cluster_distance || 0.0);
        if (distance > threshold) {{
          return;
        }}
        var source = String(edge.from);
        var target = String(edge.to);
        adjacency[source] = adjacency[source] || {{}};
        adjacency[target] = adjacency[target] || {{}};
        adjacency[source][target] = distance;
        adjacency[target][source] = distance;
        edgeByPair[edgeKey(source, target)] = edge.id;
      }});

      var remaining = Object.keys(adjacency).sort(function(left, right) {{
        return nodeDegree(adjacency, right) - nodeDegree(adjacency, left);
      }});
      var keptEdgeIds = {{}};
      while (remaining.length > 0) {{
        var seed = remaining.shift();
        var clique = [seed];
        var candidates = remaining.slice().sort(function(left, right) {{
          return nodeDegree(adjacency, right) - nodeDegree(adjacency, left);
        }});
        candidates.forEach(function(candidate) {{
          var connectedToAllMembers = clique.every(function(member) {{
            return adjacency[candidate] && adjacency[candidate][member] !== undefined;
          }});
          if (connectedToAllMembers) {{
            clique.push(candidate);
          }}
        }});
        if (clique.length > 1) {{
          for (var i = 0; i < clique.length; i += 1) {{
            for (var j = i + 1; j < clique.length; j += 1) {{
              keptEdgeIds[edgeByPair[edgeKey(clique[i], clique[j])]] = true;
            }}
          }}
        }}
        var cliqueMembers = {{}};
        clique.forEach(function(nodeId) {{
          cliqueMembers[nodeId] = true;
        }});
        remaining = remaining.filter(function(nodeId) {{
          return !cliqueMembers[nodeId];
        }});
      }}
      return keptEdgeIds;
    }}
    function applyEdgeThreshold() {{
      var threshold = parseFloat(edgeSlider.value);
      var showLabels = edgeLabelCheckbox.checked;
      var requireFullyConnected = fullyConnectedCheckbox.checked;
      var fullyConnectedEdgeIds = requireFullyConnected
        ? computeFullyConnectedEdgeIds(threshold)
        : {{}};
      var nVisible = 0;
      edgeValueLabel.textContent = threshold.toFixed(4);
      var updates = baseEdges.map(function(edge) {{
        var updatedEdge = Object.assign({{}}, edge);
        var distance = parseFloat(edge.cluster_distance || 0.0);
        var isVisible =
          distance <= threshold &&
          (!requireFullyConnected || fullyConnectedEdgeIds[edge.id] === true);
        if (isVisible) {{
          nVisible += 1;
        }}
        updatedEdge.hidden = !isVisible;
        if (showLabels && isVisible) {{
          updatedEdge.label = edge.distance_label;
        }} else {{
          delete updatedEdge.label;
        }}
        return updatedEdge;
      }});
      edges.remove(edges.getIds());
      edges.add(updates);
      visibleEdgeCount.textContent = nVisible + " visible edges";
      network.redraw();
    }}
    nodeSlider.addEventListener("input", applyNodeScale);
    edgeSlider.addEventListener("input", applyEdgeThreshold);
    edgeLabelCheckbox.addEventListener("change", applyEdgeThreshold);
    fullyConnectedCheckbox.addEventListener("change", applyEdgeThreshold);
    applyNodeScale();
    applyEdgeThreshold();
  }}
  window.addEventListener("load", setupGraphControls);
}})();
</script>
""".format(
        initial_threshold=initial_distance_threshold,
        threshold_max=max_distance,
        threshold_step=slider_step,
        size_min=SIZE_SLIDER_MIN,
        size_max=SIZE_SLIDER_MAX,
        size_step=SIZE_SLIDER_STEP,
        fallback_size=NODE_SIZE_MIN,
    )
    assert "</body>" in graph_html, f"Could not find </body> in {output_path}"
    graph_html = graph_html.replace("</body>", controls_html + "\n</body>", 1)
    with open(output_path, "w") as fd:
        fd.write(graph_html)


def save_cluster_distance_graph(
    merge_dir: str,
    metric_name: str,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    distance_threshold: float,
    suffix: str = "",
) -> None:
    graph = Network(
        height=GRAPH_HEIGHT,
        width=GRAPH_WIDTH,
        bgcolor="#ffffff",
        font_color="#1f2937",
        notebook=False,
        cdn_resources="in_line",
    )
    graph.toggle_physics(True)
    graph.set_options("""
var options = {{
  "layout": {{
    "improvedLayout": false,
    "randomSeed": 0
  }},
  "physics": {{
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {{
      "gravitationalConstant": -90,
      "centralGravity": 0.006,
      "springLength": 130,
      "springConstant": 0.08,
      "damping": 0.42,
      "avoidOverlap": 1.0
    }},
    "stabilization": {{
      "enabled": true,
      "iterations": {stabilization_iterations},
      "updateInterval": 50,
      "fit": true
    }},
    "minVelocity": 0.2
  }}
}}
""".format(stabilization_iterations=GRAPH_PHYSICS_STABILIZATION_ITERATIONS))

    for _, row in node_table.iterrows():
        cluster = int(row["cluster"])
        cluster_size = int(row["cluster_size"])
        title = (
            f"<b>Cluster {cluster}</b><br>"
            f"Size: {cluster_size}<br>"
            f"Label: {html.escape(str(row['representative_label']))}<br>"
            f"Patient: {html.escape(str(row['representative_patient']))}<br>"
            f"Centroid distance: {float(row['centroid_distance']):.4f}<br>"
            f"Path: {html.escape(str(row['representative_path']))}"
        )
        node_kwargs: dict[str, object] = {
            "label": str(cluster),
            "title": title,
            "size": _compute_graph_node_size(cluster_size=cluster_size),
            "x": float(row["x"]) * POSITION_SCALE - 0.5 * POSITION_SCALE,
            "y": float(row["y"]) * POSITION_SCALE - 0.5 * POSITION_SCALE,
        }
        if cluster_size > 0:
            node_kwargs["shape"] = "image"
            node_kwargs["image"] = str(row["image"])
        else:
            node_kwargs["shape"] = "dot"
            node_kwargs["color"] = "#9ca3af"
        graph.add_node(cluster, **node_kwargs)

    graph_edge_table = edge_table.loc[edge_table["is_kept"]].reset_index(drop=True)
    finite_distances = graph_edge_table["distance"].astype(float).to_numpy()
    min_distance = float(finite_distances.min()) if len(finite_distances) else 0.0
    edge_length_max_distance = (
        float(finite_distances.max()) if len(finite_distances) else 1.0
    )
    max_distance = max(edge_length_max_distance, distance_threshold, 1e-12)
    for _, row in graph_edge_table.iterrows():
        cluster_a = int(row["cluster_a"])
        cluster_b = int(row["cluster_b"])
        edge_distance = float(row["distance"])
        graph.add_edge(
            cluster_a,
            cluster_b,
            id=f"{cluster_a}-{cluster_b}",
            width=EDGE_WIDTH,
            color=_cluster_distance_to_color(float(row["distance_norm"])),
            hidden=edge_distance > distance_threshold,
            cluster_distance=edge_distance,
            distance_label=f"{edge_distance:.4f}",
            length=_compute_graph_edge_length(
                edge_distance=edge_distance,
                min_distance=min_distance,
                max_distance=edge_length_max_distance,
            ),
            smooth=False,
            title=f"distance: {edge_distance:.6f}",
        )

    graph_dir = ensure_merge_gmm_output_subdir(merge_dir, "graph")
    out_path = os.path.join(graph_dir, f"cluster_graph_{metric_name}{suffix}.html")
    graph.save_graph(out_path)
    inject_graph_controls(
        output_path=out_path,
        initial_distance_threshold=distance_threshold,
        max_distance=max_distance,
    )
    assert os.path.exists(out_path)


def _as_distance_thresholds(
    distance_thresholds: DistanceThresholdSpec,
) -> list[float]:
    if isinstance(distance_thresholds, (int, float, np.integer, np.floating)):
        return [float(distance_thresholds)]

    thresholds = [float(threshold) for threshold in distance_thresholds]
    assert len(thresholds) > 0, "Expected at least one graph edge distance threshold."
    return thresholds


def _format_threshold_suffix(distance_threshold: float) -> str:
    label = f"{distance_threshold:.6f}".rstrip("0").rstrip(".")
    label = label.replace("-", "m").replace(".", "p")
    return f"_threshold_{label}"


def process_distance_metric(
    merge_dir: str,
    metric_name: str,
    gmm: GaussianMixture,
    base_node_table: pd.DataFrame,
    cluster_sizes: np.ndarray,
    cluster_sample_images: dict[int, list[np.ndarray | None]],
    distance_thresholds: Sequence[float],
    dense_connected_components: bool,
) -> None:
    distance_fn = DISTANCE_METRICS[metric_name]
    distance = distance_fn(gmm)
    save_cluster_distance_matrix(
        merge_dir=merge_dir,
        metric_name=metric_name,
        distance=distance,
    )
    save_cluster_distance_image(
        merge_dir=merge_dir,
        metric_name=metric_name,
        distance=distance,
    )
    save_cluster_similarity_matrix(
        merge_dir=merge_dir,
        metric_name=metric_name,
        distance=distance,
    )
    save_cluster_similarity_image(
        merge_dir=merge_dir,
        metric_name=metric_name,
        distance=distance,
        node_table=base_node_table,
    )
    save_cluster_distance_histogram(
        merge_dir=merge_dir,
        metric_name=metric_name,
        distance=distance,
    )

    for distance_threshold in distance_thresholds:
        threshold_suffix = (
            _format_threshold_suffix(distance_threshold=distance_threshold)
            if len(distance_thresholds) > 1
            else ""
        )
        reordered_suffix = f"_reordered{threshold_suffix}"
        edge_table = compute_distance_edges(
            distance=distance,
            distance_threshold=distance_threshold,
        )
        coords = compute_threshold_graph_layout(
            distance=distance,
            edge_table=edge_table,
            cluster_sizes=cluster_sizes,
            dense_connected_components=dense_connected_components,
        )
        node_table = base_node_table.copy()
        node_table["x"] = coords[:, 0]
        node_table["y"] = coords[:, 1]
        node_table = relax_graph_layout_overlaps(node_table=node_table)
        reordered_distance, order_table = compute_reordered_distance_matrix(
            distance=distance,
            cluster_sizes=cluster_sizes,
            edge_table=edge_table,
            dense_connected_components=dense_connected_components,
        )
        save_reordered_distance_matrix(
            merge_dir=merge_dir,
            metric_name=metric_name,
            reordered_distance=reordered_distance,
            order_table=order_table,
            suffix=threshold_suffix,
        )
        ordered_clusters = order_table["cluster"].astype(int).tolist()
        save_cluster_distance_image(
            merge_dir=merge_dir,
            metric_name=metric_name,
            distance=reordered_distance,
            suffix=reordered_suffix,
        )
        save_reordered_cluster_sample_grid(
            merge_dir=merge_dir,
            metric_name=metric_name,
            ordered_clusters=ordered_clusters,
            cluster_sample_images=cluster_sample_images,
            suffix=reordered_suffix,
        )
        save_binary_threshold_matrix(
            merge_dir=merge_dir,
            metric_name=metric_name,
            distance=reordered_distance,
            distance_threshold=distance_threshold,
            suffix=reordered_suffix,
            ordered_clusters=ordered_clusters,
        )
        save_binary_threshold_image(
            merge_dir=merge_dir,
            metric_name=metric_name,
            distance=reordered_distance,
            distance_threshold=distance_threshold,
            suffix=reordered_suffix,
            node_table=base_node_table,
            ordered_clusters=ordered_clusters,
        )
        save_cluster_similarity_matrix(
            merge_dir=merge_dir,
            metric_name=metric_name,
            distance=reordered_distance,
            suffix=reordered_suffix,
            ordered_clusters=ordered_clusters,
        )
        save_cluster_similarity_image(
            merge_dir=merge_dir,
            metric_name=metric_name,
            distance=reordered_distance,
            suffix=reordered_suffix,
            node_table=base_node_table,
            ordered_clusters=ordered_clusters,
        )
        save_cluster_distance_graph(
            merge_dir=merge_dir,
            metric_name=metric_name,
            node_table=node_table,
            edge_table=edge_table,
            distance_threshold=distance_threshold,
            suffix=threshold_suffix,
        )


def merge_gmm_pipeline(
    pred_path: str,
    out_dir: str,
    gmm_model_dir: str,
    gmm_k: int,
    dataset_config_path: str,
    cell_instances_path: str,
    label_column: str = "patch_type",
    global_sample_n: int | None = None,
    distance_metrics: Sequence[str] = (
        "cosine_mean",
        "wasserstein_2",
        "symmetric_kl",
    ),
    graph_edge_distance_thresholds: dict[str, DistanceThresholdSpec] | None = None,
    dense_connected_components: bool = False,
) -> None:
    merge_dir = ensure_merge_gmm_output_dir(out_dir)
    logger.info("Starting GMM merge pipeline")
    logger.info("Output dir: %s", out_dir)
    logger.info("GMM model dir: %s", gmm_model_dir)
    logger.info("GMM k: %s", gmm_k)
    logger.info("global_sample_n: %s", global_sample_n)
    logger.info("distance_metrics: %s", distance_metrics)
    logger.info("dense_connected_components: %s", dense_connected_components)

    unsupported_metrics = sorted(set(distance_metrics).difference(DISTANCE_METRICS))
    assert (
        not unsupported_metrics
    ), f"Unsupported distance metrics: {unsupported_metrics}"

    if graph_edge_distance_thresholds is None:
        graph_edge_distance_thresholds = {
            "cosine_mean": 0.8,
            "wasserstein_2": 1.0,
            "symmetric_kl": 50.0,
        }

    db_data = load_embedding_table(
        pred_path=pred_path,
        cell_instances_path=cell_instances_path,
        label_column=label_column,
    )
    if global_sample_n is None:
        db_global_sample = db_data
        global_source_indices = db_global_sample.index.tolist()
        logger.info("Using all %d cells for GMM merge", len(db_global_sample))
    else:
        global_sample_indices = sample_global_cells(db_data, n=global_sample_n)
        db_global_sample = db_data.loc[global_sample_indices].copy()
        global_source_indices = global_sample_indices
        logger.info(
            "Using %d globally sampled cells for GMM merge", len(db_global_sample)
        )

    dataset = build_dataset(
        config_path=dataset_config_path,
        cell_instances=cell_instances_path,
    )
    _, db_embs_norm = prepare_embeddings(db_global_sample)
    selected_gmm_k = int(gmm_k)
    gmm_labels, gmm = load_gmm_partition(
        db_embs_norm=db_embs_norm,
        gmm_model_dir=gmm_model_dir,
        k=selected_gmm_k,
    )
    cluster_sizes = np.bincount(gmm_labels, minlength=selected_gmm_k)
    base_node_table = build_cluster_node_table(
        db_data=db_global_sample,
        source_indices=global_source_indices,
        gmm_labels=gmm_labels,
        db_embs_norm=db_embs_norm,
        dataset=dataset,
        n_clusters=selected_gmm_k,
    )
    cluster_sample_images = build_cluster_sample_image_table(
        source_indices=global_source_indices,
        gmm_labels=gmm_labels,
        db_embs_norm=db_embs_norm,
        gmm=gmm,
        dataset=dataset,
        n_clusters=selected_gmm_k,
    )

    for metric_name in distance_metrics:
        process_distance_metric(
            merge_dir=merge_dir,
            metric_name=metric_name,
            gmm=gmm,
            base_node_table=base_node_table,
            cluster_sizes=cluster_sizes,
            cluster_sample_images=cluster_sample_images,
            distance_thresholds=_as_distance_thresholds(
                graph_edge_distance_thresholds[metric_name]
            ),
            dense_connected_components=dense_connected_components,
        )

    logger.info("Completed GMM merge pipeline")


def main() -> None:
    configure_logging()

    pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0/models/eval/training_124999/da946e60_Apr15-04-01-36_sd1000_INF_srhumglioma2m_dev_tune2/predictions/pred.pt"
    gmm_model_dir = "../infil/gmm_models/srhumglioma2m_b1a0cbe3"
    gmm_k = 256
    out_dir = "srhumglioma2m_gmm_merge_b1a0cbe3_k256"
    run_dir = os.path.dirname(os.path.dirname(pred_path))
    dataset_config_path = os.path.join(
        run_dir,
        "config",
        "inference_dinov2_scsrhdb.yaml",
    )
    cell_instances_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srhum_glioma_2m_.csv"
    label_key = "label"

    global_sample_n = None
    distance_metrics = "cosine_mean"  # , "wasserstein_2", "symmetric_kl")
    graph_edge_distance_thresholds = {
        "cosine_mean": [0.5],  # [0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2],
        # "wasserstein_2": [0.2, 0.4, 0.5, 0.6, 0.8],
        # "symmetric_kl": [100, 200.0, 400, 500, 600, 800],
    }
    dense_connected_components = False

    merge_gmm_pipeline(
        pred_path=pred_path,
        out_dir=out_dir,
        gmm_model_dir=gmm_model_dir,
        gmm_k=gmm_k,
        dataset_config_path=dataset_config_path,
        cell_instances_path=cell_instances_path,
        label_column=label_key,
        global_sample_n=global_sample_n,
        distance_metrics=distance_metrics,
        graph_edge_distance_thresholds=graph_edge_distance_thresholds,
        dense_connected_components=dense_connected_components,
    )


if __name__ == "__main__":
    main()

import base64
import io
import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Iterable

import altair as alt
import einops
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from ts2.data.transforms import HistologyTransform
from ts2.train.main_cell_inference import SingleCellListInferenceDataset
from ts2.utils.srh_viz import prepare_three_channel_viz_image
from ts2.utils.tailwind import TC

logger = logging.getLogger(__name__)


@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _parse_proposal(value) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            stripped = stripped[1:-1]
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Could not parse proposal coordinates from value: {value!r}")


def _encode_cell_path(patch: str, proposal) -> str:
    row, col = _parse_proposal(proposal)
    return f"{patch}#{row}_{col}"


def _extract_patient_from_patch_name(patch: str) -> str:
    patch = str(patch)
    match = re.search(r"(NIO_UM_[0-9]+)", patch)
    assert match is not None, f"Could not infer patient id from patch name: {patch}"
    return match.group(1)


def load_embedding_table(
    pred_path: str,
    cell_instances_path: str,
    label_column: str = "patch_type",
) -> pd.DataFrame:
    logger.info("Loading embedding table from %s", pred_path)

    data = torch.load(pred_path)
    table = pd.DataFrame(data)
    assert (
        "embeddings" in table.columns
    ), "Expected `embeddings` column in embedding table."
    assert "path" in table.columns, "Expected `path` column in embedding table."

    label_source = pd.read_csv(cell_instances_path, dtype=str)
    label_source = label_source.copy()
    label_source["path"] = label_source.apply(
        lambda row: _encode_cell_path(row["patch"], row["proposal"]), axis=1
    )
    if "patient" not in label_source.columns:
        label_source["patient"] = label_source["patch"].map(
            _extract_patient_from_patch_name
        )
    label_source = label_source.rename(columns={label_column: "label"})[
        ["path", "label", "patient"]
    ]
    labels_per_path = label_source.groupby("path")["label"].nunique()
    ambiguous_paths = labels_per_path[labels_per_path > 1]
    assert (
        ambiguous_paths.empty
    ), "Cell instances CSV produced paths with conflicting labels: " + ", ".join(
        ambiguous_paths.index[:10]
    )
    label_source = label_source.drop_duplicates(subset=["path", "label"])

    table = table.drop(columns=["label"], errors="ignore").merge(
        label_source,
        on="path",
        how="left",
        validate="many_to_one",
    )
    missing_label_count = int(table["label"].isna().sum())
    assert (
        missing_label_count == 0
    ), f"Could not match {missing_label_count} prediction rows to labels in source CSV."
    total_patient_count = int(table["patient"].astype(str).nunique())
    logger.info("Total patients in embedding table: %d", total_patient_count)
    logger.info("Loaded %d rows with columns: %s", len(table), table.columns.tolist())
    return table


def prepare_embeddings(db_data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("Preparing normalized embeddings for full database")
    db_embs = torch.stack(db_data["embeddings"].tolist())
    assert (
        db_embs.ndim == 2
    ), f"Expected 2D embedding tensor, got shape {tuple(db_embs.shape)}"
    logger.info("db_embs shape: %s dtype: %s", tuple(db_embs.shape), db_embs.dtype)
    db_mean = db_embs.mean(dim=0)
    db_embs = db_embs - db_mean
    db_embs_norm = torch.nn.functional.normalize(db_embs, dim=1)
    return db_mean, db_embs_norm


def normalize_embeddings(
    embeddings: torch.Tensor, db_mean: torch.Tensor
) -> torch.Tensor:
    embeddings = embeddings - db_mean
    return torch.nn.functional.normalize(embeddings, dim=1)


def sample_idx(group, n: int = 8192):
    return group.sample(n=min(n, len(group)), random_state=1000).index


def sample_cells(db_data: pd.DataFrame, n_per_class: int = 8192) -> list[int]:
    logger.info("Sampling up to %d cells per label", n_per_class)
    sampled_idx = (
        db_data.groupby("label").apply(sample_idx, n=n_per_class).explode().values
    )
    cell_samples = sorted(sampled_idx)
    assert len(cell_samples) > 0, "No sampled cells were selected."
    logger.info("Sampled %d total cells", len(cell_samples))
    return cell_samples


def sample_global_cells(db_data: pd.DataFrame, n: int | None = None) -> list[int]:
    assert n > 0, f"global sample n must be positive, got {n}"
    sampled_idx = db_data.sample(n=min(n, len(db_data)), random_state=1000).index
    cell_samples = sorted(sampled_idx)
    assert len(cell_samples) > 0, "No sampled cells were selected for global sample."
    logger.info("Sampled %d total cells for global sample", len(cell_samples))
    return cell_samples


def im_to_bytestr(image) -> str:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def build_dataset(
    config_path: str, cell_instances: str
) -> SingleCellListInferenceDataset:
    logger.info(
        "Building dataset from config=%s cell_instances=%s", config_path, cell_instances
    )
    assert os.path.exists(config_path), f"Dataset config not found: {config_path}"
    assert os.path.exists(
        cell_instances
    ), f"Cell instances file not found: {cell_instances}"
    with open(config_path) as fd:
        cf = OmegaConf.create(yaml.safe_load(fd))
    cf.data.test_dataset.params.cell_instances = cell_instances
    dataset = SingleCellListInferenceDataset(
        transform=HistologyTransform(**cf.data.xform_params),
        **cf.data.test_dataset.params,
    )
    assert len(dataset) > 0, "Dataset instantiated with zero cells."
    logger.info("Dataset size: %d", len(dataset))
    return dataset


def get_sample_images(
    dataset, cell_samples: list[int]
) -> tuple[list, list[str], np.ndarray]:
    logger.info("Loading sampled cell images for TSNE tooltips")
    images = [
        dataset[i]["image"]
        for i in tqdm(cell_samples, desc="Loading sampled cell images")
    ]
    assert len(images) == len(
        cell_samples
    ), "Mismatch between sampled cells and loaded images."
    image_arrays = [
        prepare_three_channel_viz_image(image.squeeze()).numpy() for image in images
    ]
    im_str = [im_to_bytestr(image) for image in image_arrays]
    stacked = np.stack(image_arrays)
    assert (
        stacked.ndim == 4
    ), f"Expected stacked image array to be 4D, got shape {stacked.shape}"
    logger.info(
        "Loaded %d sampled images; stacked image array shape: %s",
        len(images),
        stacked.shape,
    )
    return images, im_str, stacked


def compute_tsne(
    db_sample_embs: np.ndarray,
    perplexity: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    assert len(db_sample_embs) > 0, "TSNE received zero embeddings."
    logger.info(
        "Running TSNE once on sampled embeddings: shape=%s, perplexity=%s, random_state=%s",
        db_sample_embs.shape,
        perplexity,
        random_state,
    )
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(db_sample_embs)
    min_vals = embeddings_2d.min(axis=0)
    max_vals = embeddings_2d.max(axis=0)
    return (embeddings_2d - min_vals) / (max_vals - min_vals) * 0.9 + 0.05


def _fit_single_gmm(
    k: int,
    db_embs_np: np.ndarray,
    covariance_type: str,
    max_iter: int,
    init_params: str,
    random_state: int,
) -> tuple[GaussianMixture, float, float]:
    logger.info("Fitting GMM with K=%s", k)
    assert k > 0, f"Invalid GMM component count: {k}"
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        random_state=random_state,
    )
    gmm.fit(db_embs_np)
    bic = gmm.bic(db_embs_np)
    aic = gmm.aic(db_embs_np)
    logger.info("Finished k=%s BIC=%s AIC=%s", k, bic, aic)
    return gmm, bic, aic


def fit_gmms(
    db_embs_norm: torch.Tensor,
    k_range: Iterable[int],
    covariance_type: str = "diag",
    max_iter: int = 200,
    init_params: str = "kmeans",
    random_state: int = 0,
    n_jobs: int = 1,
    parallel_backend: str = "loky",
) -> tuple[list[GaussianMixture], list[float], list[float]]:
    k_values = list(k_range)
    assert len(k_values) > 0, "k_range must contain at least one component count."
    db_embs_np = db_embs_norm.cpu().numpy()
    logger.info(
        "Starting GMM fitting on normalized embeddings with shape=%s", db_embs_np.shape
    )
    logger.info(
        "GMM fit config: ks=%s n_jobs=%s backend=%s",
        k_values,
        n_jobs,
        parallel_backend,
    )

    if n_jobs == 1:
        results = [
            _fit_single_gmm(
                k=k,
                db_embs_np=db_embs_np,
                covariance_type=covariance_type,
                max_iter=max_iter,
                init_params=init_params,
                random_state=random_state,
            )
            for k in tqdm(k_values, desc="Fitting GMMs")
        ]
    else:
        with tqdm_joblib(tqdm(total=len(k_values), desc="Fitting GMMs")):
            results = joblib.Parallel(
                n_jobs=n_jobs,
                backend=parallel_backend,
                verbose=10,
            )(
                joblib.delayed(_fit_single_gmm)(
                    k=k,
                    db_embs_np=db_embs_np,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                    init_params=init_params,
                    random_state=random_state,
                )
                for k in k_values
            )

    gmms = [gmm for gmm, _, _ in results]
    bic_scores = [bic for _, bic, _ in results]
    aic_scores = [aic for _, _, aic in results]

    return gmms, bic_scores, aic_scores


def build_tsne_axis():
    return alt.Axis(
        tickSize=0,
        values=np.linspace(0, 1, 6),
        domain=False,
        labels=False,
        title="",
    )


def build_label_color_scale(class_names: list[str]) -> alt.Scale:
    normalized_to_name = {str(name).strip().lower(): name for name in class_names}
    if len(class_names) == 2 and {"tumor", "normal"} == set(normalized_to_name):
        domain = [normalized_to_name["tumor"], normalized_to_name["normal"]]
        return alt.Scale(domain=domain, range=TC()(c="RL", s=5))

    return alt.Scale(domain=class_names, range=TC()(nc=len(class_names), s=5))


def save_tsne_plot(
    out_dir: str,
    db_sample: pd.DataFrame,
    embeddings_2d: np.ndarray,
) -> None:
    logger.info("Saving sampled TSNE plot colored by label")
    tsne_unit_axis = build_tsne_axis()
    tsne_data = db_sample[["path", "label"]].copy().reset_index(drop=True)
    tsne_data["x"] = embeddings_2d[:, 0]
    tsne_data["y"] = embeddings_2d[:, 1]
    class_names = sorted(tsne_data["label"].astype(str).unique().tolist())

    chart = (
        alt.Chart(tsne_data)
        .mark_point(filled=True)
        .encode(
            x=alt.X("x", axis=tsne_unit_axis, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y", axis=tsne_unit_axis, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "label:N",
                scale=build_label_color_scale(class_names),
                legend=alt.Legend(title="Label"),
            ),
            tooltip=["path", "label"],
        )
        .properties(width=600, height=600)
        .interactive()
    )
    chart.save(f"{out_dir}/tsne/label_tsne.png")
    chart.save(f"{out_dir}/tsne/label_tsne.pdf")
    chart.save(f"{out_dir}/tsne/label_tsne.html")
    chart.save(f"{out_dir}/tsne/label_tsne.json")


def save_gmm_tsne_plots(
    out_dir: str,
    k_range: list[int],
    gmms: list[GaussianMixture],
    metric_data: pd.DataFrame,
    metric_embs_norm: torch.Tensor,
    tsne_data: pd.DataFrame,
    tsne_embs_norm: torch.Tensor,
    embeddings_2d: np.ndarray,
    im_str: list[str],
) -> tuple[
    list[dict[str, dict[str, float]]],
    list[list[int]],
    list[dict[str, dict[str, float]]],
]:
    cluster_class_rates = []
    cluster_size = []
    cluster_patient_rates = []
    metric_embs_np = metric_embs_norm.cpu().numpy()
    tsne_embs_np = tsne_embs_norm.cpu().numpy()

    alt.data_transformers.disable_max_rows()
    tsne_unit_axis = build_tsne_axis()
    logger.info(
        "Saving per-k TSNE visualizations using the same precomputed TSNE coordinates"
    )

    for k, gmm in tqdm(
        list(zip(k_range, gmms)),
        total=len(k_range),
        desc="Saving GMM TSNE plots",
    ):
        gmm_pred = gmm.predict(metric_embs_np)
        class_rates, sizes, patient_rates = compute_cluster_membership_stats(
            gmm_pred=gmm_pred,
            metric_data=metric_data,
            k=k,
        )
        cluster_class_rates.append(class_rates)
        cluster_size.append(sizes)
        cluster_patient_rates.append(patient_rates)
        tsne_gmm_pred = gmm.predict(tsne_embs_np)

        combined_data = pd.DataFrame(
            {
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "path": tsne_data["path"].tolist(),
                "image": im_str,
                "gt": tsne_data["label"].tolist(),
                "cluster": tsne_gmm_pred,
            }
        )

        legend_selection = alt.selection_point(fields=["label"], bind="legend")
        base_chart = (
            alt.Chart(combined_data)
            .mark_point(filled=True)
            .encode(
                x=alt.X("x", axis=tsne_unit_axis, scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y", axis=tsne_unit_axis, scale=alt.Scale(domain=[0, 1])),
                tooltip=["image", "path", "cluster", "gt"],
                color=alt.Color("cluster:N"),
                opacity=alt.condition(
                    legend_selection, alt.value(0.7), alt.value(0.05)
                ),
            )
            .add_params(legend_selection)
        )

        chart = base_chart.properties(width=600, height=600).interactive()
        chart.save(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.png")
        chart.save(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.pdf")
        chart.save(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.html")
        chart.save(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.json")

    return cluster_class_rates, cluster_size, cluster_patient_rates


def topk_random_tiebreak(g):
    g = g.sample(frac=1, random_state=None)
    ranked = g.assign(rank=g["prob"].rank(method="first", ascending=False))
    return ranked[ranked["rank"] <= 5]


def pad_with_transparency(images_rgb: np.ndarray, pad: int = 1) -> np.ndarray:
    n, h, w, _ = images_rgb.shape
    padded = np.zeros((n, h + 2 * pad, w + 2 * pad, 4), dtype=images_rgb.dtype)
    padded[:, pad:-pad, pad:-pad, :3] = images_rgb
    padded[:, pad:-pad, pad:-pad, 3] = 255
    return padded


def save_mixture_samples(
    out_dir: str,
    k_range: list[int],
    gmms: list[GaussianMixture],
    db_embs_norm: torch.Tensor,
    dataset,
    source_indices: list[int],
) -> None:
    db_embs_np = db_embs_norm.cpu().numpy()
    logger.info("Saving mixture sample grids")
    assert (
        len(source_indices) == db_embs_np.shape[0]
    ), "source_indices must match the number of sampled embeddings."

    for k, gmm in tqdm(
        list(zip(k_range, gmms)),
        total=len(k_range),
        desc="Saving mixture samples",
    ):
        gmm_pred = gmm.predict_proba(db_embs_np)
        combined_data = pd.DataFrame(
            {
                "cluster": gmm_pred.argmax(axis=1),
                "prob": gmm_pred.max(axis=1),
            }
        )
        center5 = combined_data.groupby("cluster", group_keys=False).apply(
            topk_random_tiebreak
        )
        rand5 = combined_data.groupby("cluster", group_keys=False).sample(5)
        rand5["rank"] = 100
        allsample = pd.concat([center5, rand5]).sort_values(["cluster", "rank"])

        images = [
            dataset[source_indices[i]]["image"]
            for i in tqdm(
                allsample.index.tolist(),
                desc=f"Loading mixture images for k={k}",
                leave=False,
            )
        ]
        im_array = np.stack(
            [
                prepare_three_channel_viz_image(image.squeeze()).numpy()
                for image in images
            ]
        )
        im_display = einops.rearrange(
            pad_with_transparency(im_array, pad=2),
            "(b n) h w c -> (n h) (b w) c",
            b=k,
        )
        Image.fromarray(im_display).save(
            f"{out_dir}/mixture_samples/gmm{k}_clusters.png"
        )
        assert os.path.exists(f"{out_dir}/mixture_samples/gmm{k}_clusters.png")


def save_gmm_models(
    out_dir: str, k_range: list[int], gmms: list[GaussianMixture]
) -> None:
    logger.info("Saving fitted GMM model files")
    for k, gmm in tqdm(
        list(zip(k_range, gmms)),
        total=len(k_range),
        desc="Saving GMM models",
    ):
        joblib.dump(gmm, f"{out_dir}/models/gmm_g2m_m{k}.pkl")
        assert os.path.exists(f"{out_dir}/models/gmm_g2m_m{k}.pkl")


def load_gmm_models(out_dir: str, k_range: list[int]) -> list[GaussianMixture]:
    logger.info("Loading fitted GMM model files")
    gmms = []
    for k in tqdm(k_range, total=len(k_range), desc="Loading GMM models"):
        model_path = f"{out_dir}/models/gmm_g2m_m{k}.pkl"
        assert os.path.exists(model_path), f"GMM model not found: {model_path}"
        gmms.append(joblib.load(model_path))
    return gmms


def compute_cluster_membership_stats(
    gmm_pred: np.ndarray,
    metric_data: pd.DataFrame,
    k: int,
) -> tuple[dict[str, dict[str, float]], list[int], dict[str, dict[str, float]]]:
    class_names = sorted(metric_data["label"].astype(str).unique().tolist())
    patient_names = sorted(metric_data["patient"].astype(str).unique().tolist())
    cluster_class_df = pd.crosstab(
        pd.Series(gmm_pred, name="cluster"),
        pd.Series(metric_data["label"].astype(str).to_numpy(), name="label"),
        normalize="index",
    ).reindex(index=np.arange(k), columns=class_names, fill_value=0.0)
    cluster_count_s = (
        pd.Series(gmm_pred).value_counts().reindex(np.arange(k), fill_value=0)
    )
    cluster_patient_df = pd.crosstab(
        pd.Series(gmm_pred, name="cluster"),
        pd.Series(metric_data["patient"].astype(str).to_numpy(), name="patient"),
        normalize="index",
    ).reindex(index=np.arange(k), columns=patient_names, fill_value=0.0)
    cluster_class_rates = {
        str(cluster_idx): {
            class_name: float(cluster_class_df.loc[cluster_idx, class_name])
            for class_name in class_names
        }
        for cluster_idx in range(k)
    }
    cluster_size = cluster_count_s.astype(int).tolist()
    cluster_patient_rates = {
        str(cluster_idx): {
            patient_name: float(cluster_patient_df.loc[cluster_idx, patient_name])
            for patient_name in patient_names
        }
        for cluster_idx in range(k)
    }
    return cluster_class_rates, cluster_size, cluster_patient_rates


def save_db_mean(out_dir: str, db_mean: torch.Tensor) -> None:
    logger.info("Saving db_mean tensor")
    torch.save({"db_mean": db_mean}, f"{out_dir}/stats/db_mean.pt")
    assert os.path.exists(f"{out_dir}/stats/db_mean.pt")


def save_metrics(
    out_dir: str,
    k_range: list[int],
    bic_scores: list[float],
    aic_scores: list[float],
    cluster_class_rates: list[dict[str, dict[str, float]]],
    cluster_size: list[list[int]],
    cluster_patient_rates: list[dict[str, dict[str, float]]],
) -> pd.DataFrame:
    logger.info("Saving metrics CSV")
    metrics = pd.DataFrame(
        {
            "k": k_range,
            "AIC": aic_scores,
            "BIC": bic_scores,
            "class_proportions": [
                json.dumps(class_rates) for class_rates in cluster_class_rates
            ],
            "cluster_size": [
                json.dumps(cluster_sizes) for cluster_sizes in cluster_size
            ],
            "patient_proportions": [
                json.dumps(patient_rates) for patient_rates in cluster_patient_rates
            ],
        }
    )
    metrics.to_csv(f"{out_dir}/stats/gmm_g2m_metrics.csv", index=False)
    assert os.path.exists(f"{out_dir}/stats/gmm_g2m_metrics.csv")
    return metrics


def load_serialized_metrics_value(value: str, field_name: str):
    # try:
    return json.loads(value)
    # except json.JSONDecodeError:
    #    # Backward compatibility for older CSVs written with Python repr strings.
    #    parsed = literal_eval(value)
    #    logger.warning(
    #        "Parsed non-JSON %s from metrics CSV; rewriting metrics will normalize this.",
    #        field_name,
    #    )
    #    return parsed


def class_proportions_to_df(class_proportions: dict) -> pd.DataFrame:
    if not class_proportions:
        return pd.DataFrame()

    first_value = next(iter(class_proportions.values()))
    if isinstance(first_value, dict):
        class_prop_df = pd.DataFrame.from_dict(class_proportions, orient="index")
        class_prop_df.index = pd.to_numeric(class_prop_df.index)
        class_prop_df = class_prop_df.sort_index()
        class_prop_df.index.name = "cluster"
        return class_prop_df

    class_prop_df = pd.DataFrame(class_proportions)
    class_prop_df.index = pd.to_numeric(class_prop_df.index)
    class_prop_df.index.name = "cluster"
    return class_prop_df


def save_metric_plots(
    out_dir: str,
    metrics: pd.DataFrame,
    cluster_class_rates: list[dict],
) -> None:
    logger.info("Saving summary metric plots")
    melted = pd.melt(
        metrics,
        id_vars=["k"],
        value_vars=["AIC", "BIC"],
        var_name="metric",
        value_name="val",
    )
    xaxis = alt.Axis(
        tickSize=0, values=[0, 64, 128, 192, 256], title="Number of mixtures"
    )
    yaxis = alt.Axis(tickSize=0, title="Metric (x 1.0e9)")
    (
        alt.Chart(melted)
        .mark_line(point=True)
        .transform_calculate(tval="datum.val / 1000000000")
        .encode(
            x=alt.X("k", axis=xaxis, scale=alt.Scale(domain=[0, 256])),
            y=alt.Y("tval:Q", axis=yaxis, scale=alt.Scale(zero=False)),
            color="metric:N",
        )
        .interactive()
        .save(f"{out_dir}/stats/gmm_g2m_metrics.png")
    )
    assert os.path.exists(f"{out_dir}/stats/gmm_g2m_metrics.png")

    mean_cluster_purity = [
        np.mean(np.max(class_proportions_to_df(class_rates).to_numpy(), axis=1))
        for class_rates in cluster_class_rates
    ]
    (
        alt.Chart(
            pd.DataFrame(
                {"k": metrics["k"], "mean_cluster_purity": mean_cluster_purity}
            )
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("k", axis=xaxis, scale=alt.Scale(domain=[0, 256])),
            y=alt.Y(
                "mean_cluster_purity:Q",
                axis=alt.Axis(tickSize=0, title="Mean cluster purity"),
                scale=alt.Scale(zero=False),
            ),
        )
        .interactive()
        .save(f"{out_dir}/stats/gmm_g2m_cluster_confidence.pdf")
    )
    assert os.path.exists(f"{out_dir}/stats/gmm_g2m_cluster_confidence.pdf")


def get_mpl_colormap_hex_list(cmap_name: str = "RdYlGn", n: int = 256) -> list[str]:
    cmap = matplotlib.cm.get_cmap(cmap_name, n)
    return [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]


def save_cluster_stat_plots(
    out_dir: str,
    k_range: list[int],
    include_patient_count: bool = True,
) -> None:
    logger.info("Saving per-k cluster statistics plots")
    metrics = pd.read_csv(f"{out_dir}/stats/gmm_g2m_metrics.csv")
    if include_patient_count:
        assert (
            "patient_proportions" in metrics.columns
        ), "Expected `patient_proportions` column in metrics CSV. Regenerate metrics with the updated pipeline before plotting patient membership."

    for k in tqdm(k_range, desc="Saving cluster stat plots"):
        class_proportions = load_serialized_metrics_value(
            metrics.loc[metrics["k"] == k, "class_proportions"].item(),
            field_name="class_proportions",
        )
        cluster_sizes = load_serialized_metrics_value(
            metrics.loc[metrics["k"] == k, "cluster_size"].item(),
            field_name="cluster_size",
        )
        class_prop_df = class_proportions_to_df(class_proportions)
        class_names = class_prop_df.columns.astype(str).tolist()
        is_tumor_normal_pair = len(class_names) == 2 and {
            class_name.strip().lower() for class_name in class_names
        } == {"tumor", "normal"}

        class_prop_data = class_prop_df.reset_index()
        class_prop_data = class_prop_data.melt(
            id_vars="cluster",
            var_name="label",
            value_name="proportion",
        )
        class_prop_data["cluster"] = pd.to_numeric(class_prop_data["cluster"])
        class_prop_data["proportion"] = pd.to_numeric(class_prop_data["proportion"])
        if is_tumor_normal_pair:
            label_order = {
                class_name: 0 if class_name.strip().lower() == "tumor" else 1
                for class_name in class_names
            }
            class_prop_data["stack_order"] = class_prop_data["label"].map(label_order)
        else:
            class_prop_data["stack_order"] = (
                class_prop_data.groupby("cluster")["proportion"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
        incidence_data = pd.DataFrame(
            {"cluster": np.arange(k), "incidence": np.asarray(cluster_sizes) / 1000}
        )
        incidence_data["cluster"] = pd.to_numeric(incidence_data["cluster"])
        incidence_data["incidence"] = pd.to_numeric(incidence_data["incidence"])
        if include_patient_count:
            patient_proportions = load_serialized_metrics_value(
                metrics.loc[metrics["k"] == k, "patient_proportions"].item(),
                field_name="patient_proportions",
            )
            patient_prop_df = class_proportions_to_df(patient_proportions)
            assert not patient_prop_df.empty, f"No patient proportions found for k={k}"
            patient_count_data = pd.DataFrame(
                {
                    "cluster": np.arange(k),
                    "patient_count": (patient_prop_df.to_numpy() > 0).sum(axis=1),
                }
            )
            patient_count_data["cluster"] = pd.to_numeric(patient_count_data["cluster"])
            patient_count_data["patient_count"] = pd.to_numeric(
                patient_count_data["patient_count"]
            )
            max_patient_count = int(patient_count_data["patient_count"].max())
            patient_count_axis_values = list(range(0, max_patient_count + 200, 200))

        class_prop = (
            alt.Chart(class_prop_data)
            .mark_bar(width=512 / k)
            .encode(
                x=alt.X(
                    "cluster:Q",
                    scale=alt.Scale(domain=[-0.5, k - 0.5]),
                    axis=alt.Axis(
                        tickSize=0,
                        values=np.linspace(0, k - 1, k),
                        domain=False,
                        labels=False,
                        title="",
                    ),
                ),
                y=alt.Y(
                    "proportion:Q",
                    stack="zero",
                    axis=alt.Axis(
                        tickSize=0,
                        domain=False,
                        labels=True,
                        title="Class proportion",
                    ),
                ),
                color=alt.Color(
                    "label:N",
                    scale=build_label_color_scale(class_names),
                    legend=alt.Legend(title="Class"),
                ),
                order=alt.Order("stack_order:Q", sort="ascending"),
            )
            .properties(height=60, width=800)
        )

        incidence = (
            alt.Chart(incidence_data)
            .mark_bar(width=512 / k)
            .encode(
                x=alt.X(
                    "cluster:Q",
                    scale=alt.Scale(domain=[-0.5, k - 0.5]),
                    axis=alt.Axis(
                        tickSize=0,
                        values=np.linspace(0, k - 1, k),
                        domain=False,
                        labels=True,
                        title="Cluster",
                    ),
                ),
                y=alt.Y(
                    "incidence:Q",
                    axis=alt.Axis(
                        tickSize=0,
                        domain=False,
                        labels=True,
                        title="Cells (K)",
                    ),
                ),
                color=alt.value("#475569"),
            )
            .properties(height=60, width=800)
        )

        if include_patient_count:
            patient_count = (
                alt.Chart(patient_count_data)
                .mark_bar(width=512 / k)
                .encode(
                    x=alt.X(
                        "cluster:Q",
                        scale=alt.Scale(domain=[-0.5, k - 0.5]),
                        axis=alt.Axis(
                            tickSize=0,
                            values=np.linspace(0, k - 1, k),
                            domain=False,
                            labels=True,
                            title="Cluster",
                        ),
                    ),
                    y=alt.Y(
                        "patient_count:Q",
                        axis=alt.Axis(
                            tickSize=0,
                            domain=False,
                            labels=True,
                            title="Patients",
                            values=patient_count_axis_values,
                        ),
                    ),
                    color=alt.value("#0ea5e9"),
                    tooltip=["cluster:Q", "patient_count:Q"],
                )
                .properties(height=60, width=800)
            )
            chart = (
                (class_prop & incidence & patient_count)
                .configure_axis(labelFontSize=14, titleFontSize=14)
                .configure_legend(titleFontSize=14)
                .interactive()
            )
        else:
            chart = (
                (class_prop & incidence)
                .configure_axis(labelFontSize=14, titleFontSize=14)
                .configure_legend(titleFontSize=14)
                .interactive()
            )
        chart.save(f"{out_dir}/stats/cluster{k}_stats.pdf")
        chart.save(f"{out_dir}/stats/cluster{k}_stats.png")
        assert os.path.exists(f"{out_dir}/stats/cluster{k}_stats.pdf")
        assert os.path.exists(f"{out_dir}/stats/cluster{k}_stats.png")


def ensure_output_dirs(out_dir: str) -> None:
    os.makedirs(f"{out_dir}/tsne", exist_ok=True)
    os.makedirs(f"{out_dir}/models", exist_ok=True)
    os.makedirs(f"{out_dir}/stats", exist_ok=True)
    os.makedirs(f"{out_dir}/mixture_samples", exist_ok=True)
    logger.info("Ensured output directories under %s", out_dir)


def fit_and_save_gmm_pipeline(
    pred_path: str,
    out_dir: str,
    k_range: list[int],
    dataset_config_path: str,
    cell_instances_path: str,
    label_column: str = "patch_type",
    tsne_n_per_class: int = 8192,
    global_sample_n: int | None = None,
) -> pd.DataFrame:
    ensure_output_dirs(out_dir)
    assert os.path.isdir(out_dir), f"Output directory was not created: {out_dir}"
    logger.info("Starting full GMM fit pipeline")
    logger.info("Output dir: %s", out_dir)
    logger.info("k_range: %s", k_range)
    logger.info("tsne_n_per_class: %s", tsne_n_per_class)
    logger.info("global_sample_n: %s", global_sample_n)

    db_data = load_embedding_table(
        pred_path=pred_path,
        cell_instances_path=cell_instances_path,
        label_column=label_column,
    )
    if global_sample_n is None:
        db_global_sample = db_data
        global_source_indices = db_global_sample.index.tolist()
        logger.info("Using all %d cells for global fitting set", len(db_global_sample))
    else:
        global_sample_indices = sample_global_cells(db_data, n=global_sample_n)
        db_global_sample = db_data.loc[global_sample_indices].copy()
        global_source_indices = global_sample_indices

    tsne_sample_indices = sample_cells(db_global_sample, n_per_class=tsne_n_per_class)
    db_tsne_sample = db_global_sample.loc[tsne_sample_indices].copy()
    tsne_source_indices = tsne_sample_indices
    tsne_embs = torch.stack(db_tsne_sample["embeddings"].tolist())
    logger.info("TSNE embedding matrix shape: %s", tuple(tsne_embs.shape))
    embeddings_2d = compute_tsne(tsne_embs.numpy())
    save_tsne_plot(
        out_dir=out_dir,
        db_sample=db_tsne_sample,
        embeddings_2d=embeddings_2d,
    )

    dataset = build_dataset(
        config_path=dataset_config_path,
        cell_instances=cell_instances_path,
    )
    _, im_str, _ = get_sample_images(dataset, tsne_source_indices)

    db_mean, db_embs_norm = prepare_embeddings(db_global_sample)
    tsne_embs_norm = normalize_embeddings(tsne_embs, db_mean)
    gmms, bic_scores, aic_scores = fit_gmms(
        db_embs_norm=db_embs_norm,
        k_range=k_range,
        n_jobs=4,
    )
    cluster_class_rates, cluster_size, cluster_patient_rates = save_gmm_tsne_plots(
        out_dir=out_dir,
        k_range=k_range,
        gmms=gmms,
        metric_data=db_global_sample,
        metric_embs_norm=db_embs_norm,
        tsne_data=db_tsne_sample,
        tsne_embs_norm=tsne_embs_norm,
        embeddings_2d=embeddings_2d,
        im_str=im_str,
    )
    save_gmm_models(out_dir=out_dir, k_range=k_range, gmms=gmms)
    save_mixture_samples(
        out_dir=out_dir,
        k_range=k_range,
        gmms=gmms,
        db_embs_norm=db_embs_norm,
        dataset=dataset,
        source_indices=global_source_indices,
    )
    save_db_mean(
        out_dir=out_dir,
        db_mean=db_mean,
    )
    metrics = save_metrics(
        out_dir=out_dir,
        k_range=k_range,
        bic_scores=bic_scores,
        aic_scores=aic_scores,
        cluster_class_rates=cluster_class_rates,
        cluster_size=cluster_size,
        cluster_patient_rates=cluster_patient_rates,
    )
    save_metric_plots(
        out_dir=out_dir,
        metrics=metrics,
        cluster_class_rates=cluster_class_rates,
    )
    save_cluster_stat_plots(out_dir=out_dir, k_range=k_range)
    logger.info("Completed full GMM fit pipeline")
    return metrics


def regenerate_metrics_for_existing_models(
    out_dir: str,
    pred_path: str,
    cell_instances_path: str,
    label_column: str,
) -> pd.DataFrame:
    logger.info("Regenerating metrics CSV using existing saved GMM models")
    metrics_path = f"{out_dir}/stats/gmm_g2m_metrics.csv"
    assert os.path.exists(metrics_path), f"Metrics CSV not found: {metrics_path}"
    existing_metrics = pd.read_csv(metrics_path)
    assert "k" in existing_metrics.columns, "Expected `k` column in metrics CSV."
    assert "AIC" in existing_metrics.columns, "Expected `AIC` column in metrics CSV."
    assert "BIC" in existing_metrics.columns, "Expected `BIC` column in metrics CSV."
    k_range = existing_metrics["k"].astype(int).tolist()
    assert len(k_range) > 0, "No k values found in metrics CSV."

    db_data = load_embedding_table(
        pred_path=pred_path,
        cell_instances_path=cell_instances_path,
        label_column=label_column,
    )
    _, db_embs_norm = prepare_embeddings(db_data)
    db_embs_np = db_embs_norm.cpu().numpy()
    gmms = load_gmm_models(out_dir=out_dir, k_range=k_range)

    cluster_class_rates = []
    cluster_size = []
    cluster_patient_rates = []
    for k, gmm in tqdm(
        list(zip(k_range, gmms)),
        total=len(k_range),
        desc="Regenerating cluster metrics",
    ):
        gmm_pred = gmm.predict(db_embs_np)
        class_rates, sizes, patient_rates = compute_cluster_membership_stats(
            gmm_pred=gmm_pred,
            metric_data=db_data,
            k=k,
        )
        cluster_class_rates.append(class_rates)
        cluster_size.append(sizes)
        cluster_patient_rates.append(patient_rates)

    return save_metrics(
        out_dir=out_dir,
        k_range=k_range,
        bic_scores=existing_metrics["BIC"].astype(float).tolist(),
        aic_scores=existing_metrics["AIC"].astype(float).tolist(),
        cluster_class_rates=cluster_class_rates,
        cluster_size=cluster_size,
        cluster_patient_rates=cluster_patient_rates,
    )


def main() -> None:
    logging_format_str = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        + "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_str,
    )

    # pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/dd7f97e0_Jun06-21-19-30_sd1000_INFDB_NOIN_dev/predictions/pred.pt"

    pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0/models/eval/training_124999/da946e60_Apr15-04-01-36_sd1000_INF_srhumglioma2m_dev_tune2/predictions/pred.pt"
    out_dir = "srhumglioma2m_largek_b1a0cbe3"

    # pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0/models/eval/training_124999/3b928e8e_Apr15-03-28-54_sd1000_INF_srhumglioma2m_dev_tune0/predictions/pred.pt"
    # out_dir = "srhumglioma2m_04e0bf39"

    k_range = [2, 8, 16, 24, 32, 64, 128, 256, 512, 1024]  # 
    run_dir = os.path.dirname(os.path.dirname(pred_path))
    dataset_config_path = os.path.join(
        run_dir,
        "config",
        "inference_dinov2_scsrhdb.yaml",
    )

    cell_instances_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srhum_glioma_2m_.csv"
    # cell_instances_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_1dot4m_.csv"
    label_key = "label"  # "patch_type"

    tsne_n_per_class = 2048
    global_sample_n = None

    fit_and_save_gmm_pipeline(
        pred_path=pred_path,
        out_dir=out_dir,
        k_range=k_range,
        dataset_config_path=dataset_config_path,
        cell_instances_path=cell_instances_path,
        label_column=label_key,
        tsne_n_per_class=tsne_n_per_class,
        global_sample_n=global_sample_n,
    )


def main_regenerate_cluster_stats() -> None:
    logging_format_str = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        + "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_str,
    )

    pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0/models/eval/training_124999/da946e60_Apr15-04-01-36_sd1000_INF_srhumglioma2m_dev_tune2/predictions/pred.pt"
    out_dir = "srhumglioma2m_b1a0cbe3"
    cell_instances_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srhum_glioma_2m_.csv"
    label_key = "label"
    regenerate_metrics = True
    include_patient_count = True

    metrics_path = f"{out_dir}/stats/gmm_g2m_metrics.csv"
    assert os.path.exists(metrics_path), f"Metrics CSV not found: {metrics_path}"

    if regenerate_metrics:
        metrics = regenerate_metrics_for_existing_models(
            out_dir=out_dir,
            pred_path=pred_path,
            cell_instances_path=cell_instances_path,
            label_column=label_key,
        )
    else:
        metrics = pd.read_csv(metrics_path)
    assert "k" in metrics.columns, "Expected `k` column in metrics CSV."
    k_range = metrics["k"].astype(int).tolist()
    assert len(k_range) > 0, "No k values found in metrics CSV."

    save_cluster_stat_plots(
        out_dir=out_dir,
        k_range=k_range,
        include_patient_count=include_patient_count,
    )


if __name__ == "__main__":
    main()
    # main_regenerate_cluster_stats()

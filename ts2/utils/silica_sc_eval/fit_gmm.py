import base64
import io
import json
import logging
import os
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

logger = logging.getLogger(__name__)


def load_embedding_table(pred_path: str) -> pd.DataFrame:
    logger.info("Loading embedding table from %s", pred_path)
    assert os.path.exists(pred_path), f"Embedding file not found: {pred_path}"
    data = torch.load(pred_path)
    table = pd.DataFrame(data)
    assert len(table) > 0, f"No rows loaded from embedding file: {pred_path}"
    assert (
        "embeddings" in table.columns
    ), "Expected `embeddings` column in embedding table."
    assert "label" in table.columns, "Expected `label` column in embedding table."
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


def sample_idx(group, n: int = 8192):
    return group.sample(n=min(n, len(group)), random_state=1000).index


def sample_cells(db_data: pd.DataFrame, n_per_class: int = 8192) -> list[int]:
    logger.info("Sampling up to %d cells per label for TSNE", n_per_class)
    sampled_idx = (
        db_data.groupby("label").apply(sample_idx, n=n_per_class).explode().values
    )
    cell_samples = sorted(sampled_idx)
    assert len(cell_samples) > 0, "No sampled cells were selected for TSNE."
    logger.info("Sampled %d total cells for TSNE", len(cell_samples))
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


def fit_gmms(
    db_embs_norm: torch.Tensor,
    k_range: Iterable[int],
    covariance_type: str = "diag",
    max_iter: int = 200,
    init_params: str = "kmeans",
    random_state: int = 0,
) -> tuple[list[GaussianMixture], list[float], list[float]]:
    gmms = []
    bic_scores = []
    aic_scores = []
    db_embs_np = db_embs_norm.cpu().numpy()
    logger.info(
        "Starting GMM fitting on normalized embeddings with shape=%s", db_embs_np.shape
    )

    for k in tqdm(k_range, desc="Fitting GMMs"):
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
        gmms.append(gmm)
        bic_scores.append(gmm.bic(db_embs_np))
        aic_scores.append(gmm.aic(db_embs_np))
        logger.info("Finished k=%s BIC=%s AIC=%s", k, bic_scores[-1], aic_scores[-1])

    return gmms, bic_scores, aic_scores


def build_tsne_axis():
    return alt.Axis(
        tickSize=0,
        values=np.linspace(0, 1, 6),
        domain=False,
        labels=False,
        title="",
    )


def save_gmm_tsne_plots(
    out_dir: str,
    k_range: list[int],
    gmms: list[GaussianMixture],
    db_data: pd.DataFrame,
    db_embs_norm: torch.Tensor,
    cell_samples: list[int],
    embeddings_2d: np.ndarray,
    im_str: list[str],
) -> tuple[list[list[float]], list[list[int]]]:
    cluster_positive_rates = []
    cluster_size = []
    db_embs_np = db_embs_norm.cpu().numpy()

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
        gmm_pred = gmm.predict(db_embs_np)
        gmm_pred_df = pd.DataFrame(
            {"cluster": gmm_pred, "label": db_data["label"] == "tumor"}
        )
        cluster_positive_rates.append(
            gmm_pred_df.groupby("cluster").mean()["label"].tolist()
        )
        cluster_size.append(gmm_pred_df.groupby("cluster").count()["label"].tolist())

        combined_data = pd.DataFrame(
            {
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "path": db_data.iloc[cell_samples]["path"],
                "image": im_str,
                "gt": db_data.iloc[cell_samples]["label"],
                "cluster": gmm_pred[cell_samples],
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
        assert os.path.exists(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.png")
        assert os.path.exists(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.pdf")
        assert os.path.exists(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.html")
        assert os.path.exists(f"{out_dir}/tsne/gmm_g2m_m{k}_tsne.json")

    return cluster_positive_rates, cluster_size


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
) -> None:
    db_embs_np = db_embs_norm.cpu().numpy()
    logger.info("Saving mixture sample grids")

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
            dataset[i]["image"]
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


def save_metrics_and_db_mean(
    out_dir: str,
    db_mean: torch.Tensor,
    k_range: list[int],
    bic_scores: list[float],
    aic_scores: list[float],
    cluster_positive_rates: list[list[float]],
    cluster_size: list[list[int]],
) -> pd.DataFrame:
    logger.info("Saving db_mean and metrics CSV")
    torch.save({"db_mean": db_mean}, f"{out_dir}/stats/db_mean.pt")
    metrics = pd.DataFrame(
        {
            "k": k_range,
            "AIC": aic_scores,
            "BIC": bic_scores,
            "pos_rate": cluster_positive_rates,
            "cluster_size": cluster_size,
        }
    )
    metrics.to_csv(f"{out_dir}/stats/gmm_g2m_metrics.csv", index=False)
    assert os.path.exists(f"{out_dir}/stats/db_mean.pt")
    assert os.path.exists(f"{out_dir}/stats/gmm_g2m_metrics.csv")
    return metrics


def save_metric_plots(
    out_dir: str, metrics: pd.DataFrame, cluster_positive_rates: list[list[float]]
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

    mean_conf = [
        np.abs([c - 0.5 for c in pos_rate]).mean() * 2
        for pos_rate in cluster_positive_rates
    ]
    (
        alt.Chart(pd.DataFrame({"k": metrics["k"], "mean_conf": mean_conf}))
        .mark_line(point=True)
        .encode(
            x=alt.X("k", axis=xaxis, scale=alt.Scale(domain=[0, 256])),
            y=alt.Y(
                "mean_conf:Q",
                axis=alt.Axis(tickSize=0, title="Mean mixture confidence"),
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


def save_cluster_stat_plots(out_dir: str, k_range: list[int]) -> None:
    logger.info("Saving per-k cluster statistics plots")
    metrics = pd.read_csv(f"{out_dir}/stats/gmm_g2m_metrics.csv")
    hex_colors = get_mpl_colormap_hex_list("RdYlGn", 256)

    for k in tqdm(k_range, desc="Saving cluster stat plots"):
        data = pd.DataFrame(
            {
                "pos_rate": json.loads(
                    metrics.loc[metrics["k"] == k, "pos_rate"].item()
                ),
                "incidence": json.loads(
                    metrics.loc[metrics["k"] == k, "cluster_size"].item()
                ),
            }
        )
        data["incidence"] /= 1000
        data["cluster"] = np.arange(k)

        base_chart = alt.Chart(data).mark_bar(width=512 / k)
        pos_rate = base_chart.encode(
            x=alt.X(
                "cluster",
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
                "pos_rate",
                axis=alt.Axis(
                    tickSize=0,
                    domain=False,
                    labels=True,
                    title="Tumor rate",
                ),
            ),
            color=alt.Color(
                "pos_rate:Q",
                scale=alt.Scale(domain=[1, 0], range=hex_colors),
                legend=None,
            ),
        ).properties(height=60, width=800)

        incidence = base_chart.encode(
            x=alt.X(
                "cluster",
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
                "incidence",
                axis=alt.Axis(
                    tickSize=0,
                    domain=False,
                    labels=True,
                    title="Cells (K)",
                ),
            ),
            color=alt.value("#475569"),
        ).properties(height=60, width=800)

        (
            (pos_rate & incidence)
            .configure_axis(labelFontSize=14, titleFontSize=14)
            .configure_legend(titleFontSize=14)
            .interactive()
            .save(f"{out_dir}/stats/cluster{k}_stats.pdf")
        )
        assert os.path.exists(f"{out_dir}/stats/cluster{k}_stats.pdf")


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
) -> pd.DataFrame:
    ensure_output_dirs(out_dir)
    assert os.path.isdir(out_dir), f"Output directory was not created: {out_dir}"
    logger.info("Starting full GMM fit pipeline")
    logger.info("Output dir: %s", out_dir)
    logger.info("k_range: %s", k_range)

    db_data = load_embedding_table(pred_path)

    cell_samples = sample_cells(db_data, n_per_class=8192)
    db_sample = db_data.iloc[cell_samples]
    all_embs = torch.stack(db_sample["embeddings"].tolist()).numpy()
    logger.info("Sampled embedding matrix shape for TSNE: %s", all_embs.shape)
    embeddings_2d = compute_tsne(all_embs)

    dataset = build_dataset(
        config_path=dataset_config_path,
        cell_instances=cell_instances_path,
    )
    _, im_str, _ = get_sample_images(dataset, cell_samples)

    db_mean, db_embs_norm = prepare_embeddings(db_data)
    gmms, bic_scores, aic_scores = fit_gmms(
        db_embs_norm=db_embs_norm,
        k_range=k_range,
    )
    cluster_positive_rates, cluster_size = save_gmm_tsne_plots(
        out_dir=out_dir,
        k_range=k_range,
        gmms=gmms,
        db_data=db_data,
        db_embs_norm=db_embs_norm,
        cell_samples=cell_samples,
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
    )
    metrics = save_metrics_and_db_mean(
        out_dir=out_dir,
        db_mean=db_mean,
        k_range=k_range,
        bic_scores=bic_scores,
        aic_scores=aic_scores,
        cluster_positive_rates=cluster_positive_rates,
        cluster_size=cluster_size,
    )
    save_metric_plots(
        out_dir=out_dir,
        metrics=metrics,
        cluster_positive_rates=cluster_positive_rates,
    )
    save_cluster_stat_plots(out_dir=out_dir, k_range=k_range)
    assert len(os.listdir(f"{out_dir}/models")) > 0, "No model files were written."
    assert len(os.listdir(f"{out_dir}/stats")) > 0, "No stats files were written."
    assert len(os.listdir(f"{out_dir}/tsne")) > 0, "No TSNE files were written."
    assert (
        len(os.listdir(f"{out_dir}/mixture_samples")) > 0
    ), "No mixture sample files were written."
    logger.info("Completed full GMM fit pipeline")
    return metrics


def main() -> None:
    logging_format_str = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        + "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_str,
    )

    pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/dd7f97e0_Jun06-21-19-30_sd1000_INFDB_NOIN_dev/predictions/pred.pt"
    out_dir = "silica_gmm_reproduce"
    k_range = [2, 8, 16, 24, 32, 64, 128, 256]
    dataset_config_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/train/config/chengjia/inference_dinov2_scsrhdb.yaml"
    cell_instances_path = "/nfs/turbo/umms-tocho/data/data_splits/chengjia/silica_databank/srhum_glioma_2m_.csv"

    fit_and_save_gmm_pipeline(
        pred_path=pred_path,
        out_dir=out_dir,
        k_range=k_range,
        dataset_config_path=dataset_config_path,
        cell_instances_path=cell_instances_path,
    )


if __name__ == "__main__":
    main()

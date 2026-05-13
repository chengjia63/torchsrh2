import logging
import os

import altair as alt
import einops
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from ts2.utils.silica_eval.clustering.metrics import (
    class_proportions_to_df,
    compute_cluster_membership_stats,
    load_serialized_metrics_value,
)
from ts2.utils.srh_viz import prepare_three_channel_viz_image
from ts2.utils.tailwind import TC

logger = logging.getLogger(__name__)


def build_tsne_axis() -> alt.Axis:
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


def topk_random_tiebreak(g: pd.DataFrame) -> pd.DataFrame:
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


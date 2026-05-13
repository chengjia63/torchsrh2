import logging
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from ts2.utils.silica_eval.clustering.data import (
    _encode_cell_path,
    _extract_patient_from_patch_name,
    _parse_proposal,
    build_dataset,
    get_sample_images,
    im_to_bytestr,
    load_embedding_table,
    sample_cells,
    sample_global_cells,
    sample_idx,
)
from ts2.utils.silica_eval.clustering.embeddings import (
    compute_tsne,
    normalize_embeddings,
    prepare_embeddings,
)
from ts2.utils.silica_eval.clustering.metrics import (
    class_proportions_to_df,
    compute_cluster_membership_stats,
    load_serialized_metrics_value,
    save_db_mean,
    save_metrics,
)
from ts2.utils.silica_eval.clustering.models import (
    _fit_single_gmm,
    fit_gmms,
    load_gmm_models,
    save_gmm_models,
)
from ts2.utils.silica_eval.clustering.plots import (
    build_label_color_scale,
    build_tsne_axis,
    get_mpl_colormap_hex_list,
    pad_with_transparency,
    save_cluster_stat_plots,
    save_gmm_tsne_plots,
    save_metric_plots,
    save_mixture_samples,
    save_tsne_plot,
    topk_random_tiebreak,
)
from ts2.utils.silica_eval.clustering.utils import (
    configure_logging,
    ensure_output_dirs,
)

logger = logging.getLogger(__name__)


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
    configure_logging()

    # pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/dd7f97e0_Jun06-21-19-30_sd1000_INFDB_NOIN_dev/predictions/pred.pt"

    pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0/models/eval/training_124999/da946e60_Apr15-04-01-36_sd1000_INF_srhumglioma2m_dev_tune2/predictions/pred.pt"
    out_dir = "srhumglioma2m_largek_b1a0cbe3"

    # pred_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0/models/eval/training_124999/3b928e8e_Apr15-03-28-54_sd1000_INF_srhumglioma2m_dev_tune0/predictions/pred.pt"
    # out_dir = "srhumglioma2m_04e0bf39"

    k_range = [2048]  # [2, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
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
    configure_logging()

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

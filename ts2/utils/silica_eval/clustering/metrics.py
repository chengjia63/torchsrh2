import json
import logging
import os

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


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


def load_serialized_metrics_value(value: str, field_name: str) -> object:
    return json.loads(value)


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


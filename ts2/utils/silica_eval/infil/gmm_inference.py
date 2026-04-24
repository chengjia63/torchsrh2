import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def _parse_serialized_score_value(score_value):
    if isinstance(score_value, str):
        return json.loads(score_value)
    return score_value


def _normalize_component_class_scores(
    score_value: dict[Any, dict[str, float]],
    k: int,
) -> dict[int, dict[str, float]]:
    normalized_scores = {}
    for raw_component_idx, class_scores in score_value.items():
        component_idx = int(str(raw_component_idx).strip())
        normalized_scores[component_idx] = class_scores

    missing_components = sorted(set(range(k)).difference(normalized_scores))
    assert (
        not missing_components
    ), f"Missing class proportions for components: {missing_components}"
    return normalized_scores


def gmm_inf_impl(
    gmm: GaussianMixture,
    mixture_score: np.ndarray,
    inf_embs: np.ndarray,
    mixture_hard: bool = True,
    point_hard: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Computing GMM responsibilities")
    mixture_score = np.asarray(mixture_score)
    assert (
        mixture_score.shape[0] == gmm.n_components
    ), "Mismatch between score vector and number of GMM components."

    if mixture_hard:
        mixture_score = (mixture_score > 0.5).astype(float)

    responsibilities = gmm.predict_proba(inf_embs)

    if point_hard:
        hard_assignments = np.argmax(responsibilities, axis=1)
        responsibilities = np.eye(gmm.n_components)[hard_assignments]

    pos_scores = responsibilities @ mixture_score
    return pos_scores, responsibilities


def load_db_mean(db_mean_or_path) -> torch.Tensor:
    logger.info("Loading db_mean")
    if isinstance(db_mean_or_path, str):
        assert os.path.exists(
            db_mean_or_path
        ), f"db_mean file not found: {db_mean_or_path}"
        db_mean_obj = torch.load(db_mean_or_path, weights_only=False)
        if isinstance(db_mean_obj, dict):
            assert (
                "db_mean" in db_mean_obj
            ), "Expected `db_mean` key in db_mean artifact."
            db_mean = db_mean_obj["db_mean"]
        else:
            db_mean = db_mean_obj
    else:
        db_mean = db_mean_or_path
    db_mean = torch.as_tensor(db_mean)
    assert db_mean.ndim == 1, f"Expected 1D db_mean, got shape {tuple(db_mean.shape)}"
    logger.info("Loaded db_mean with shape %s", tuple(db_mean.shape))
    return db_mean


def load_gmm(gmm_or_path) -> GaussianMixture:
    if isinstance(gmm_or_path, (str, os.PathLike)):
        logger.info("Loading GMM from %s", gmm_or_path)
        assert os.path.exists(gmm_or_path), f"GMM file not found: {gmm_or_path}"
        return joblib.load(gmm_or_path)
    return gmm_or_path


def center_and_normalize_embeddings(
    db_mean: torch.Tensor,
    inf_embs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert (
        inf_embs.ndim == 2
    ), f"Expected 2D inf_embeddings, got shape {tuple(inf_embs.shape)}"
    assert (
        db_mean.shape[0] == inf_embs.shape[1]
    ), f"db_mean dim {db_mean.shape[0]} does not match embedding dim {inf_embs.shape[1]}"
    inf_embs = inf_embs - db_mean
    inf_embs_norm = torch.nn.functional.normalize(inf_embs, dim=1)
    logger.info(
        "Normalized inference embeddings with shape %s", tuple(inf_embs_norm.shape)
    )
    return db_mean, inf_embs_norm


def load_mixture_score(
    metrics_or_path,
    k: int,
    zero_components: tuple[int, ...] = (),
    target_label: str | None = None,
    negative_label: str | None = None,
) -> np.ndarray:
    logger.info("Loading mixture score for k=%s", k)
    if isinstance(metrics_or_path, str):
        assert os.path.exists(
            metrics_or_path
        ), f"Metrics CSV not found: {metrics_or_path}"
        metrics = pd.read_csv(metrics_or_path)
    else:
        metrics = metrics_or_path
    assert "k" in metrics.columns, "Expected `k` column in metrics."
    assert (
        "class_proportions" in metrics.columns
    ), "Expected `class_proportions` column in metrics."

    metric_row = metrics.loc[metrics["k"] == k]
    if metric_row.empty:
        raise ValueError(f"No metrics row found for k={k}.")

    score_value = metric_row.iloc[0]["class_proportions"]
    score_value = _parse_serialized_score_value(score_value)

    if isinstance(score_value, dict):
        first_value = next(iter(score_value.values())) if score_value else None
        if isinstance(first_value, dict):
            component_class_scores = _normalize_component_class_scores(score_value, k)
            cluster_labels = sorted(
                {
                    label
                    for cluster_scores in component_class_scores.values()
                    for label in cluster_scores
                }
            )
            if target_label is not None and negative_label is not None:
                raise ValueError("Set only one of `target_label` or `negative_label`.")

            if negative_label is not None:
                if negative_label not in cluster_labels:
                    raise ValueError(
                        f"Requested negative_label={negative_label!r}, available labels are "
                        f"{cluster_labels}."
                    )
                mixture_score = np.asarray(
                    [
                        1.0
                        - float(component_class_scores[component_idx][negative_label])
                        for component_idx in range(k)
                    ],
                    dtype=float,
                )
            else:
                if target_label is None:
                    if "tumor" in cluster_labels:
                        target_label = "tumor"
                    else:
                        raise ValueError(
                            "Metrics contain multiclass proportions; set `target_label` or "
                            "`negative_label` to select how scores are loaded."
                        )
                mixture_score = np.asarray(
                    [
                        float(component_class_scores[component_idx][target_label])
                        for component_idx in range(k)
                    ],
                    dtype=float,
                )
        else:
            if target_label is None:
                if "tumor" in score_value:
                    target_label = "tumor"
                else:
                    raise ValueError(
                        "Metrics contain multiclass proportions; set `target_label` to "
                        "select which class score to load."
                    )
            if target_label not in score_value:
                raise ValueError(
                    f"Requested target_label={target_label!r}, available labels are "
                    f"{sorted(score_value.keys())}."
                )
            mixture_score = np.asarray(score_value[target_label], dtype=float)
    else:
        mixture_score = np.asarray(score_value, dtype=float)

    for component_idx in zero_components:
        mixture_score[component_idx] = 0.0

    assert (
        mixture_score.shape[0] == k
    ), f"Expected mixture score length {k}, got {mixture_score.shape[0]}"
    logger.info("Loaded mixture score of length %d", mixture_score.shape[0])
    return mixture_score


def instantiate_gmm(gmm_cf: dict) -> dict[str, Any]:
    logger.info("Instantiating GMM metadata")
    assert "model_path" in gmm_cf, "Expected `model_path` in gmm_cf."
    assert "db_mean_path" in gmm_cf, "Expected `db_mean_path` in gmm_cf."
    assert "metrics_path" in gmm_cf, "Expected `metrics_path` in gmm_cf."
    assert "k" in gmm_cf, "Expected `k` in gmm_cf."
    gmm = load_gmm(gmm_cf["model_path"])
    k = gmm_cf["k"]
    if gmm.n_components != k:
        raise ValueError(
            f"Requested k={k}, but loaded GMM has n_components={gmm.n_components}."
        )

    db_mean = load_db_mean(gmm_cf["db_mean_path"])
    mixture_score = load_mixture_score(
        metrics_or_path=gmm_cf["metrics_path"],
        k=k,
        zero_components=tuple(gmm_cf.get("zero_components", ())),
        target_label=gmm_cf.get("target_label"),
        negative_label=gmm_cf.get("negative_label"),
    )

    return {
        "gmm": gmm,
        "db_mean": db_mean,
        "mixture_score": mixture_score,
        "k": k,
    }


def run_gmm_inference(
    inf_embeddings,
    gmm: GaussianMixture,
    db_mean,
    mixture_score,
    mixture_hard: bool = False,
    point_hard: bool = False,
) -> dict[str, Any]:
    logger.info("Running GMM inference on %d embeddings", len(inf_embeddings))
    inf_embeddings = torch.as_tensor(inf_embeddings)
    assert len(inf_embeddings) > 0, "No inference embeddings provided."
    db_mean = load_db_mean(db_mean)

    _, inf_embs_norm = center_and_normalize_embeddings(
        db_mean=db_mean,
        inf_embs=inf_embeddings,
    )
    gmm = load_gmm(gmm)
    mixture_score = np.asarray(mixture_score, dtype=float)
    pos_scores, responsibilities = gmm_inf_impl(
        gmm=gmm,
        mixture_score=mixture_score,
        inf_embs=inf_embs_norm.cpu().numpy(),
        mixture_hard=mixture_hard,
        point_hard=point_hard,
    )
    assert responsibilities.shape[0] == len(
        inf_embeddings
    ), "Responsibility row count does not match number of inference embeddings."
    logger.info("Finished GMM inference")

    return {
        "prediction": 1.0 - pos_scores,
        "assignment": responsibilities,
        "mixture_score": mixture_score,
    }

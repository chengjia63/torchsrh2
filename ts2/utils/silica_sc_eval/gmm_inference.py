import ast
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

def gmm_inf_impl(
    gmm: GaussianMixture,
    mixture_score: np.ndarray,
    inf_embs: np.ndarray,
    mixture_hard: bool = True,
    point_hard: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    mixture_score = np.asarray(mixture_score)
    assert mixture_score.shape[0] == gmm.n_components, (
        "Mismatch between score vector and number of GMM components."
    )

    if mixture_hard:
        mixture_score = (mixture_score > 0.5).astype(float)

    responsibilities = gmm.predict_proba(inf_embs)

    if point_hard:
        hard_assignments = np.argmax(responsibilities, axis=1)
        responsibilities = np.eye(gmm.n_components)[hard_assignments]

    pos_scores = responsibilities @ mixture_score
    return pos_scores, responsibilities


def load_db_mean(db_mean_or_path) -> torch.Tensor:
    if isinstance(db_mean_or_path, str):
        db_mean_obj = torch.load(db_mean_or_path)
        if isinstance(db_mean_obj, dict):
            db_mean = db_mean_obj["db_mean"]
        else:
            db_mean = db_mean_obj
    else:
        db_mean = db_mean_or_path
    return torch.as_tensor(db_mean)


def load_gmm(gmm_or_path) -> GaussianMixture:
    if isinstance(gmm_or_path, (str, os.PathLike)):
        return joblib.load(gmm_or_path)
    return gmm_or_path


def center_and_normalize_embeddings(
    db_mean: torch.Tensor,
    inf_embs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inf_embs = inf_embs - db_mean
    inf_embs_norm = torch.nn.functional.normalize(inf_embs, dim=1)
    return db_mean, inf_embs_norm


def load_mixture_score(
    metrics_or_path,
    k: int,
    zero_components: tuple[int, ...] = (),
) -> np.ndarray:
    if isinstance(metrics_or_path, str):
        metrics = pd.read_csv(metrics_or_path)
    else:
        metrics = metrics_or_path

    metric_row = metrics.loc[metrics["k"] == k]
    if metric_row.empty:
        raise ValueError(f"No metrics row found for k={k}.")

    pos_rate = metric_row.iloc[0]["pos_rate"]
    if isinstance(pos_rate, str):
        mixture_score = np.asarray(ast.literal_eval(pos_rate), dtype=float)
    else:
        mixture_score = np.asarray(pos_rate, dtype=float)

    for component_idx in zero_components:
        mixture_score[component_idx] = 0.0

    return mixture_score


def instantiate_gmm(gmm_cf: dict) -> dict[str, Any]:
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
    inf_embeddings = torch.as_tensor(inf_embeddings)
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

    return {
        "prediction": 1.0 - pos_scores,
        "assignment": responsibilities,
        "mixture_score": mixture_score,
    }

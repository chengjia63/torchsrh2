import logging
import os
from typing import Iterable, Sequence

import joblib
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from ts2.utils.silica_eval.clustering.utils import tqdm_joblib

logger = logging.getLogger(__name__)


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
    random_state: int | Sequence[int] = 0,
    n_jobs: int = 1,
    parallel_backend: str = "loky",
) -> tuple[list[GaussianMixture], list[float], list[float]]:
    k_values = [int(k) for k in k_range]
    assert len(k_values) > 0, "k_range must contain at least one component count."
    if isinstance(random_state, int):
        random_states = [int(random_state)]
    else:
        random_states = [int(seed) for seed in random_state]
    assert len(random_states) > 0, "random_state must contain at least one seed."
    specs = [(seed, k) for seed in random_states for k in k_values]
    db_embs_np = db_embs_norm.cpu().numpy()
    logger.info(
        "Starting GMM fitting on normalized embeddings with shape=%s", db_embs_np.shape
    )
    logger.info(
        "GMM fit config: ks=%s random_states=%s n_jobs=%s backend=%s",
        k_values,
        random_states,
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
                random_state=seed,
            )
            for seed, k in tqdm(specs, desc="Fitting GMMs")
        ]
    else:
        with tqdm_joblib(tqdm(total=len(specs), desc="Fitting GMMs")):
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
                    random_state=seed,
                )
                for seed, k in specs
            )

    gmms = [gmm for gmm, _, _ in results]
    bic_scores = [bic for _, bic, _ in results]
    aic_scores = [aic for _, _, aic in results]

    return gmms, bic_scores, aic_scores


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

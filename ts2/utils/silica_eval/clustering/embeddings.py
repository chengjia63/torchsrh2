import logging

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


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


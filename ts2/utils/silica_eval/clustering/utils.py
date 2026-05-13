import logging
import os
from contextlib import contextmanager
from typing import Iterator, Protocol

import joblib

logger = logging.getLogger(__name__)


class _TqdmLike(Protocol):
    def update(self, n: int) -> object:
        ...

    def close(self) -> object:
        ...


def configure_logging() -> None:
    logging_format_str = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        + "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_str,
    )


def ensure_output_dirs(out_dir: str) -> None:
    os.makedirs(f"{out_dir}/tsne", exist_ok=True)
    os.makedirs(f"{out_dir}/models", exist_ok=True)
    os.makedirs(f"{out_dir}/stats", exist_ok=True)
    os.makedirs(f"{out_dir}/mixture_samples", exist_ok=True)
    logger.info("Ensured output directories under %s", out_dir)


def ensure_consensus_output_dirs(out_dir: str) -> None:
    ensure_output_dirs(out_dir)
    os.makedirs(f"{out_dir}/consensus", exist_ok=True)
    logger.info("Ensured consensus output directories under %s", out_dir)


@contextmanager
def tqdm_joblib(tqdm_object: _TqdmLike) -> Iterator[_TqdmLike]:
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: object, **kwargs: object) -> object:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

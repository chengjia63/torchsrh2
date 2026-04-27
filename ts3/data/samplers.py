import itertools
import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class ShardedInfiniteEpochSampler(Sampler[int]):
    """Finite epoch windows over an infinite, sharded index stream.

    This keeps Lightning's epoch-based training/checkpointing semantics while
    avoiding dataset-level modulo indexing for repeated sampling.
    """

    def __init__(
        self,
        *,
        sample_count: int,
        samples_per_epoch: int | None = None,
        repeat_factor: int | float = 1,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        advance: int = 0,
    ):
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        if repeat_factor <= 0:
            raise ValueError(f"repeat_factor must be positive, got {repeat_factor}")
        if samples_per_epoch is None:
            samples_per_epoch = int(sample_count * repeat_factor)
        if samples_per_epoch <= 0:
            raise ValueError(
                f"samples_per_epoch must be positive, got {samples_per_epoch}"
            )
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if advance < 0:
            raise ValueError(f"advance must be non-negative, got {advance}")

        self.sample_count = sample_count
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.advance = advance
        self.epoch = 0

        self._global_samples_per_epoch = (
            math.ceil(samples_per_epoch / world_size) * world_size
        )
        self._local_samples_per_epoch = self._global_samples_per_epoch // world_size

    def __iter__(self) -> Iterator[int]:
        start = self.epoch * self._global_samples_per_epoch + self.advance
        stop = (self.epoch + 1) * self._global_samples_per_epoch + self.advance
        stream = itertools.islice(self._infinite_indices(), start, stop)
        yield from itertools.islice(stream, self.rank, None, self.world_size)

    def __len__(self) -> int:
        return self._local_samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _infinite_indices(self) -> Iterator[int]:
        if not self.shuffle:
            while True:
                yield from range(self.sample_count)
            return

        generator = torch.Generator()
        permutation = 0
        while True:
            generator.manual_seed(self.seed + permutation)
            yield from torch.randperm(
                self.sample_count, generator=generator
            ).tolist()
            permutation += 1

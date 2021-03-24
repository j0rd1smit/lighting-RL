import random
from typing import Iterator

from torch.utils.data import Dataset, Sampler


class UniformSampler(Sampler):
    def __init__(
        self,
        buffer: Dataset,
        samples_per_epoch: int,
    ) -> None:
        super().__init__(None)
        self.buffer = buffer
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.samples_per_epoch):
            yield random.randint(0, len(self.buffer) - 1)

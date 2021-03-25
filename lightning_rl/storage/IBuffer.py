import abc
from typing import Union

import torch
from torch.utils.data import Dataset

from lightning_rl.environmental.SampleBatch import SampleBatch


class Buffer(abc.ABC, Dataset):
    @abc.abstractmethod
    def __getitem__(self, idxs: Union[int, torch.Tensor]) -> SampleBatch:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class DynamicBuffer(Buffer):
    @abc.abstractmethod
    def append(self, batch: SampleBatch) -> None:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass

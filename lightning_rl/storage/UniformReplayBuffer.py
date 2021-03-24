from typing import Dict, Union

import torch

from lightning_rl.storage.IBuffer import DynamicBuffer
from lightning_rl.storage.SampleBatch import SampleBatch


class UniformReplayBuffer(DynamicBuffer):
    EXCLUDED_KEYS = [SampleBatch.IDX]

    def __init__(self, capacity: int) -> None:

        self.buffer: Dict[str, torch.Tensor] = {}
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

    def append(self, batch: SampleBatch) -> None:
        if len(self.buffer) == 0:
            for k, v in batch.items():
                if k not in self.EXCLUDED_KEYS:
                    shape = (self.capacity,) + v.shape
                    self.buffer[k] = torch.zeros(shape, dtype=v.dtype)

        for k, v in batch.items():
            if k not in self.EXCLUDED_KEYS:
                self.buffer[k][self.pointer] = v

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __getitem__(self, idxs: Union[int, torch.Tensor]) -> SampleBatch:
        if isinstance(idxs, int):
            idxs = torch.tensor([idxs])

        assert isinstance(idxs, torch.Tensor)
        batch = {k: self.buffer[k][idxs] for k in self.buffer}
        batch[SampleBatch.IDX] = torch.from_numpy(idxs)

        return SampleBatch(batch)

    def __len__(self) -> int:
        return self.size

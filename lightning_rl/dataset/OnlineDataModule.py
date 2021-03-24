from typing import Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Sampler

from lightning_rl.storage.IBuffer import DynamicBuffer


class OnlineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        buffer: DynamicBuffer,
        sampler: Sampler,
        batch_size: int,
        *,
        pin_memory: bool = True,
        n_workers: int = 0,
    ) -> None:
        super().__init__()
        self.buffer = buffer
        self.sampler = sampler

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.n_workers = n_workers

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.buffer,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def make_callbacks(self) -> List[Callback]:
        return []

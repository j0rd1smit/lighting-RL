from typing import Any, Optional

import gym
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from lightning_rl.callbacks.OnlineDataCollectionCallback import (
    OnlineDataCollectionCallback,
    PostProcessFunction,
)
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.storage.IBuffer import DynamicBuffer
from lightning_rl.types import Policy


class OnlineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        buffer: Dataset,
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
            pin_memory=self.pin_memory,
        )

    def create_online_data_collection_callback(
        self,
        env: gym.Env,
        policy: Policy,
        n_samples_per_step: int,
        n_populate_steps: int = 0,
        post_process_function: Optional[PostProcessFunction] = None,
        clear_buffer_after_batch: bool = False,
    ) -> OnlineDataCollectionCallback:
        assert (
            isinstance(self.buffer, DynamicBuffer)
            and self.buffer.capacity >= n_populate_steps
        ), f"Requiest an DynamicBuffer and n_populate_steps less than or equal to buffer capacity but {type(DynamicBuffer)} and {self.buffer.capacity} >= {n_populate_steps}"
        env_loop = EnvironmentLoop(env, policy)

        return OnlineDataCollectionCallback(
            self.buffer,
            env_loop,
            n_samples_per_step=n_samples_per_step,
            n_populate_steps=n_populate_steps,
            post_process_function=post_process_function,
            clear_buffer_after_batch=clear_buffer_after_batch,
        )

from typing import Callable, List, Optional, Tuple

import gym
from gym.vector import SyncVectorEnv
from pytorch_lightning import Callback

from lightning_rl.callbacks.OnlineDataCollectionCallback import OnlineDataCollectionCallback, PostProcessFunction
from lightning_rl.dataset.OnlineDataModule import OnlineDataModule
from lightning_rl.dataset.samplers.EntireBufferSampler import EntireBufferSampler
from lightning_rl.dataset.samplers.UniformSampler import UniformSampler
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.storage.UniformReplayBuffer import UniformReplayBuffer
from lightning_rl.types import Policy

EnvBuilder = Callable[[], gym.Env]


def on_policy_dataset(
    env_builder: EnvBuilder,
    select_online_actions: Policy,
    # batch
    batch_size: int = 4000,
    # online callback
    n_envs: int = 10,
    steps_per_epoch: int = 5000,
    # post processing
    post_process_function: Optional[PostProcessFunction] = None,
) -> Tuple[OnlineDataModule, List[Callback]]:
    buffer = UniformReplayBuffer(batch_size)

    samples_per_epoch = steps_per_epoch * batch_size
    sampler = EntireBufferSampler(buffer, samples_per_epoch)

    data_module = OnlineDataModule(buffer, batch_size, sampler=sampler, pin_memory=True, n_workers=0)

    if n_envs > 1:
        online_env = SyncVectorEnv([env_builder for _ in range(n_envs)])
    else:
        online_env = env_builder()

    n_samples_per_step = batch_size
    env_loop = EnvironmentLoop(online_env, select_online_actions)

    online_step_callback = OnlineDataCollectionCallback(
        buffer,
        env_loop,
        n_samples_per_step=n_samples_per_step,
        n_populate_steps=0,
        post_process_function=post_process_function,
        clear_buffer_before_gather=True,
    )

    return data_module, [online_step_callback]


def off_policy_dataset(
    env_builder: EnvBuilder,
    select_online_actions: Policy,
    # buffer
    capacity: int = 100_000,
    # batch
    batch_size: int = 32,
    # online callback
    n_envs: int = 1,
    steps_per_epoch: int = 5000,
    n_populate_steps: int = 10000,
    # post processing
    post_process_function: Optional[PostProcessFunction] = None,
) -> Tuple[OnlineDataModule, List[Callback]]:
    buffer = UniformReplayBuffer(capacity)

    samples_per_epoch = steps_per_epoch * batch_size
    sampler = UniformSampler(buffer, samples_per_epoch)

    data_module = OnlineDataModule(buffer, batch_size, sampler=sampler, pin_memory=True, n_workers=0)

    if n_envs > 1:
        online_env = SyncVectorEnv([env_builder for _ in range(n_envs)])
    else:
        online_env = env_builder()

    n_samples_per_step = batch_size
    env_loop = EnvironmentLoop(online_env, select_online_actions)

    online_step_callback = OnlineDataCollectionCallback(
        buffer,
        env_loop,
        n_samples_per_step=n_samples_per_step,
        n_populate_steps=n_populate_steps,
        post_process_function=post_process_function,
        clear_buffer_before_gather=False,
    )

    return data_module, [online_step_callback]

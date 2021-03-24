import math
from typing import Callable, Optional

from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.storage.IBuffer import DynamicBuffer
from lightning_rl.storage.SampleBatch import SampleBatch

PostProcessFunction = Callable[[SampleBatch], SampleBatch]


class DataGatheringLoop:
    def __init__(
        self,
        buffer: DynamicBuffer,
        env_loop: EnvironmentLoop,
        n_samples_per_step: int,
        post_process_function: Optional[PostProcessFunction] = None,
    ) -> None:
        self.buffer = buffer
        self.env_loop = env_loop

        self.n_samples_per_step = n_samples_per_step
        self.post_process_function = post_process_function

    def step(self) -> None:
        batch = SampleBatch.concat_samples(
            [
                self.env_loop.step()
                for _ in range(
                    math.ceil(self.n_samples_per_step / self.env_loop.n_enviroments)
                )
            ]
        )
        batches_per_episode = batch.split_by_episode()

        for batch_per_episode in batches_per_episode:
            if self.post_process_function is not None:
                batch_per_episode = self.post_process_function(batch_per_episode)

            self.buffer.append(batch_per_episode)

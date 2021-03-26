from typing import Dict, Optional

import gym
import pytorch_lightning as pl
import torch
from gym.vector import SyncVectorEnv, VectorEnv
from torch.optim import Optimizer

from lightning_rl.callbacks.EnvironmentEvaluationCallback import (
    EnvironmentEvaluationCallback,
)
from lightning_rl.dataset.dataset_builder import off_policy_dataset
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.types import Action


class QTableModel(pl.LightningModule):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 0.1,
    ) -> None:
        super(QTableModel, self).__init__()

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.q_values = torch.ones([n_observations, n_actions])

    @property
    def training_mode(self) -> bool:
        return False

    @property
    def automatic_optimization(self) -> bool:
        return False

    def select_actions(self, x: torch.Tensor) -> Action:
        actions = torch.argmax(self.q_values[x], 1)
        return list(map(int, actions)), {}

    def select_online_actions(self, x: torch.Tensor) -> Action:
        if torch.rand([1])[0] < 0.25:
            actions = torch.randint(
                low=0, high=self.n_actions - 1, size=x.shape[:1]
            ).numpy()
            return list(map(int, actions)), {}
        else:
            return self.select_actions(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        td_errors = 0
        batch_size = len(batch[SampleBatch.OBSERVATION_NEXTS])

        for i in range(batch_size):
            obs = batch[SampleBatch.OBSERVATIONS][i][0]
            action = batch[SampleBatch.ACTIONS][i][0]
            reward = batch[SampleBatch.REWARDS][i][0]
            dones = batch[SampleBatch.DONES][i][0]
            obs_next = batch[SampleBatch.OBSERVATION_NEXTS][i][0]

            q_current = self.q_values[obs][action]
            q_next = torch.max(self.q_values[obs_next])
            q_target = reward + (1 - dones) * self.gamma * q_next

            td_error = q_target - q_current
            td_errors += td_error

            self.q_values[obs][action] = self.q_values[obs][action] + self.lr * td_error

        self.log("td_error", td_errors / batch_size, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        return [None], []


def main():
    import numpy as np

    env_name = "Taxi-v3"
    # env = gym.make(env_name)
    env_eval = gym.make(env_name)
    env = SyncVectorEnv([lambda: gym.make("Taxi-v3") for _ in range(5)])

    n_observations = (
        env.observation_space.n
        if not isinstance(env, VectorEnv)
        else env.envs[0].observation_space.n
    )
    n_actions = (
        env.action_space.n if not isinstance(env, VectorEnv) else env.action_space[0].n
    )
    model = QTableModel(
        n_observations=n_observations,
        n_actions=n_actions,
    )

    data_module = off_policy_dataset(capacity=100, batch_size=10, steps_per_epoch=100)
    online_step_callback = data_module.create_online_data_collection_callback(
        env,
        model.select_online_actions,
        n_samples_per_step=10,
        n_populate_steps=100,
    )
    env_loop = EnvironmentLoop(env_eval, model.select_actions)
    eval_callback = EnvironmentEvaluationCallback(env_loop)

    trainer = pl.Trainer(
        gpus=0,
        fast_dev_run=False,
        max_epochs=250,
        callbacks=[online_step_callback, eval_callback],
        logger=False,
        checkpoint_callback=False,
    )

    trainer.fit(model, data_module)

    env = gym.make(env_name)
    o = env.reset()
    d = False
    r_total = 0
    while not d:
        a, _ = model.select_actions(np.expand_dims(np.array(o), 0))
        o, r, d, _ = env.step(a[0])
        env.render()
        r_total += r

    print(r_total)


if __name__ == "__main__":
    main()

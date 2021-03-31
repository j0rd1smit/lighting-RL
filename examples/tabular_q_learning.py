from typing import Dict, Optional

import gym
import pytorch_lightning as pl
import torch
from gym.vector import VectorEnv
from torch.optim import Optimizer

from lightning_rl.builders import on_policy_dataset
from lightning_rl.callbacks.EnvironmentEvaluationCallback import EnvironmentEvaluationCallback
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
    def automatic_optimization(self) -> bool:
        return False

    def select_actions(self, x: torch.Tensor) -> Action:
        actions = torch.argmax(self.q_values[x], 1)
        return actions

    def select_online_actions(self, x: torch.Tensor) -> Action:
        if torch.rand([1])[0] < 0.25:
            actions = torch.randint(low=0, high=self.n_actions - 1, size=x.shape[:1])
            return actions
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
            obs = batch[SampleBatch.OBSERVATIONS][i]
            action = batch[SampleBatch.ACTIONS][i]
            reward = batch[SampleBatch.REWARDS][i]
            dones = batch[SampleBatch.DONES][i]
            obs_next = batch[SampleBatch.OBSERVATION_NEXTS][i]

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
    env_name = "Taxi-v3"
    env = gym.make(env_name)

    n_observations = env.observation_space.n if not isinstance(env, VectorEnv) else env.envs[0].observation_space.n
    n_actions = env.action_space.n if not isinstance(env, VectorEnv) else env.action_space[0].n
    model = QTableModel(
        n_observations=n_observations,
        n_actions=n_actions,
    )

    """
    data_module, callbacks = off_policy_dataset(
        lambda: gym.make(env_name),
        model.select_online_actions,
        capacity=100,
        n_populate_steps=100,
        steps_per_epoch=250,
    )
    """
    data_module, callbacks = on_policy_dataset(
        lambda: gym.make(env_name),
        model.select_online_actions,
        batch_size=32,
        steps_per_epoch=250,
    )

    env_loop = EnvironmentLoop(env, model.select_actions)
    eval_callback = EnvironmentEvaluationCallback(env_loop)

    trainer = pl.Trainer(
        gpus=0,
        fast_dev_run=False,
        max_epochs=50,
        callbacks=callbacks + [eval_callback],
        logger=False,
        checkpoint_callback=False,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

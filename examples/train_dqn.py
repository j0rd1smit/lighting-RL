import copy
from typing import Dict, Optional

import gym
import pytorch_lightning as pl
import torch
from gym.vector import VectorEnv
from pytorch_lightning.core.decorators import auto_move_data
from torch.optim import Optimizer

from lightning_rl.callbacks.EnvironmentEvaluationCallback import (
    EnvironmentEvaluationCallback,
)
from lightning_rl.dataset.dataset_builder import off_policy_dataset
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.types import Action


class DQNModel(pl.LightningModule):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 0.1,  # TODO
    ) -> None:
        super(DQNModel, self).__init__()
        self.save_hyperparameters()

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr

        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_actions),
        )
        self.network_target = copy.deepcopy(self.network)

    @auto_move_data
    def select_actions(self, x: torch.Tensor) -> Action:
        pass

    @auto_move_data
    def select_online_actions(self, x: torch.Tensor) -> Action:
        print(x)
        if torch.rand([1])[0] < 0.25:
            pass

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
        return [optimizer], []


def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env_eval = gym.make(env_name)
    # env = SyncVectorEnv([lambda: gym.make("Taxi-v3") for _ in range(5)])

    n_observations = (
        env.observation_space.shape[0] if not isinstance(env, VectorEnv) else None
    )
    n_actions = env.action_space.n if not isinstance(env, VectorEnv) else None
    model = DQNModel(
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
        fast_dev_run=True,
        max_epochs=250,
        callbacks=[online_step_callback, eval_callback],
        logger=False,
        checkpoint_callback=False,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

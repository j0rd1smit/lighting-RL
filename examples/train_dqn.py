import copy
from typing import Dict, Optional

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data
from torch.optim import Optimizer

from lightning_rl.callbacks.EnvironmentEvaluationCallback import EnvironmentEvaluationCallback
from lightning_rl.dataset.dataset_builder import off_policy_dataset
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.types import Action


class DQNModel(pl.LightningModule):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 0.00025,
        eps_frames: int = 25_000,
        eps_start: float = 1.0,
        eps_min: float = 0.1,
        sync_rate: int = 1000,
    ) -> None:
        super(DQNModel, self).__init__()
        self.save_hyperparameters()

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_actions),
        )
        self.target = copy.deepcopy(self.network)

        self.eps = torch.nn.parameter.Parameter(torch.tensor([eps_start], dtype=torch.float32), requires_grad=False)

    @auto_move_data
    def select_actions(self, x: torch.Tensor) -> Action:
        q_values = self.network(x)

        return torch.argmax(q_values, 1), {}

    @auto_move_data
    def select_online_actions(self, x: torch.Tensor) -> Action:
        if np.random.random() <= self.eps:
            actions = torch.randint(low=0, high=self.n_actions - 1, size=x.shape[:1])

            return actions, {}

        return self.select_actions(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        q_values = self.network(batch[SampleBatch.OBSERVATIONS])
        q_values = torch.gather(q_values, -1, batch[SampleBatch.ACTIONS].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_values_next = self.network(batch[SampleBatch.OBSERVATION_NEXTS])
            q_values_targets = self.target(batch[SampleBatch.OBSERVATION_NEXTS])

            selected_action = torch.argmax(q_values_next, -1, keepdim=True)
            q_next = torch.gather(q_values_targets, -1, selected_action).squeeze(-1)

            rewards = batch[SampleBatch.REWARDS]
            dones = batch[SampleBatch.DONES]

            assert same_or_broadcastable(q_next.shape, rewards.shape), f"{q_next.shape} != {rewards.shape}"
            q_target = rewards + (1.0 - dones) * self.hparams.gamma * q_next
            q_target = q_target.detach()

        assert q_values.shape == q_target.shape
        td_error = q_values - q_target
        loss = F.smooth_l1_loss(q_values, q_target).mean()

        with torch.no_grad():
            self.eps.data = torch.clamp(self.eps - 1 / self.hparams.eps_frames, self.hparams.eps_min)
            self.log("DQN/eps", self.eps, prog_bar=True, on_epoch=True, on_step=False)

        self.log("DQN/td_error", torch.mean(td_error), prog_bar=False, on_epoch=True, on_step=False)
        self.log("DQN/loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("DQN/q_values/mean", torch.mean(q_values), prog_bar=False, on_epoch=True, on_step=False)
        self.log("DQN/q_values/max", torch.max(q_values), prog_bar=False, on_epoch=True, on_step=False)
        self.log("DQN/q_values/min", torch.min(q_values), prog_bar=False, on_epoch=True, on_step=False)

        if self.global_step % self.hparams.sync_rate == 0:
            self.target.load_state_dict(self.network.state_dict())

        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
        return [optimizer], []


def same_or_broadcastable(one, other):
    assert isinstance(one, tuple) and isinstance(
        other, tuple
    ), f"Expected tuple input but got {type(one)} and {type(other)}"
    if one == other:
        return True

    if one[-1] == 1 and one[:-1] == other[:-1]:
        return True

    if other[-1] == 1 and one[:-1] == other[:-1]:
        return True

    return False


def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = DQNModel(
        n_observations=n_observations,
        n_actions=n_actions,
    )

    data_module, callbacks = off_policy_dataset(
        lambda: gym.make(env_name),
        model.select_online_actions,
        capacity=100_000,
        n_populate_steps=10_000,
        steps_per_epoch=1000,
        batch_size=32,
    )

    env_loop = EnvironmentLoop(env, model.select_actions)
    eval_callback = EnvironmentEvaluationCallback(env_loop)

    trainer = pl.Trainer(
        gpus=0,
        fast_dev_run=False,
        max_epochs=100,
        callbacks=callbacks + [eval_callback],
        logger=False,
        checkpoint_callback=False,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

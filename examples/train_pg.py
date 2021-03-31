from typing import Dict, Optional

import gym
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data
from torch.distributions import Categorical
from torch.optim import Optimizer

from lightning_rl.builders import on_policy_dataset
from lightning_rl.callbacks.EnvironmentEvaluationCallback import EnvironmentEvaluationCallback
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.post_processing_utils import Postprocessing, compute_advantages
from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.types import Action


class PGModel(pl.LightningModule):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 0.00025,
        use_value_network: bool = False,
    ) -> None:
        super(PGModel, self).__init__()
        self.save_hyperparameters()

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
        )

        if use_value_network:
            self.value_network = torch.nn.Sequential(
                torch.nn.Linear(n_observations, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )
        else:
            self.value_network = None

    @property
    def automatic_optimization(self) -> bool:
        return True

    @auto_move_data
    def select_actions(self, x: torch.Tensor) -> Action:
        pi_features = self.network(x)

        return torch.argmax(pi_features, 1)

    @auto_move_data
    def select_online_actions(self, x: torch.Tensor) -> Action:
        pi_features = self.network(x)
        dist = Categorical(logits=pi_features)

        return dist.sample()

    @auto_move_data
    def agent_info(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        info = {}
        if self.hparams.use_value_network:
            with torch.no_grad():
                info[SampleBatch.VALE_PREDICTIONS] = (
                    self.value_network(batch[SampleBatch.OBSERVATION_NEXTS]).squeeze().cpu()
                )

        return info

    def post_process_function(self, batch: SampleBatch) -> SampleBatch:
        if float(batch[SampleBatch.DONES][-1]) == 0.0 and self.hparams.use_value_network:
            with torch.no_grad():
                last_r = float(
                    self.value_network(batch[SampleBatch.OBSERVATION_NEXTS][[-1]].to(self.device)).squeeze().cpu()
                )
        else:
            last_r = 0.0

        return compute_advantages(batch, last_r=last_r, gamma=0.99, use_critic=self.hparams.use_value_network)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if optimizer_idx == 0 or optimizer_idx == None:
            logits = self.network(batch[SampleBatch.OBSERVATIONS])
            dist = Categorical(logits=logits)

            log_pi = dist.log_prob(batch[SampleBatch.ACTIONS])
            assert log_pi.shape == batch[Postprocessing.ADVANTAGES].shape

            return -torch.mean(log_pi * batch[Postprocessing.ADVANTAGES])
        else:
            pred_values = self.value_network(batch[SampleBatch.OBSERVATION_NEXTS])
            target_values = batch[Postprocessing.VALUE_TARGETS].unsqueeze(-1)

            return F.smooth_l1_loss(pred_values, target_values).mean()

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
        if self.hparams.use_value_network:
            optimizer_v = torch.optim.Adam(self.value_network.parameters(), lr=self.hparams.lr)
            return [optimizer, optimizer_v], []
        else:
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
    # env_name = "MountainCar-v0"
    env = gym.make(env_name)

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = PGModel(
        n_observations=n_observations,
        n_actions=n_actions,
    )

    data_module, callbacks = on_policy_dataset(
        lambda: gym.make(env_name),
        model.select_online_actions,
        batch_size=4000,
        steps_per_epoch=25,
        n_envs=10,
        post_process_function=model.post_process_function,
        fetch_agent_info=model.agent_info,
    )

    env_loop = EnvironmentLoop(env, model.select_actions)
    eval_callback = EnvironmentEvaluationCallback(env_loop)

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=False,
        max_epochs=100,
        callbacks=callbacks + [eval_callback],
        logger=False,
        checkpoint_callback=False,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

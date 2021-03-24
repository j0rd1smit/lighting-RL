import gym
import numpy as np
import torch
from gym.vector import AsyncVectorEnv, VectorEnv

from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.types import ActionAgentInfoTuple, Observation, Policy, Seed


class EnvironmentLoop:
    def __init__(self, env: gym.Env, policy: Policy) -> None:
        self.env = env
        self.policy = policy

        self._obs = self.env.reset()

        self._is_vectorized = isinstance(
            env,
            VectorEnv,
        )
        self._done = not self._is_vectorized
        self._episode_ids = np.arange(self.n_enviroments)

        assert not isinstance(env, AsyncVectorEnv), "Async is not supported."

    @property
    def n_enviroments(self) -> int:
        if not self._is_vectorized:
            return 1
        return self.env.observation_space.shape[0]

    def seed(self, seed: Seed = None) -> None:
        self.env.seed(seed)

    def reset(self) -> None:
        self._obs = self.env.reset()
        self._done = False

    def step(self) -> SampleBatch:
        return self._step(self.policy)

    def _step(self, policy: Policy) -> SampleBatch:
        if not self._is_vectorized and self._done:
            self.reset()

        obs = self._obs
        action, agent_info = policy(obs)
        assert isinstance(action, np.ndarray) or isinstance(action, float)

        obs_next, r, d, _ = self.env.step(action)

        batch = {
            SampleBatch.OBSERVATIONS: self._cast_obs(obs),
            SampleBatch.ACTIONS: torch.tensor(action),
            SampleBatch.REWARDS: torch.tensor(r, dtype=torch.float32),
            SampleBatch.DONES: torch.tensor(d, dtype=torch.float32),
            SampleBatch.OBSERVATION_NEXTS: self._cast_obs(obs_next),
            SampleBatch.EPS_ID: torch.from_numpy(self._episode_ids.copy()),
            **agent_info,
        }

        for k, v in batch.items():
            batch[k] = self._batch_if_needed(v)

        if not self._is_vectorized:
            self._done = d
        elif np.any(d):
            self._update_episodes_ids_if_needed(d)

        return SampleBatch(batch)

    def _cast_obs(self, obs: Observation) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            if obs.dtype in [np.float64]:
                obs = obs.astype(np.float32)

            return torch.from_numpy(obs)

        return torch.tensor(obs)

    def _batch_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_vectorized:
            x = torch.unsqueeze(x, 0)

        return x

    def _update_episodes_ids_if_needed(self, d: np.ndarray) -> None:
        for i in range(self.n_enviroments):
            if d[i]:
                self._episode_ids[i] = self._episode_ids.max() + 1

    def sample(self) -> SampleBatch:
        return self._step(self._sample_policy)

    def _sample_policy(self, _: Observation) -> ActionAgentInfoTuple:
        return self.env.action_space.sample(), {}

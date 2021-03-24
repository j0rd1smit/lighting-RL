import unittest
from unittest import mock

import gym
import torch
from gym.vector import SyncVectorEnv
from parameterized import parameterized

from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.SampleBatch import SampleBatch
from tests.environmental.SampleBatchTest import assert_sample_batch


class EnvironmentLoopTest(unittest.TestCase):
    def test_n_environment_is_correct_for_non_vector_env(self):
        env = self._create_non_vector_env()
        loop = EnvironmentLoop(env, self._create_policy(env))

        self.assertEqual(loop.n_enviroments, 1)

    def _create_non_vector_env(self):
        return gym.make("CartPole-v1")

    def _create_policy(self, env, agent_info=None):
        def policy(_):
            return (
                env.action_space.sample(),
                agent_info if agent_info is not None else {},
            )

        return policy

    @parameterized.expand(
        [
            [1],
            [2],
            [5],
        ]
    )
    def test_n_enviroments_is_correct_for_vector_env(self, n_envs):
        env = self._create_vector_env(n_envs)
        loop = EnvironmentLoop(env, self._create_policy(env))

        self.assertEqual(loop.n_enviroments, n_envs)

    def _create_vector_env(self, n_env):
        return SyncVectorEnv([self._create_non_vector_env for _ in range(n_env)])

    @parameterized.expand(
        [
            [2, 0, False],
            [2, 1, False],
            [25, 0, False],
            [2, 1, True],
        ]
    )
    def test_seed_and_reset_always_return_the_same_samples(
        self, n_steps, seed, vectorized
    ):
        env1 = (
            self._create_non_vector_env()
            if not vectorized
            else self._create_vector_env(2)
        )
        loop1 = EnvironmentLoop(env1, self._create_policy(env1))

        env2 = (
            self._create_non_vector_env()
            if not vectorized
            else self._create_vector_env(2)
        )
        loop2 = EnvironmentLoop(env2, self._create_policy(env2))

        loop1.seed(seed)
        batch1 = SampleBatch.concat_samples([loop1.step() for _ in range(n_steps)])
        loop2.seed(seed)
        batch2 = SampleBatch.concat_samples([loop2.step() for _ in range(n_steps)])

        assert_sample_batch(batch1, batch2)

    def test_non_vector_env_return_batch_sample_batch(self):
        env = self._create_non_vector_env()
        loop = EnvironmentLoop(env, self._create_policy(env))

        batch = loop.step()

        for k, v in batch.items():
            self.assertTrue(
                len(v.shape) > 0, msg=f"k={k} is not batched it has shape {v.shape}"
            )
            self.assertTrue(
                len(v.shape) > 0, msg=f"k={k} is not batched it has shape {v.shape}"
            )
        self.assertTrue(
            len(batch[SampleBatch.OBSERVATIONS].shape) > 1,
            msg=f"k=OBSERVATIONS is not batched it has shape {batch[SampleBatch.OBSERVATIONS].shape}",
        )

    def test_vector_env_return_batch_sample_batch(self):
        n_envs = 3
        env = self._create_vector_env(n_envs)
        loop = EnvironmentLoop(env, self._create_policy(env))

        batch = loop.step()

        for k, v in batch.items():
            self.assertEqual(
                v.shape[:1],
                (n_envs,),
                msg=f"k={k} is not batched it has shape {v.shape}",
            )
        self.assertTrue(
            len(batch[SampleBatch.OBSERVATIONS].shape) > 1,
            msg=f"k=OBSERVATIONS is not batched it has shape {batch[SampleBatch.OBSERVATIONS].shape}",
        )

    def test_policy_must_return_cannot_be_a_tensor(self):
        def policy(_):
            return torch.tensor(0), {}

        env = self._create_non_vector_env()
        loop = EnvironmentLoop(env, policy)

        self.assertRaises(AssertionError, loop.step)

    def test_non_vector_env_gets_resets_automatically_on_done(self):
        env = mock.Mock(wraps=self._create_non_vector_env())
        loop = EnvironmentLoop(env, self._create_policy(env))
        loop.seed(1)
        env.reset_mock()

        for _ in range(500):
            batch = loop.step()
            if batch[SampleBatch.DONES][0]:
                break
        loop.step()

        env.reset.assert_called_once()

    def test_vector_env_gets_resets_automatically_on_done(self):
        env = self._create_vector_env(1)
        spy = mock.Mock(wraps=env.envs[0])
        env.envs[0] = spy
        loop = EnvironmentLoop(env, self._create_policy(env))
        loop.seed(1)
        spy.reset_mock()

        for _ in range(500):
            batch = loop.step()
            if batch[SampleBatch.DONES][0]:
                break
        loop.step()

        spy.reset.assert_called_once()

    def test_non_vector_env_increment_episode_ids(self):
        env = self._create_non_vector_env()
        loop = EnvironmentLoop(env, self._create_policy(env))
        loop.seed(1)

        prev_batch = loop.step()
        for _ in range(500):
            prev_batch = loop.step()
            if prev_batch[SampleBatch.DONES][0]:
                break
        new_episode_batch = loop.step()

        self.assertEqual(int(prev_batch[SampleBatch.EPS_ID][0]), 0)
        self.assertEqual(int(new_episode_batch[SampleBatch.EPS_ID][0]), 1)
        self.assertNotEqual(
            prev_batch[SampleBatch.EPS_ID], new_episode_batch[SampleBatch.EPS_ID]
        )

    @parameterized.expand([[1], [2], [5]])
    def test_vector_envs_give_each_parallel_env_its_own_id(self, n_envs):
        env = self._create_vector_env(n_envs)
        loop = EnvironmentLoop(env, self._create_policy(env))

        batch = loop.step()
        episode_ids = torch.unique(batch[SampleBatch.EPS_ID])

        torch.testing.assert_allclose(
            episode_ids, torch.arange(n_envs, dtype=episode_ids.dtype)
        )

    @parameterized.expand([[1], [2], [5]])
    def test_vector_env_increment_episode_ids(self, n_envs):
        env = self._create_vector_env(n_envs)
        loop = EnvironmentLoop(env, self._create_policy(env))
        loop.seed(1)

        prev_batch = loop.step()
        for _ in range(500):
            prev_batch = loop.step()
            if prev_batch[SampleBatch.DONES][0]:
                break
        new_episode_batch = loop.step()

        self.assertEqual(int(prev_batch[SampleBatch.EPS_ID][0]), 0)
        self.assertTrue(int(new_episode_batch[SampleBatch.EPS_ID][0]) > 0)

    def test_batch_contains_agent_info(self):
        env = self._create_non_vector_env()
        agent_info = {SampleBatch.VF_PREDS: torch.rand([1, 1])}

        def policy(_):
            return 0, agent_info

        loop = EnvironmentLoop(env, policy)
        batch = loop.step()

        torch.testing.assert_allclose(
            batch[SampleBatch.VF_PREDS], agent_info[SampleBatch.VF_PREDS]
        )


if __name__ == "__main__":
    unittest.main()

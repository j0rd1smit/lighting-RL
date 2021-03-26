import unittest
from unittest import mock

import gym
import torch
from gym.vector import SyncVectorEnv, VectorEnv
from parameterized import parameterized

from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.SampleBatch import SampleBatch
from tests.environmental.test_SampleBatchTest import assert_sample_batch


class EnvironmentLoopTest(unittest.TestCase):
    @parameterized.expand(
        [
            ["n_steps=1", 1],
            ["n_steps=2", 2],
        ]
    )
    def test_shapes_are_correct_env_with_discrete_obs_and_action_spaces(
        self, _, n_steps
    ):
        env = gym.make("Taxi-v3")

        loop = EnvironmentLoop(env, self._create_discrite_policy(env))
        batch = loop.step()
        for _ in range(1, n_steps):
            batch = loop.step()

        self._assert_has_shapes(batch, default=(1,))
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.REWARDS: torch.float32,
                SampleBatch.DONES: torch.float32,
            },
            default=torch.int64,
        )

    def _assert_has_shapes(self, batch, *, expected=None, default=None):
        if expected is None:
            expected = {}
        for k, v in batch.items():
            expected_v = expected.get(k, default)
            self.assertEqual(v.shape, expected_v, msg=f"{k} has an unexpected shape!")

    def _assert_has_dtype(self, batch, *, expected=None, default=None):
        if expected is None:
            expected = {}
        for k, v in batch.items():
            expected_v = expected.get(k, default)
            self.assertEqual(v.dtype, expected_v, msg=f"{k} has an unexpected dtype!")

    def _create_discrite_policy(self, env, agent_info=None):
        n_actions = (
            env.action_space.n
            if not isinstance(env, VectorEnv)
            else env.action_space[0].n
        )

        if agent_info is None:
            agent_info = {}

        def policy(x):
            actions = torch.randint(low=0, high=n_actions - 1, size=x.shape[:1])

            return actions, agent_info

        return policy

    @parameterized.expand(
        [
            ["n_steps=1", 1],
            ["n_steps=2", 2],
        ]
    )
    def test_shapes_are_correct_env_with_discrete_obs_and_action_spaces_sample(
        self, _, n_steps
    ):
        env = gym.make("Taxi-v3")

        loop = EnvironmentLoop(env, self._create_discrite_policy(env))
        batch = loop.sample()
        for _ in range(1, n_steps):
            batch = loop.sample()

        self._assert_has_shapes(batch, default=(1,))
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.REWARDS: torch.float32,
                SampleBatch.DONES: torch.float32,
            },
            default=torch.int64,
        )

    @parameterized.expand(
        [
            ["n_env=1_n_steps=1", 1, 1],
            ["n_env=1_n_steps=1", 1, 2],
            ["env=2_n_steps=1", 2, 1],
            ["env=2_n_steps=2", 2, 2],
        ]
    )
    def test_shapes_are_correct_env_with_discrete_obs_and_action_spaces_vector_env(
        self, _, n_envs, n_steps
    ):
        env = SyncVectorEnv([lambda: gym.make("Taxi-v3") for _ in range(n_envs)])
        loop = EnvironmentLoop(env, self._create_discrite_policy(env))

        batch = loop.step()
        for _ in range(1, n_steps):
            batch = loop.step()

        self._assert_has_shapes(batch, default=(n_envs,))
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.REWARDS: torch.float32,
                SampleBatch.DONES: torch.float32,
            },
            default=torch.int64,
        )

    @parameterized.expand(
        [
            ["n_env=1_n_steps=1", 1, 1],
            ["n_env=1_n_steps=1", 1, 2],
            ["env=2_n_steps=1", 2, 1],
            ["env=2_n_steps=2", 2, 2],
        ]
    )
    def test_shapes_are_correct_env_with_discrete_obs_and_action_spaces_vector_env(
        self, _, n_envs, n_steps
    ):
        env = SyncVectorEnv([lambda: gym.make("Taxi-v3") for _ in range(n_envs)])
        loop = EnvironmentLoop(env, self._create_discrite_policy(env))

        batch = loop.sample()
        for _ in range(1, n_steps):
            batch = loop.sample()

        self._assert_has_shapes(batch, default=(n_envs,))
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.REWARDS: torch.float32,
                SampleBatch.DONES: torch.float32,
            },
            default=torch.int64,
        )

    @parameterized.expand(
        [
            ["n_steps=1", 1],
            ["n_steps=2", 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_obs_and_discrete_action_spaces(
        self, _, n_steps
    ):
        env = gym.make("CartPole-v0")
        observation_shape = (1,) + env.observation_space.shape

        loop = EnvironmentLoop(env, self._create_discrite_policy(env))
        batch = loop.step()
        for _ in range(1, n_steps):
            batch = loop.step()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
            },
            default=(1,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.ACTIONS: torch.int64,
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand(
        [
            ["n_steps=1", 1],
            ["n_steps=2", 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_obs_and_discrete_action_spaces_sample(
        self, _, n_steps
    ):
        env = gym.make("CartPole-v0")
        observation_shape = (1,) + env.observation_space.shape

        loop = EnvironmentLoop(env, self._create_discrite_policy(env))
        batch = loop.sample()
        for _ in range(1, n_steps):
            batch = loop.sample()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
            },
            default=(1,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.ACTIONS: torch.int64,
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand(
        [
            ["n_envs=1_n_steps=1", 1, 1],
            ["n_envs=1_n_steps=1", 1, 2],
            ["n_envs=2_n_steps=1", 2, 1],
            ["n_envs=2_n_steps=2", 2, 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_obs_and_discrete_action_spaces_vector(
        self, _, n_envs, n_steps
    ):

        env = SyncVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(n_envs)])
        observation_shape = env.observation_space.shape

        loop = EnvironmentLoop(env, self._create_discrite_policy(env))
        batch = loop.step()
        for _ in range(1, n_steps):
            batch = loop.step()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
            },
            default=(n_envs,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.ACTIONS: torch.int64,
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand(
        [
            ["n_envs=1_n_steps=1", 1, 1],
            ["n_envs=1_n_steps=1", 1, 2],
            ["n_envs=2_n_steps=1", 2, 1],
            ["n_envs=2_n_steps=2", 2, 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_obs_and_discrete_action_spaces_vector_sample(
        self, _, n_envs, n_steps
    ):

        env = SyncVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(n_envs)])
        observation_shape = env.observation_space.shape

        loop = EnvironmentLoop(env, self._create_discrite_policy(env))
        batch = loop.sample()
        for _ in range(1, n_steps):
            batch = loop.sample()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
            },
            default=(n_envs,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.ACTIONS: torch.int64,
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand(
        [
            ["n_steps=1", 1],
            ["n_steps=2", 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_action_spaces(self, _, n_steps):
        env_name = "MountainCarContinuous-v0"
        env = gym.make(env_name)
        observation_shape = (1,) + env.observation_space.shape
        action_shape = (1,) + env.action_space.shape

        loop = EnvironmentLoop(env, self._create_continuouse_policy(env))
        batch = loop.step()
        for _ in range(1, n_steps):
            batch = loop.step()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
                SampleBatch.ACTIONS: action_shape,
            },
            default=(1,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    def _create_continuouse_policy(self, env, agent_info=None):
        action_shape = (
            env.action_space.shape
            if not isinstance(env, VectorEnv)
            else env.action_space[0].shape
        )

        agent_info = {} if agent_info is None else agent_info

        def policy(x):
            actions = torch.rand(x.shape[:1] + action_shape)

            return actions, agent_info

        return policy

    @parameterized.expand(
        [
            ["n_steps=1", 1],
            ["n_steps=2", 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_action_spaces_sample(
        self, _, n_steps
    ):
        env_name = "MountainCarContinuous-v0"
        env = gym.make(env_name)
        observation_shape = (1,) + env.observation_space.shape
        action_shape = (1,) + env.action_space.shape

        loop = EnvironmentLoop(env, self._create_continuouse_policy(env))
        batch = loop.sample()
        for _ in range(1, n_steps):
            batch = loop.sample()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
                SampleBatch.ACTIONS: action_shape,
            },
            default=(1,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand(
        [
            ["n_envs=1_n_steps=1", 1, 1],
            ["n_envs=1_n_steps=1", 1, 2],
            ["n_envs=2_n_steps=1", 2, 1],
            ["n_envs=2_n_steps=2", 2, 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_action_spaces_vector(
        self, _, n_envs, n_steps
    ):
        env_name = "MountainCarContinuous-v0"
        env = SyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
        observation_shape = (n_envs,) + env.envs[0].observation_space.shape
        action_shape = (n_envs,) + env.envs[0].action_space.shape

        loop = EnvironmentLoop(env, self._create_continuouse_policy(env))
        batch = loop.step()
        for _ in range(1, n_steps):
            batch = loop.step()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
                SampleBatch.ACTIONS: action_shape,
            },
            default=(n_envs,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand(
        [
            ["n_envs=1_n_steps=1", 1, 1],
            ["n_envs=1_n_steps=1", 1, 2],
            ["n_envs=2_n_steps=1", 2, 1],
            ["n_envs=2_n_steps=2", 2, 2],
        ]
    )
    def test_shapes_are_correct_env_with_continuous_action_spaces_vector_sample(
        self, _, n_envs, n_steps
    ):
        env_name = "MountainCarContinuous-v0"
        env = SyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
        observation_shape = (n_envs,) + env.envs[0].observation_space.shape
        action_shape = (n_envs,) + env.envs[0].action_space.shape

        loop = EnvironmentLoop(env, self._create_continuouse_policy(env))
        batch = loop.sample()
        for _ in range(1, n_steps):
            batch = loop.sample()

        self._assert_has_shapes(
            batch,
            expected={
                SampleBatch.OBSERVATIONS: observation_shape,
                SampleBatch.OBSERVATION_NEXTS: observation_shape,
                SampleBatch.ACTIONS: action_shape,
            },
            default=(n_envs,),
        )
        self._assert_has_dtype(
            batch,
            expected={
                SampleBatch.EPS_ID: torch.int64,
            },
            default=torch.float32,
        )

    @parameterized.expand([["MountainCarContinuous-v0"], ["Taxi-v3"], ["CartPole-v0"]])
    def test_obs_next_becomes_obs(self, env_name):
        loop = self._create_env_loop(env_name)

        batch1 = loop.step()
        batch2 = loop.step()

        torch.testing.assert_allclose(
            batch1[SampleBatch.OBSERVATION_NEXTS], batch2[SampleBatch.OBSERVATIONS]
        )

    def _create_env_loop(self, env_name, n_envs=None, agent_info=None):
        if n_envs is None:
            env = gym.make(env_name)
        else:
            env = SyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])

        if env_name == "MountainCarContinuous-v0":
            return EnvironmentLoop(
                env, self._create_continuouse_policy(env, agent_info)
            )

        if env_name == "Taxi-v3" or "CartPole" in env_name:
            return EnvironmentLoop(env, self._create_discrite_policy(env, agent_info))

        raise RuntimeError("Unknown env", env_name)

    @parameterized.expand([["MountainCarContinuous-v0"], ["Taxi-v3"], ["CartPole-v0"]])
    def test_batch_contains_agent_info(self, env_name):
        agent_info = {SampleBatch.VF_PREDS: torch.rand([1, 1])}
        loop = self._create_env_loop(env_name, agent_info=agent_info)

        batch = loop.step()

        torch.testing.assert_allclose(
            batch[SampleBatch.VF_PREDS], agent_info[SampleBatch.VF_PREDS]
        )

    @parameterized.expand([[1], [2], [5]])
    def test_vector_env_increment_episode_ids(self, n_envs):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name, n_envs)

        last_batch_episode = self._step_unstil_episode_is_done(loop)
        new_episode_batch = loop.step()

        # assert epsisode id increase the step after done
        self.assertTrue(
            int(new_episode_batch[SampleBatch.EPS_ID][0]) > 0,
            msg=f"Eps ID = {int(new_episode_batch[SampleBatch.EPS_ID][0])} but must be > 0",
        )

    def _step_unstil_episode_is_done(self, loop, max_step=10_000, env_idx=0):
        prev_batch = loop.step()
        for _ in range(max_step):
            prev_batch = loop.step()
            if prev_batch[SampleBatch.DONES][env_idx]:
                break
        return prev_batch

    @parameterized.expand([[1], [2], [5]])
    def test_vector_envs_give_each_parallel_env_its_own_id(self, n_envs):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name, n_envs)

        batch = loop.step()
        episode_ids = torch.unique(batch[SampleBatch.EPS_ID])

        self.assertEqual(len(episode_ids), n_envs)

    def test_non_vector_env_increment_episode_ids(self):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name)

        last_batch_episode = self._step_unstil_episode_is_done(loop)
        new_episode_batch = loop.step()

        self.assertNotEqual(
            last_batch_episode[SampleBatch.EPS_ID],
            new_episode_batch[SampleBatch.EPS_ID],
        )

    def test_vector_env_gets_resets_automatically_on_done(self):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name, n_envs=2)
        spy = mock.Mock(wraps=loop.env.envs[0])
        loop.env.envs[0] = spy
        spy.reset_mock()

        self._step_unstil_episode_is_done(loop, env_idx=0)

        spy.reset.assert_called_once()

    def test_non_vector_env_gets_resets_automatically_on_done(self):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name)
        spy = mock.Mock(wraps=loop.env)
        loop.env = spy
        loop.reset()
        spy.reset_mock()

        last_batch_episode = self._step_unstil_episode_is_done(loop)
        new_episode_batch = loop.step()

        spy.reset.assert_called_once()

    @parameterized.expand(
        [
            ["n_steps=2,seed=0", 2, 0],
            ["n_steps=2,seed=1", 2, 1],
            ["n_steps=25,seed=0", 25, 0],
            ["n_steps=2,seed=1", 2, 1],
        ]
    )
    def test_seed_and_reset_always_return_the_same_samples(
        self,
        _,
        n_steps,
        seed,
    ):
        env_name = "CartPole-v0"
        loop1 = self._create_env_loop(env_name)
        loop2 = self._create_env_loop(env_name)

        loop1.seed(seed)
        batch1 = SampleBatch.concat_samples([loop1.step() for _ in range(n_steps)])
        loop2.seed(seed)
        batch2 = SampleBatch.concat_samples([loop2.step() for _ in range(n_steps)])

        assert_sample_batch(batch1, batch2)

    @parameterized.expand(
        [
            ["1", 1],
            ["2", 2],
            ["5", 5],
        ]
    )
    def test_n_enviroments_is_correct_for_vector_env(self, _, n_envs):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name, n_envs=n_envs)

        self.assertEqual(loop.n_enviroments, n_envs)

    def test_n_environment_is_correct_for_non_vector_env(self):
        env_name = "CartPole-v0"
        loop = self._create_env_loop(env_name)

        self.assertEqual(loop.n_enviroments, 1)


if __name__ == "__main__":
    unittest.main()

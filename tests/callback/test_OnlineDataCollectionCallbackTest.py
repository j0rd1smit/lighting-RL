import unittest
from unittest import mock

import torch
from parameterized import parameterized

from lightning_rl.callbacks.OnlineDataCollectionCallback import (
    OnlineDataCollectionCallback,
)
from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.storage.UniformReplayBuffer import UniformReplayBuffer


class OnlineDataCollectionCallbackTest(unittest.TestCase):
    @parameterized.expand([[0], [1], [2], [5]])
    def test_can_prefill_buffer_before_training_if_needed(self, n_populate_steps):
        n_envs = 1

        buffer = UniformReplayBuffer(5)
        env_loop = self._create_env_loop_mock(n_envs)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=n_populate_steps,
            post_process_function=None,
        )

        self._mock_env_loop_step(env_loop, n_envs, n_steps=n_populate_steps)
        callback.on_fit_start(None, None)

        self.assertEqual(len(buffer), n_populate_steps)

    def _create_env_loop_mock(self, n_envs):

        env_loop = mock.Mock()
        env_loop.n_enviroments = n_envs

        return env_loop

    def _mock_env_loop_step(self, env_loop_mock, n_envs, n_steps=2):
        return_values = [self._create_sample_batch(n_envs) for _ in range(n_steps - 1)]
        return_values.append(
            self._create_sample_batch(n_envs, dones=True),
        )

        env_loop_mock.step.side_effect = return_values
        env_loop_mock.sample.side_effect = return_values

        return return_values

    def _create_sample_batch(self, n_envs, dones=False, episode_ids=None):
        if dones == True:
            dones = [True for _ in range(n_envs)]
        if dones == False:
            dones = [False for _ in range(n_envs)]

        if episode_ids is None:
            episode_ids = list(range(n_envs))

        return SampleBatch(
            {
                SampleBatch.OBSERVATIONS: torch.rand([n_envs, 1]),
                SampleBatch.ACTIONS: torch.rand([n_envs, 1]),
                SampleBatch.REWARDS: torch.rand([n_envs]),
                SampleBatch.DONES: torch.tensor(dones).float(),
                SampleBatch.OBSERVATION_NEXTS: torch.rand([n_envs, 1]),
                SampleBatch.EPS_ID: torch.tensor(episode_ids).long(),
            }
        )

    @parameterized.expand(
        [
            ["n_populate_steps=1_n_envs=1_n_samples_per_step=1", 1, 1, 1],
            ["n_populate_steps=2_n_envs=1_n_samples_per_step=1", 2, 1, 1],
            ["n_populate_steps=1_n_envs=2_n_samples_per_step=1", 1, 2, 1],
            ["n_populate_steps=1_n_envs=1_n_samples_per_step=2", 1, 1, 2],
        ]
    )
    def test_can_prefill_buffer_before_training_if_needed_vector_env_to_atleast_n_populate_steps(
        self, _, n_populate_steps, n_envs, n_samples_per_step
    ):
        buffer = UniformReplayBuffer(100)
        env_loop = self._create_env_loop_mock(n_envs)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=n_samples_per_step,
            n_populate_steps=n_populate_steps,
            post_process_function=None,
        )

        self._mock_env_loop_step(
            env_loop, n_envs, n_steps=n_populate_steps * n_samples_per_step
        )

        callback.on_fit_start(None, None)

        self.assertTrue(
            len(buffer) >= n_populate_steps, msg=f"{len(buffer)} < {n_populate_steps}"
        )

    def test_can_skip_prefill_buffer_before_training_if_not_needed(self):
        buffer = mock.Mock()
        buffer.__len__ = mock.Mock(return_value=0)
        buffer.capacity = 10
        env_loop = self._create_env_loop_mock(1)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=0,
            post_process_function=None,
        )

        callback.on_fit_start(None, None)

        buffer.append.assert_not_called()

    @parameterized.expand(
        [
            ["buffer_size_eqaul_to_n_populate_steps", 1, 1],
            ["buffer_size_larger_than_n_populate_steps", 1, 2],
        ]
    )
    def test_can_skip_prefill_buffer_before_training_if(
        self, _, n_populate_steps, buffer_size
    ):
        buffer = mock.Mock()
        buffer.__len__ = mock.Mock(return_value=buffer_size)
        buffer.capacity = buffer_size
        env_loop = self._create_env_loop_mock(1)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=n_populate_steps,
            post_process_function=None,
        )

        callback.on_fit_start(None, None)

        buffer.append.assert_not_called()

    @parameterized.expand(
        [
            ["n_envs_equal_to_n_samples_per_step", 1, 1],
            ["n_envs_less_than_n_samples_per_step", 1, 2],
            ["n_envs_more_than_n_samples_per_step", 2, 1],
        ]
    )
    def test_adds_atleast_the_required_amount_of_samples(
        self, _, n_envs, n_samples_per_step
    ):

        buffer = UniformReplayBuffer(2 * n_samples_per_step)
        env_loop = self._create_env_loop_mock(n_envs)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=n_samples_per_step,
            n_populate_steps=0,
            post_process_function=None,
        )

        self._mock_env_loop_step(env_loop, n_envs, n_steps=n_samples_per_step)
        callback.on_batch_end(None, None)

        self.assertTrue(
            len(buffer) >= n_samples_per_step,
            msg=f"{len(buffer)} < {n_samples_per_step}",
        )

    def test_clear_buffer_before_batch(self):
        buffer = mock.Mock(wraps=UniformReplayBuffer(5))
        env_loop = self._create_env_loop_mock(1)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=0,
            post_process_function=None,
            clear_buffer_before_gather=True,
        )

        self._mock_env_loop_step(env_loop, 1, n_steps=1)
        callback.on_batch_end(None, None)

        buffer.clear.assert_called_once()

    def test_do_not_clear_buffer_before_batch(self):
        buffer = mock.Mock(wraps=UniformReplayBuffer(5))
        env_loop = self._create_env_loop_mock(1)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=0,
            post_process_function=None,
            clear_buffer_before_gather=False,
        )

        self._mock_env_loop_step(env_loop, 1, n_steps=1)
        callback.on_batch_end(None, None)

        buffer.clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()

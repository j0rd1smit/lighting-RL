import unittest
from unittest import mock

import gym
import torch
from gym.vector import SyncVectorEnv
from parameterized import parameterized

from lightning_rl.callbacks.OnlineDataCollectionCallback import (
    OnlineDataCollectionCallback,
)
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.storage.UniformReplayBuffer import UniformReplayBuffer


class OnlineDataCollectionCallbackTest(unittest.TestCase):
    @parameterized.expand([[0], [1], [2], [5]])
    def test_can_prefill_buffer_before_training_if_needed(self, n_populate_steps):
        buffer = UniformReplayBuffer(5)
        env_loop = self._create_env_loop()
        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=n_populate_steps,
            post_process_function=None,
        )

        callback.on_fit_start(None, None)

        self.assertEqual(len(buffer), n_populate_steps)

    @parameterized.expand([[1, 1], [2, 2], [2, 3], [0, 1]])
    def test_doesnt_add_more_data_if_buffer_is_suffiently_full(
        self, n_populate_steps, buffer_size
    ):
        buffer = mock.Mock()
        buffer.__len__ = mock.Mock(return_value=buffer_size)

        env_loop = self._create_env_loop()
        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=n_populate_steps,
            post_process_function=None,
        )

        callback.on_fit_start(None, None)

        buffer.append.assert_not_called()

    def test_clear_buffer(self):
        buffer = mock.Mock()

        env_loop = self._create_env_loop()
        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=0,
            clear_buffer_after_batch=True,
        )

        callback.on_train_batch_end(None, None, None, None, None, None)

        buffer.clear.assert_called()

    def test_dont_clear_buffer(self):
        buffer = mock.Mock()

        env_loop = self._create_env_loop()
        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=1,
            n_populate_steps=0,
            clear_buffer_after_batch=False,
        )

        callback.on_train_batch_end(None, None, None, None, None, None)

        buffer.clear.assert_not_called()

    @parameterized.expand(
        [
            [1],
            [2],
            [5],
        ]
    )
    def test_step_adds_the_correct_number_of_samples_to_the_buffer(
        self, n_samples_per_step
    ):
        buffer = UniformReplayBuffer(5)
        env_loop = self._create_env_loop()
        post_process_function = None
        n_populate_steps = 0
        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step,
            n_populate_steps,
            post_process_function,
        )

        callback.on_train_batch_start(None, None, None, None, None)

        self.assertEqual(len(buffer), n_samples_per_step)

    def _create_env_loop(self, *, n_envs=None):
        if n_envs is not None:
            env = SyncVectorEnv(
                [lambda: gym.make("CartPole-v1") for _ in range(n_envs)]
            )
        else:
            env = gym.make("CartPole-v1")
        return EnvironmentLoop(env, lambda _: tuple([env.action_space.sample(), {}]))

    def test_step_stores_the_batches_correctly_in_the_buffer(self):
        seed = 0
        buffer = UniformReplayBuffer(5)

        env_loop = self._create_env_loop()
        env_loop.seed(seed)

        env_loop2 = self._create_env_loop()
        env_loop2.seed(seed)

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=2,
            n_populate_steps=0,
            post_process_function=None,
        )

        callback.on_train_batch_start(None, None, None, None, None)

        stored_batch = buffer[0]
        expected_batch = env_loop2.step()
        for k in expected_batch:
            torch.testing.assert_allclose(stored_batch[k], expected_batch[k])

    def test_after_step_the_buffer_receieves_post_processed_batches(self):
        seed = 0
        buffer = UniformReplayBuffer(5)

        env_loop = self._create_env_loop()
        env_loop.seed(seed)

        def post_process_function(batch):
            batch[SampleBatch.REWARDS] = torch.zeros_like(batch[SampleBatch.REWARDS])
            return batch

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=2,
            n_populate_steps=0,
            post_process_function=post_process_function,
        )

        callback.on_train_batch_start(None, None, None, None, None)

        torch.testing.assert_allclose(
            buffer[0][SampleBatch.REWARDS],
            torch.zeros_like(buffer[0][SampleBatch.REWARDS]),
        )

    def test_post_processor_receive_batch_per_episode(self):
        seed = 0
        buffer = UniformReplayBuffer(5)

        env_loop = self._create_env_loop(n_envs=5)
        env_loop.seed(seed)

        def post_process_function(batch):
            self.assertEqual(len(torch.unique(batch[SampleBatch.EPS_ID])), 1)
            return batch

        callback = OnlineDataCollectionCallback(
            buffer,
            env_loop,
            n_samples_per_step=2,
            n_populate_steps=0,
            post_process_function=post_process_function,
        )

        callback.on_train_batch_start(None, None, None, None, None)


if __name__ == "__main__":
    unittest.main()

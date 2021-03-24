import unittest

import gym
import torch
from gym.vector import SyncVectorEnv
from parameterized import parameterized

from lightning_rl.environmental.DataGatheringLoop import DataGatheringLoop
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop
from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.storage.UniformReplayBuffer import UniformReplayBuffer


class DataGatheringLoopTest(unittest.TestCase):
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
        data_gather_loop = DataGatheringLoop(
            buffer, env_loop, n_samples_per_step, post_process_function
        )

        data_gather_loop.step()

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

        post_process_function = None
        data_gather_loop = DataGatheringLoop(buffer, env_loop, 2, post_process_function)

        data_gather_loop.step()

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

        data_gather_loop = DataGatheringLoop(buffer, env_loop, 2, post_process_function)

        data_gather_loop.step()

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

        data_gather_loop = DataGatheringLoop(buffer, env_loop, 2, post_process_function)

        data_gather_loop.step()


if __name__ == "__main__":
    unittest.main()

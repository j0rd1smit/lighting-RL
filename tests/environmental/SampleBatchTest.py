import unittest

import torch
from parameterized import parameterized

from lightning_rl.storage.SampleBatch import SampleBatch


def assert_sample_batch(one, other):
    for k in set(one.keys()).union(set(other.keys())):
        torch.testing.assert_allclose(one[k], other[k])


class SampleBatchTest(unittest.TestCase):
    def test_cannot_create_empty_sample_batch(self):
        self.assertRaises(AssertionError, lambda: SampleBatch({}))

    def test_cannot_create_empty_sample_unbatched_float(self):
        self.assertRaises(
            AssertionError,
            lambda: SampleBatch({SampleBatch.OBSERVATIONS: torch.tensor(1.0)}),
        )

    @parameterized.expand(
        [
            [1],
            [2],
            [10],
        ]
    )
    def test_n_samples_is_correct_for_diff_batch_sizes(self, batch_size):
        batch = SampleBatch({SampleBatch.OBSERVATIONS: torch.zeros([batch_size])})
        self.assertEqual(batch.n_samples, batch_size)

    @parameterized.expand(
        [
            [SampleBatch.OBSERVATIONS, 1, 1, 1],
            [SampleBatch.OBSERVATIONS, 2, 1, 2],
            [SampleBatch.OBSERVATION_NEXTS, 1, 5, 3],
            [SampleBatch.ACTIONS, 1, 1, 1],
            [SampleBatch.REWARDS, 2, 1, 2],
            [SampleBatch.DONES, 1, 5, 3],
        ]
    )
    def test_concat_samples_stack_shapes_correctly(
        self, key, batch_size_one, batch_size_other, obs_dim
    ):
        batch_one = SampleBatch({key: torch.rand([batch_size_one, obs_dim])})
        batch_other = SampleBatch({key: torch.rand([batch_size_other, obs_dim])})

        batch = SampleBatch.concat_samples([batch_one, batch_other])
        self.assertEqual(batch[key].shape[0], batch_size_other + batch_size_one)
        self.assertEqual(batch[key].shape[1], obs_dim)

    def test_concat_samples_stack_works_with_multiple_keys(self):
        keys = [
            SampleBatch.OBSERVATIONS,
            SampleBatch.ACTIONS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
        ]
        batch_size_one = 2
        batch_size_other = 2
        obs_dim = 1
        batch_one = SampleBatch(
            {key: torch.rand([batch_size_one, obs_dim]) for key in keys}
        )
        batch_other = SampleBatch(
            {key: torch.rand([batch_size_other, obs_dim]) for key in keys}
        )

        batch = SampleBatch.concat_samples([batch_one, batch_other])
        for key in keys:
            self.assertEqual(batch[key].shape[0], batch_size_other + batch_size_one)
            self.assertEqual(batch[key].shape[1], obs_dim)

    def test_split_by_episode_can_handle_a_single_epsidode(self):
        batch = self._create_dummy_sample_batch(2)
        batch_per_episode = batch.split_by_episode()
        self.assertEqual(len(batch_per_episode), 1)
        assert_sample_batch(batch, batch_per_episode[-1])

    def _create_dummy_sample_batch(self, batch_size, episode_id=0):
        return SampleBatch(
            {
                SampleBatch.OBSERVATIONS: torch.rand([batch_size, 2]),
                SampleBatch.ACTIONS: torch.rand([batch_size, 2]),
                SampleBatch.REWARDS: torch.rand([batch_size]),
                SampleBatch.DONES: torch.rand([batch_size]),
                SampleBatch.OBSERVATION_NEXTS: torch.rand([batch_size, 2]),
                SampleBatch.EPS_ID: torch.zeros([batch_size]) + episode_id,
            }
        )

    @parameterized.expand(
        [
            [1],
            [2],
            [5],
        ]
    )
    def test_split_by_episode_can_handle_a_multiple_epsidode(self, n_episodes):
        episodes = [
            self._create_dummy_sample_batch(2, episode_id=i) for i in range(n_episodes)
        ]
        batch = SampleBatch.concat_samples(episodes)
        batch_per_episode = batch.split_by_episode()
        self.assertEqual(len(batch_per_episode), n_episodes)

        for i, episode in enumerate(episodes):
            assert_sample_batch(episodes[i], batch_per_episode[i])


if __name__ == "__main__":
    unittest.main()

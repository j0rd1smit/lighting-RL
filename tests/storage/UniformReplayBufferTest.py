import unittest

import torch
from parameterized import parameterized

from lightning_rl.environmental.SampleBatch import SampleBatch
from lightning_rl.storage.UniformReplayBuffer import UniformReplayBuffer


class UniformReplayBufferTest(unittest.TestCase):
    CAPACITY = 10

    @parameterized.expand([[0], [1], [10], [1000]])
    def test_capacity(self, capacity):
        self.assertEqual(UniformReplayBuffer(capacity).capacity, capacity)

    def test_append(self):
        buffer = UniformReplayBuffer(self.CAPACITY)
        sample_batch = self._create_sample_batch()

        buffer.append(sample_batch)

        self._assert_batch_equal(buffer[0], sample_batch)

    def _create_sample_batch(self):
        return SampleBatch(
            {
                SampleBatch.OBSERVATIONS: torch.rand([1, 4]),
                SampleBatch.OBSERVATION_NEXTS: torch.rand([1, 4]),
                SampleBatch.REWARDS: torch.randint(0, 8, [1]),
                SampleBatch.ACTIONS: torch.randint(0, 8, [1]),
                SampleBatch.DONES: torch.rand(1) > 0.5,
            }
        )

    def _assert_batch_equal(self, batch, other):
        for k, v in batch.items():
            if k not in UniformReplayBuffer.EXCLUDED_KEYS:
                torch.testing.assert_allclose(batch[k], other[k])

    def test_get_item_contains_idx(self):
        buffer = UniformReplayBuffer(self.CAPACITY)
        sample_batch = self._create_sample_batch()
        self.assertNotIn(SampleBatch.IDX, sample_batch)

        buffer.append(sample_batch)

        retrieved_batch = buffer[0]

        self.assertIn(SampleBatch.IDX, retrieved_batch)

    def test_cannot_append_with_different_shape(self):
        buffer = UniformReplayBuffer(self.CAPACITY)

        batch1 = SampleBatch(
            {
                SampleBatch.OBSERVATIONS: torch.rand([1, 8]),
                SampleBatch.REWARDS: torch.rand([1, 1]),
            }
        )
        buffer.append(batch1)

        batch2 = SampleBatch(
            {
                SampleBatch.OBSERVATIONS: torch.rand([1, 5]),
                SampleBatch.REWARDS: torch.rand([1, 1]),
            }
        )

        self.assertRaises(Exception, lambda: buffer.append(batch2))

    @parameterized.expand([[2, 1], [3, 2], [3, 4], [3, 6], [10, 5]])
    def test__relationship_between_size_and_capacity(self, capacity, n_samples):
        expected_size = min(capacity, n_samples)
        buffer = UniformReplayBuffer(capacity)
        self.assertEqual(buffer.size, 0)

        for _ in range(n_samples):
            batch = self._create_sample_batch()
            buffer.append(batch)

        self.assertEqual(len(buffer), expected_size)

    @parameterized.expand([[2, 1], [3, 2], [3, 4], [3, 6], [10, 5]])
    def test_append_more_than_capacity_should_go_round_about(self, capacity, n_samples):
        buffer = UniformReplayBuffer(capacity)

        batches = [None for _ in range(capacity)]
        for i in range(n_samples):
            batch = self._create_sample_batch()
            buffer.append(batch)
            batches[i % capacity] = batch

        for i in range(buffer.size):
            self._assert_batch_equal(buffer[i], batches[i])


if __name__ == "__main__":
    unittest.main()

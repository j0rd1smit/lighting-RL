import unittest
from unittest import mock

from parameterized import parameterized

from lightning_rl.dataset.samplers.UniformSampler import UniformSampler


class UniformSamplerTest(unittest.TestCase):
    @parameterized.expand([[1], [2], [5]])
    def test_samples_do_not_go_out_of_index(self, dataset_size):
        buffer = mock.MagicMock()
        buffer.__len__.return_value = dataset_size
        samples_per_epoch = 4 * dataset_size
        sampler = UniformSampler(buffer, samples_per_epoch)

        samples = list(iter(sampler))
        self.assertEqual(max(samples), dataset_size - 1)
        self.assertEqual(min(samples), 0)

    def test_largest_index_increase_as_buffer_size_increases_dynamicaly(self):
        dataset_size = 2
        buffer = mock.MagicMock()
        buffer.__len__.return_value = dataset_size
        samples_per_epoch = 5 * dataset_size
        sampler = UniformSampler(buffer, samples_per_epoch)

        samples = list(iter(sampler))
        self.assertEqual(max(samples), dataset_size - 1)

        buffer.__len__.return_value = dataset_size + 1
        samples = list(iter(sampler))
        self.assertEqual(max(samples), dataset_size)

    @parameterized.expand([[1], [2], [5]])
    def test_returns_the_correct_amount_of_samples_per_epoch(self, samples_per_epoch):
        buffer = mock.MagicMock()
        buffer.__len__.return_value = 2
        sampler = UniformSampler(buffer, samples_per_epoch)

        samples = list(iter(sampler))
        self.assertEqual(len(samples), samples_per_epoch)


if __name__ == "__main__":
    unittest.main()

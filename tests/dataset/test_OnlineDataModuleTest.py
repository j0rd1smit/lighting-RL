import unittest

import torch
from parameterized import parameterized
from torch.utils.data import TensorDataset

from lightning_rl.dataset.OnlineDataModule import OnlineDataModule
from lightning_rl.dataset.samplers.UniformSampler import UniformSampler


class OnlineDataModuleTest(unittest.TestCase):
    def test_dataloader_can_sample_entire_dataset(self):
        data = torch.rand(1, 4)
        target_data = torch.rand(1, 1)
        buffer = TensorDataset(data, target_data)
        sampler = UniformSampler(buffer, 1)

        data_module = OnlineDataModule(buffer, 1, sampler=sampler, pin_memory=False)
        x, y = next(iter(data_module.train_dataloader()))

        torch.testing.assert_allclose(x, data)
        torch.testing.assert_allclose(y, target_data)

    @parameterized.expand([[1], [2], [5]])
    def test_dataloader_give_correct_batch_size(self, batch_size):
        data = torch.rand(10, 4)
        target_data = torch.rand(10, 1)
        buffer = TensorDataset(data, target_data)
        sampler = UniformSampler(buffer, 10)

        data_module = OnlineDataModule(
            buffer, batch_size, sampler=sampler, pin_memory=False
        )
        x, y = next(iter(data_module.train_dataloader()))

        self.assertEqual(x.shape[0], batch_size)
        self.assertEqual(y.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()

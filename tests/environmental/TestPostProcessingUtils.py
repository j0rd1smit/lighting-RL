import unittest

import torch
from parameterized import parameterized

from lightning_rl.environmental.post_processing_utils import (
    compute_advantages,
    discount_cumsum,
    Postprocessing,
)
from lightning_rl.environmental.SampleBatch import SampleBatch


class TestPostProcessingUtils(unittest.TestCase):
    @parameterized.expand(
        [
            [[0, 0, 0], [0, 0, 0], 0.99],
            [[0, 0, 1], [0.9801, 0.99, 1], 0.99],
            [[0, 0, 1], [0.81, 0.90, 1], 0.90],
            [[0, 1, 1], [2, 2, 1], 1],
        ]
    )
    def test_discount_cumsum(self, values, answers, gamma):
        values = discount_cumsum(torch.tensor(values, dtype=torch.float32), gamma)
        answers = torch.tensor(answers, dtype=torch.float32)

        self.assertEqual(values.dtype, answers.dtype)
        torch.testing.assert_allclose(values, answers)

    @parameterized.expand(
        [
            [[0, 0, 1], 0.99, 0],
            [[0, 0, 0], 0.99, 0],
            [[0, 1, 0], 0.99, 0],
            [[0, 1, 0], 1, 0],
            [[0, 0, 1], 0.9, 0],
            [[0, 0, 1], 1, 1],
        ]
    )
    def test_compute_advantages_no_use_gae_no_critic(self, rewards, gamma, last_r):

        batch = SampleBatch(
            {SampleBatch.REWARDS: torch.tensor(rewards, dtype=torch.float32)}
        )
        original_rewards = batch[SampleBatch.REWARDS]

        batch = compute_advantages(
            batch, last_r=last_r, gamma=gamma, use_gae=False, use_critic=False
        )

        torch.testing.assert_allclose(
            batch[Postprocessing.ADVANTAGES],
            discount_cumsum(
                torch.tensor(rewards + [last_r], dtype=torch.float32), gamma
            ),
        )

    @parameterized.expand(
        [
            [[0, 0, 1], [0, 0, 1, 1], 0.99, 0],
            [[0, 1, 1], [0, 0, 0, 0], 0.99, 0],
            [[0, 1, 1], [0, 0, 0, 0], 0.99, 1],
            [[0, 0], [0, 0, 0], 0.99, 0],
        ]
    )
    def test_compute_advantages_no_use_gae_critic(
        self, rewards, vf_predict, gamma, last_r
    ):

        batch = SampleBatch(
            {
                SampleBatch.REWARDS: torch.tensor(rewards, dtype=torch.float32),
                SampleBatch.VF_PREDS: torch.tensor(vf_predict, dtype=torch.float32),
            }
        )

        batch = compute_advantages(
            batch, last_r=last_r, gamma=gamma, use_gae=False, use_critic=True
        )
        discounted_return = discount_cumsum(
            torch.tensor(rewards + [last_r], dtype=torch.float32), gamma
        )
        expected_advantage = discounted_return - batch[SampleBatch.VF_PREDS]
        torch.testing.assert_allclose(
            batch[Postprocessing.ADVANTAGES], expected_advantage
        )
        torch.testing.assert_allclose(
            batch[Postprocessing.VALUE_TARGETS], discounted_return
        )


if __name__ == "__main__":
    unittest.main()

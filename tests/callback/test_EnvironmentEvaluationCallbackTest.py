import unittest
from unittest import mock

import torch
from parameterized import parameterized

from lightning_rl.callbacks.EnvironmentEvaluationCallback import (
    EnvironmentEvaluationCallback,
)
from lightning_rl.environmental.SampleBatch import SampleBatch


class EnvironmentEvaluationCallbackTest(unittest.TestCase):
    @parameterized.expand(
        [
            ["1_times", 1],
            ["2_times", 2],
            ["3_times", 3],
        ]
    )
    def test_if_seed_is_provided_it_will_seed_the_loop_before_every_run(self, _, n):
        n_eval_episodes = 1
        seed = 0
        env_loop = self._create_env_loop_mock(n_eval_episodes)

        callback = EnvironmentEvaluationCallback(
            env_loop,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
        )

        for i in range(1, n):
            self._mock_env_loop_step(env_loop, n_eval_episodes)
            callback.on_train_epoch_end(mock.Mock(), mock.Mock(), mock.Mock())

            self.assertEqual(env_loop.seed.call_count, i)
            env_loop.seed.assert_called_with(seed)

    def _create_env_loop_mock(self, n_eval_episodes):
        n_envs = n_eval_episodes

        env_loop = mock.Mock()
        env_loop.n_enviroments = n_envs

        return env_loop

    def _mock_env_loop_step(self, env_loop_mock, n_envs, n_steps=2):
        assert n_steps >= 1
        return_values = [self._create_sample_batch(n_envs) for _ in range(n_steps - 1)]
        return_values.append(
            self._create_sample_batch(n_envs, dones=True),
        )

        env_loop_mock.step.side_effect = return_values

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

    @parameterized.expand([[True], [False]])
    def test_can_enable_eval_mode(self, was_in_training_mode):
        n_eval_episodes = 1
        env_loop = self._create_env_loop_mock(n_eval_episodes)
        pl_module = mock.Mock()
        pl_module.training_mode = was_in_training_mode

        callback = EnvironmentEvaluationCallback(env_loop, to_eval=True, n_eval_episodes=n_eval_episodes)
        self._mock_env_loop_step(env_loop, n_eval_episodes)
        callback.on_train_epoch_end(mock.Mock(), pl_module, mock.Mock())

        pl_module.eval.assert_called_once()
        if was_in_training_mode:
            pl_module.train.assert_called_once()

    @parameterized.expand(
        [
            ["n_steps=1_n_eval_episodes=1", 1, 1],
            ["n_steps=1_n_eval_episodes=2", 1, 2],
            ["n_steps=2_n_eval_episodes=1", 2, 1],
            ["n_steps=2_n_eval_episodes=1", 2, 2],
            ["n_steps=3_n_eval_episodes=1", 3, 1],
        ]
    )
    def test_measure_length_correctly(self, _, n_steps, n_eval_episodes):
        env_loop = self._create_env_loop_mock(n_eval_episodes)

        def _test_case_callback(lengths):
            self.assertListEqual(list(lengths), [n_steps for _ in range(n_eval_episodes)])

        callback = EnvironmentEvaluationCallback(
            env_loop,
            n_eval_episodes=1,
            length_mappers={"": _test_case_callback},
            return_mappers={},
            mean_return_in_progress_bar=False,
        )
        self._mock_env_loop_step(env_loop, n_eval_episodes, n_steps=n_steps)
        callback.on_train_epoch_end(mock.Mock(), mock.Mock(), mock.Mock())

    @parameterized.expand(
        [
            ["n_eval_episodes=1_n_steps=2", 1, 2],
            ["n_eval_episodes=2_n_steps=2", 2, 2],
            ["n_eval_episodes=2_n_steps=4", 2, 4],
        ]
    )
    def test_measure_return_correctly(self, _, n_eval_episodes, n_steps):
        env_loop = self._create_env_loop_mock(n_eval_episodes)

        side_effects = self._mock_env_loop_step(env_loop, n_eval_episodes, n_steps=n_steps)

        def _test_case_callback(rewards):
            expected_rewards = list(sum(batch[SampleBatch.REWARDS].double() for batch in side_effects).numpy())
            self.assertListEqual(list(rewards), expected_rewards)

        callback = EnvironmentEvaluationCallback(
            env_loop,
            n_eval_episodes=1,
            length_mappers={},
            return_mappers={"": _test_case_callback},
            mean_return_in_progress_bar=False,
        )

        callback.on_train_epoch_end(mock.Mock(), mock.Mock(), mock.Mock())


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

import gym
from gym.vector import SyncVectorEnv
from parameterized import parameterized

from lightning_rl.callbacks.EnvironmentEvaluationCallback import (
    EnvironmentEvaluationCallback,
)
from lightning_rl.environmental.EnvironmentLoop import EnvironmentLoop


class EnvironmentEvaluationCallbackTest(unittest.TestCase):
    @parameterized.expand(
        [
            [0, None],
            [1, None],
            [1, 5],
            [1, 10],
        ]
    )
    def test_with_seeded_always_eval_on_the_same_instances(self, seed, n_envs):
        n_eval_episodes = 10
        env_loop = self._create_env_loop(n_envs=n_envs)

        return_mapper_mock = mock.Mock()
        return_mappers = {"": return_mapper_mock}

        length_mapper_mock = mock.Mock()
        length_mappers = {"": length_mapper_mock}

        callback = EnvironmentEvaluationCallback(
            env_loop,
            n_eval_episodes=n_eval_episodes,
            return_mappers=return_mappers,
            length_mappers=length_mappers,
            seed=seed,
        )

        # Eval for the first time to get init value
        callback.on_epoch_end(mock.Mock(), mock.Mock())

        return_mapper_mock.assert_called()
        first_eval_return = return_mapper_mock.call_args.args[0]

        length_mapper_mock.assert_called()
        first_eval_length = length_mapper_mock.call_args.args[0]

        # Verify that it the return and length don't change if the seed and policy remain fixed.
        for i in range(1, 3):
            return_mapper_mock.reset_mock()
            length_mapper_mock.reset_mock()

            callback.on_epoch_end(mock.Mock(), mock.Mock())

            return_mapper_mock.assert_called()
            self.assertEqual(
                list(first_eval_return),
                list(return_mapper_mock.call_args.args[0]),
                msg=f"Returns are no longer consistent for the same seed. For the {i}th eval.",
            )

            length_mapper_mock.assert_called()
            self.assertEqual(
                list(first_eval_length),
                list(length_mapper_mock.call_args.args[0]),
                msg=f"Lengths are no longer consistent for the same seed. For the {i}th eval.",
            )

    def _create_env_loop(self, *, n_envs=None):
        if n_envs is not None:
            env = SyncVectorEnv(
                [lambda: gym.make("CartPole-v0") for _ in range(n_envs)]
            )
        else:
            env = gym.make("CartPole-v0")

        def policy(_):
            return env.action_space.sample(), {}

        return EnvironmentLoop(env, policy)

    @parameterized.expand([[True], [False]])
    def test_can_enable_eval_mode(self, was_in_training_mode):
        env_loop = self._create_env_loop()
        callback = EnvironmentEvaluationCallback(
            env_loop,
            to_eval=True,
        )
        trainer = mock.Mock()
        pl_module = mock.Mock()
        pl_module.training_mode = was_in_training_mode

        callback.on_epoch_end(trainer, pl_module)

        pl_module.eval.assert_called_once()
        if was_in_training_mode:
            pl_module.train.assert_called_once()

    def test_measure_length_correctly(self):
        env = mock.Mock(wraps=gym.make("CartPole-v0"))

        def policy(_):
            return 0, {}

        env_loop = EnvironmentLoop(env, policy)

        callback = EnvironmentEvaluationCallback(
            env_loop,
            n_eval_episodes=1,
            length_mappers={"": lambda o: o[0]},
            return_mappers={},
            mean_return_in_progress_bar=False,
        )
        pl_module = mock.Mock()
        callback.on_epoch_end(mock.Mock(), pl_module)

        # test case assume only one call and achieves this using custom mappers
        pl_module.log.assert_called_once()

        times_called = env.step.call_count
        logged_length = pl_module.log.call_args.args[1]
        self.assertEqual(times_called, logged_length)

    def test_measure_return_correctly(self):
        env = mock.Mock(wraps=gym.make("CartPole-v0"))

        def policy(_):
            return 0, {}

        env_loop = EnvironmentLoop(env, policy)

        callback = EnvironmentEvaluationCallback(
            env_loop,
            n_eval_episodes=1,
            length_mappers={},
            return_mappers={"": lambda o: o[0]},
            mean_return_in_progress_bar=False,
        )
        pl_module = mock.Mock()
        callback.on_epoch_end(mock.Mock(), pl_module)

        # test case assume only one call and achieves this using custom mappers
        pl_module.log.assert_called_once()

        times_called = env.step.call_count  # 1 reward per time step
        logged_return = pl_module.log.call_args.args[1]
        self.assertEqual(times_called, logged_return)


if __name__ == "__main__":
    unittest.main()

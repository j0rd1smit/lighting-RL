import unittest


class EnvironmentLoopTest(unittest.TestCase):
    def test_(self):
        self.assertTrue(True)

    def test_n_enviroments_is_correct_for_non_vector_env(self):
        pass

    def test_n_enviroments_is_correct_for_vector_env(self):
        pass

    def test_seed_and_reset_always_return_the_same_samples(self):
        pass

    def test_non_vector_env_return_batch_sample_batch(self):
        pass

    def test_vector_env_return_batch_sample_batch(self):
        pass

    def test_policy_must_return_np_arrary_or_number(self):
        pass

    def test_non_vector_env_gets_resetted_automaticaly_on_done(self):
        pass

    def test_vector_env_gets_resetted_automaticaly_on_done(self):
        pass

    def test_non_vector_env_increment_episode_ids(self):
        pass

    def test_vector_env_increment_episode_ids(self):
        pass


if __name__ == "__main__":
    unittest.main()

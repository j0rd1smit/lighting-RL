from lightning_rl.dataset.OnlineDataModule import OnlineDataModule
from lightning_rl.dataset.samplers.UniformSampler import UniformSampler
from lightning_rl.storage.UniformReplayBuffer import UniformReplayBuffer


def dataset_builder() -> None:
    # TODO
    pass


def off_policy_dataset(
    capacity: int, batch_size: int, steps_per_epoch: int
) -> OnlineDataModule:
    buffer = UniformReplayBuffer(capacity)

    samples_per_epoch = steps_per_epoch * batch_size  # TODO
    sampler = UniformSampler(buffer, samples_per_epoch)

    return OnlineDataModule(buffer, sampler, batch_size, pin_memory=True, n_workers=0)

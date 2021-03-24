import scipy.signal
import torch

from lightning_rl.environmental.SampleBatch import SampleBatch


class Postprocessing:
    ADVANTAGES = "ADVANTAGES"
    VALUE_TARGETS = "VALUE_TARGETS"


# https://github.com/ray-project/ray/blob/a3eb64311e4b0ca507e841049b0bac3be4dbf249/rllib/evaluation/postprocessing.py#L84
def compute_advantages(
    batch: SampleBatch,
    last_r: float,
    gamma: float,
    lambda_: float = 1.0,
    *,
    use_gae: bool = False,
    use_critic: bool = False,
) -> SampleBatch:
    assert (
        SampleBatch.VF_PREDS in batch or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use gae without using a value function"

    if use_gae:
        raise NotImplementedError
    else:
        rewards_plus_v = torch.cat(
            [
                batch[SampleBatch.REWARDS],
                torch.zeros_like(batch[SampleBatch.REWARDS][0]).unsqueeze(0) + last_r,
            ]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)

        if use_critic:
            batch[Postprocessing.ADVANTAGES] = (
                discounted_returns - batch[SampleBatch.VF_PREDS]
            )
            batch[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            batch[Postprocessing.ADVANTAGES] = discounted_returns

    return batch


def discount_cumsum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    x = x.numpy()
    dtype = x.dtype

    return torch.from_numpy(
        scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]
        .astype(dtype)
        .copy()
    )

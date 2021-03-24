from typing import List

import torch


class SampleBatch(dict):
    OBSERVATIONS = "OBSERVATIONS"
    ACTIONS = "ACTIONS"
    REWARDS = "REWARDS"
    DONES = "DONES"
    OBSERVATION_NEXTS = "OBSERVATION_NEXTS"
    IDX = "IDX"
    VF_PREDS = "VF_PREDS"
    EPS_ID = "EPS_ID"

    @property
    def n_samples(self) -> int:
        if len(self) == 0:
            return 0

        lenght = -1
        for k, v in self.items():
            if len(v.shape) == 0:
                continue
            if lenght == -1:
                lenght = len(v)

            assert len(v) == lenght, f"k={k} len(v) = {len(v)} lenght={lenght}"

        return lenght

    @staticmethod
    def concat_samples(samples: List["SampleBatch"]) -> "SampleBatch":
        concat_samples = []
        for s in samples:
            concat_samples.append(s)

        out = {}
        for k in concat_samples[0].keys():
            out[k] = torch.stack([s[k] for s in concat_samples], 0)

        return SampleBatch(out)

    def split_by_episode(self) -> List["SampleBatch"]:
        assert SampleBatch.EPS_ID in self

        episode_ids = torch.unique(self[SampleBatch.EPS_ID])
        slices = []
        for episode_id in episode_ids:
            slices.append(
                SampleBatch(
                    {
                        k: v[self[SampleBatch.EPS_ID] == episode_id]
                        for k, v in self.items()
                    }
                )
            )

        return slices

    def slice(self, start: int, end: int) -> "SampleBatch":
        return SampleBatch({k: v[start:end] for k, v in self.items()})

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

Seed = Optional[int]

Action = torch.Tensor
Observation = Union[np.ndarray]


ActionAgentInfoTuple = Tuple[torch.Tensor, Dict[str, torch.Tensor]]
Policy = Callable[[torch.Tensor], ActionAgentInfoTuple]

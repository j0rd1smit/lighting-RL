from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

Seed = Optional[int]

Action = Union[float, np.ndarray]
Observation = Union[float, np.ndarray]


ActionAgentInfoTuple = Tuple[Action, Dict]
Policy = Callable[[Observation], ActionAgentInfoTuple]

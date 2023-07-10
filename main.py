import numpy as np
from enum import Enum
from dataclasses import dataclass

maze = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
])

Direction = Enum('Direction', ['UP', 'DOWN', 'RIGHT', 'LEFT'])


@dataclass
class State:
    location: (float, float)


print(maze)

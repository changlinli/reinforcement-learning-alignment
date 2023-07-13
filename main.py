import numpy as np
from enum import Enum
from dataclasses import dataclass

from numpy import ndarray
from numpy.random import Generator
from torch import nn
from pyrsistent import v, pvector, PVector, PMap

maze = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
])

Action = Enum('Action', ['UP', 'DOWN', 'RIGHT', 'LEFT'])

print(maze)

device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(maze.size, maze.size),
            nn.ReLU(),
            nn.Linear(maze.size, maze.size),
            nn.ReLU(),
            nn.Linear(maze.size, maze.size),
            nn.ReLU(),
            nn.Linear(maze.size, len(Action)),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



@dataclass
class State:
    location: (float, float)

    def to_numpy(self) -> ndarray:
        return np.asarray(self.location)


@dataclass
class Episode:
    state: State
    action: Action
    reward: float
    next_state: State
    is_game_over: bool


memory: PVector[Episode] = v()


@dataclass
class TrainingExample:
    guess: State
    target: PMap[Action, float]


def sample_from_episodes(
        episodes: PVector[Episode],
        random_generator: Generator,
        sample_size: int,
        model: NeuralNetwork,
) -> PVector[TrainingExample]:
    sample_episode_idxs = random_generator.choice(range(len(episodes)), size=sample_size)
    sample_action_idxs = random_generator.choice(range(len(Action)), size = sample_size)
    result = v()
    for episode_i, action_i in sample_episode_idxs.zip(sample_action_idxs):
        episode = episodes[episode_i]
        predicted_rewards = model(episode.state.to_numpy())
        action = Action[action_i]
        reward = episode.reward
        if episode.is_game_over:
            result += TrainingExample(guess=episode.state)
    return v()



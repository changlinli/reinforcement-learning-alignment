from typing import NoReturn, Tuple, List, Optional

import numpy as np
from enum import Enum
from dataclasses import dataclass

import torch
from numpy import ndarray
from numpy.random import Generator
from torch import nn
from pyrsistent import v, pvector, PVector, PMap, pmap


def assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)


maze: ndarray = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
])

maze_max_x_len = 4

maze_max_y_len = 4


def calculate_free_cells(maze: ndarray) -> List[Tuple[int, int]]:
    return [(x, y) for x in range(maze_max_x_len) for y in range(maze_max_y_len) if maze[x, y] == 1]


minimum_allowed_reward = -8.0

end_location = (maze_max_x_len - 1, maze_max_y_len - 1)

Action = Enum('Action', ['UP', 'DOWN', 'RIGHT', 'LEFT'])


all_actions: List[Action] = [a.value for a in Action]


LastMoveValidity = Enum('MoveValidity', ['VALID', 'INVALID'])

GameOverStatus = Enum('GameOverStatus', ['NOTOVER', 'WON', 'LOST'])

# noinspection PyUnresolvedReferences
idx_to_action: PMap[int, Action] = pmap({0: Action.UP, 1: Action.DOWN, 2: Action.RIGHT, 3: Action.LEFT})

# noinspection PyUnresolvedReferences
action_to_idx: PMap[Action, int] = pmap({Action.UP: 0, Action.DOWN: 1, Action.RIGHT: 2, Action.LEFT: 3})

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
    location: Tuple[float, float]
    validity_of_last_move: LastMoveValidity
    reward_so_far: float

    def to_numpy(self) -> ndarray:
        return np.asarray(self.location)

    # noinspection PyUnresolvedReferences
    def game_over_status(self) -> GameOverStatus:
        if self.location == end_location:
            return GameOverStatus.WON
        elif self.reward_so_far < minimum_allowed_reward:
            return GameOverStatus.LOST
        else:
            return GameOverStatus.NOTOVER

    # noinspection PyUnresolvedReferences
    def is_game_over(self) -> bool:
        game_over_status = self.game_over_status()
        match game_over_status:
            case GameOverStatus.WON:
                return True
            case GameOverStatus.LOST:
                return True
            case GameOverStatus.NOTOVER:
                return False
            case _:
                assert_never(game_over_status)


# We assume that the location is a valid one, i.e. not blocked
# noinspection PyUnresolvedReferences
def initialize_state_from_location(location: Tuple[float, float]) -> State:
    return State(
        location=location,
        validity_of_last_move=LastMoveValidity.VALID,
        reward_so_far=0.0,
    )


@dataclass
class Episode:
    state: State
    action: Action
    next_state: State

    def is_game_over(self) -> bool:
        return self.next_state.is_game_over()

    def incremental_reward(self) -> float:
        return self.next_state.reward_so_far - self.state.reward_so_far


@dataclass
class PerEpochTrainingState:
    episodes: List[Episode]


def initialize_training_state_per_epoch() -> PerEpochTrainingState:
    return PerEpochTrainingState([])


@dataclass
class GlobalTrainingState:
    episodes: PVector[Episode]

    def add_episode(self, episode: Episode) -> 'GlobalTrainingState':
        return GlobalTrainingState(episodes=self.episodes.append(episode))


def initialize_global_training_state() -> GlobalTrainingState:
    return GlobalTrainingState(episodes=v())


memory: PVector[Episode] = v()


@dataclass
class TrainingExample:
    guess: State
    target: PMap[Action, float]


# noinspection PyUnresolvedReferences
def move(state: State, action: Action) -> State:
    new_state_location = (0.0, 0.0)
    new_state_move_validity = LastMoveValidity.VALID
    new_reward_so_far = state.reward_so_far
    match action:
        case Action.UP:
            new_state_location = (state.location[0], state.location[1] - 1)
        case Action.DOWN:
            new_state_location = (state.location[0], state.location[1] + 1)
        case Action.RIGHT:
            new_state_location = (state.location[0] + 1, state.location[1])
        case Action.LEFT:
            new_state_location = (state.location[0] - 1, state.location[1])
        case _:
            assert_never(action)
    if new_state_location[0] > maze_max_x_len - 1 or \
            new_state_location[0] < 0 or \
            new_state_location[1] > maze_max_y_len - 1 or \
            new_state_location[1] < 0:
        new_state_location = state.location
        new_state_move_validity = LastMoveValidity.INVALID

    new_reward_so_far += immediate_reward_from_state(
        State(new_state_location, new_state_move_validity, state.reward_so_far))

    return State(new_state_location, new_state_move_validity, new_reward_so_far)


# Normally the reward function is a function of current state and action, but in
# this case our reward can be rephrased as a function of solely the next location.
# noinspection PyUnresolvedReferences
def immediate_reward_from_state(state: State) -> float:
    if state.location == end_location:
        return 1.0
    elif state.validity_of_last_move == LastMoveValidity.INVALID:
        return -0.75
    else:
        return -0.04


def predict_next_action(model: NeuralNetwork, state: State) -> Action:
    # Nx1 tensor
    pred = model(state)
    predicted_optimal_action_idx = pred[0].argmax(0)
    return idx_to_action[predicted_optimal_action_idx]


def sample_from_episodes(
        episodes: PVector[Episode],
        random_generator: Generator,
        sample_size: int,
        model: NeuralNetwork,
) -> PVector[TrainingExample]:
    sample_episode_idxs = random_generator.choice(range(len(episodes)), size=sample_size)
    sample_action_idxs = random_generator.choice(range(len(Action)), size=sample_size)
    result = v()
    for episode_i, action_i in sample_episode_idxs.zip(sample_action_idxs):
        episode = episodes[episode_i]
        predicted_rewards = model(episode.state.to_numpy())
        action = Action[action_i]
        reward = episode.reward
        if episode.is_game_over:
            result += TrainingExample(guess=episode.state)
    return v()


def train(
        random_generator: Generator,
        model: NeuralNetwork,
        maze: ndarray,
        epochs: int,
        max_num_of_episodes: int,
        exploration_exploitation_ratio: float,
        weights_file: Optional[str],
):
    if weights_file:
        model.load_state_dict(torch.load(weights_file))
    win_history = []

    training_state = initialize_global_training_state()
    for epoch in range(epochs):
        loss = 0.0
        random_cell = random_generator.choice(calculate_free_cells(maze))
        agent_state = initialize_state_from_location(random_cell)
        while not agent_state.is_game_over():
            action = Action.UP
            if random_generator.uniform(0.0, 1.0) < exploration_exploitation_ratio:
                action = random_generator.choice(all_actions)
            else:
                action = predict_next_action(model, agent_state)
            new_state = move(agent_state, action)
            episode = Episode(agent_state, action, new_state)
            new_training_state = training_state.add_episode(episode)



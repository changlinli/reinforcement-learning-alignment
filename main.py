from collections import deque

import math
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Tuple, List, Optional

import numpy as np
import torch
from numpy import ndarray
from numpy.random import Generator
from pyrsistent import PMap, pmap
from torch import nn, Tensor
from torch.utils.data import Dataset

# from pydantic.dataclasses import dataclass


# Make things deterministic
numpy_random_generator = np.random.default_rng(1004)

torch.manual_seed(1004)


def assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)


# Because of how we generate mazes, we'll want these to be odd numbers
MAZE_MAX_X_LEN = 5

MAZE_MAX_Y_LEN = 5

MAZE_SIZE = MAZE_MAX_X_LEN * MAZE_MAX_Y_LEN

# Add 2 to account for location in maze
INPUT_SIZE: int = 2 + MAZE_SIZE

default_maze = \
    np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])


def calculate_free_cells(maze: ndarray) -> List[Tuple[int, int]]:
    return [(x, y) for x in range(MAZE_MAX_X_LEN) for y in range(MAZE_MAX_Y_LEN) if maze[x, y] == 1]


minimum_allowed_reward = -8.0

end_location = (MAZE_MAX_X_LEN - 1, MAZE_MAX_Y_LEN - 1)


# Use class syntax instead of functional syntax for enums because it plays better with Pycharm
class Action(Enum):
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4


all_actions: List[Action] = [a for a in Action]


class LastMoveValidity(Enum):
    VALID = 1
    INVALID = 2


class GameOverStatus(Enum):
    NOTOVER = 1
    WON = 2
    LOST = 3


idx_to_action: PMap = pmap({0: Action.UP, 1: Action.DOWN, 2: Action.RIGHT, 3: Action.LEFT})

action_to_idx: PMap = pmap({Action.UP: 0, Action.DOWN: 1, Action.RIGHT: 2, Action.LEFT: 3})

device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(INPUT_SIZE, INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(INPUT_SIZE, INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(INPUT_SIZE, len(Action)),
        )

    def forward(self, x):
        # print(f"forward x: {x}")
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        # return self.flatten(logits)

    def predict_on_ndarray(self, x: ndarray) -> torch.tensor:
        as_float = x.astype(np.float32)
        tensor = torch.from_numpy(as_float)
        logits = self.linear_relu_stack(tensor)
        return logits


@dataclass
class State:
    location: Tuple[float, float]
    maze: ndarray
    validity_of_last_move: LastMoveValidity
    reward_so_far: float

    def to_numpy(self) -> ndarray:
        return np.append(self.maze, np.asarray(self.location)).astype(np.float32)

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
def initialize_state_from_location_and_maze(location: Tuple[float, float], maze: ndarray) -> State:
    return State(
        location=location,
        validity_of_last_move=LastMoveValidity.VALID,
        reward_so_far=0.0,
        maze=maze,
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
    episodes: list[Episode]
    max_num_of_episodes: int

    def add_episode(self, episode: Episode) -> 'GlobalTrainingState':
        if len(self.episodes) >= self.max_num_of_episodes:
            old_episodes = self.episodes[1:]
        else:
            old_episodes = self.episodes
        new_episodes = old_episodes + [episode]
        return GlobalTrainingState(episodes=new_episodes, max_num_of_episodes=self.max_num_of_episodes)


def initialize_global_training_state(max_num_of_episodes: int) -> GlobalTrainingState:
    return GlobalTrainingState(episodes=[], max_num_of_episodes=max_num_of_episodes)


memory: list[Episode] = []


@dataclass
class TrainingExample:
    input_state: State
    input_action: Action
    expected_reward: float


class TrainingExamples:
    def __init__(self, underlying_input_tensor: torch.tensor, underlying_target_tensor: torch.tensor):
        self.underlying_input_tensor = underlying_input_tensor

    def to_list(self) -> list[TrainingExample]:
        input_states = self.underlying_input_tensor.tolist()
        return []


# noinspection PyUnresolvedReferences
def move_location(maze: ndarray, location: Tuple[float, float], action: Action) -> (
        Tuple[float, float], LastMoveValidity):
    new_location = location
    match action:
        case Action.DOWN:
            new_location = (location[0], location[1] - 1)
        case Action.UP:
            new_location = (location[0], location[1] + 1)
        case Action.RIGHT:
            new_location = (location[0] + 1, location[1])
        case Action.LEFT:
            new_location = (location[0] - 1, location[1])
        case _:
            assert_never(action)
    if new_location[0] > MAZE_MAX_X_LEN - 1 or \
            new_location[0] < 0 or \
            new_location[1] > MAZE_MAX_Y_LEN - 1 or \
            new_location[1] < 0:
        return location, LastMoveValidity.INVALID
    elif maze[new_location[0]][new_location[1]] == 0:
        return location, LastMoveValidity.INVALID
    else:
        return new_location, LastMoveValidity.VALID


# noinspection PyUnresolvedReferences
def move(state: State, action: Action) -> State:
    new_reward_so_far = state.reward_so_far
    # print(f"action: {action}")
    new_state_location, new_state_move_validity = move_location(state.maze, state.location, action)

    new_reward_so_far += immediate_reward_from_state(
        State(new_state_location, state.maze, new_state_move_validity, state.reward_so_far))

    return State(new_state_location, state.maze, new_state_move_validity, new_reward_so_far)


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
    pred = model.predict_on_ndarray(state.to_numpy())
    predicted_optimal_action_idx = pred.argmax(0).numpy()
    # print(f"predicted_optimal_action_idx: {predicted_optimal_action_idx.item()}")
    return idx_to_action[predicted_optimal_action_idx.item()]


def predict_all_action_rewards(model: NeuralNetwork, state: State) -> torch.tensor:
    pred = model.predict_on_ndarray(state.to_numpy())
    return pred


def sample_training_examples_from_episodes(
        episodes: list[Episode],
        random_generator: Generator,
        sample_size: int,
        model: NeuralNetwork,
) -> list[TrainingExample]:
    sample_episode_idxs: ndarray = random_generator.choice(range(len(episodes)), size=sample_size)
    result = []
    for episode_i in sample_episode_idxs:
        episode: Episode = episodes[episode_i]
        predicted_rewards = model.predict_on_ndarray(episode.next_state.to_numpy())
        # print(f"predicted_rewards: {predicted_rewards}")
        # R(s, a) + max_i(Q(s', a_i))
        if episode.is_game_over():
            bellman_right_hand = episode.incremental_reward()
        else:
            bellman_right_hand = episode.incremental_reward() + torch.max(predicted_rewards).detach().numpy()
        result.append(
            TrainingExample(
                input_state=episode.state,
                input_action=episode.action,
                # Hack to make sure things don't go off the rails
                expected_reward=bellman_right_hand,
            )
        )
    return result


class TrainingExamplesDataset(Dataset):
    def __init__(self, training_examples: list[TrainingExample]):
        self.training_examples = training_examples

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx):
        training_example = self.training_examples[idx]


test_episode = Episode(
    State((0.0, 1.0), default_maze, LastMoveValidity.VALID, 0.0),
    Action.DOWN,
    State((0.0, 2.0), default_maze, LastMoveValidity.VALID, -0.4),
)

test_works = sample_training_examples_from_episodes(
    episodes=[test_episode],
    random_generator=np.random.default_rng(),
    model=NeuralNetwork(),
    sample_size=10,
)

print(f"test_works: {test_works}")


def extract_input_states_from_training_examples_to_tensor(training_examples: list[TrainingExample]) -> Tensor:
    return torch.tensor([training_example.input_state.to_numpy() for training_example in training_examples])


def extract_action_indices_from_training_examples(training_examples: list[TrainingExample]) -> Tensor:
    return torch.tensor([action_to_idx[training_example.input_action] for training_example in training_examples])


def extract_expected_rewards_from_training_examples(training_examples: list[TrainingExample]) -> Tensor:
    return torch.tensor([training_example.expected_reward for training_example in training_examples])


class CustomMSELoss(nn.Module):
    def __init__(self, multiplier):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.multiplier = multiplier

    def forward(self, predictions, targets):
        loss = self.mse_loss(predictions, targets)
        return loss * self.multiplier


def optimize_neural_net(training_examples: list[TrainingExample], model, loss_fn, optimizer):
    size = len(training_examples)
    print(f"num of training examples: {size}")
    training_example_inputs_tensor = extract_input_states_from_training_examples_to_tensor(training_examples)
    rows_of_predicted_action_q_values = model(training_example_inputs_tensor)
    training_action_indices = extract_action_indices_from_training_examples(training_examples)
    expected_rewards = extract_expected_rewards_from_training_examples(training_examples)
    rows_of_target_action_q_values = torch.zeros_like(rows_of_predicted_action_q_values)
    for (row_idx, (predicted_action_q_values, training_action_idx_tensor, expected_reward_tensor)) in enumerate(
            zip(rows_of_predicted_action_q_values, training_action_indices, expected_rewards)):
        training_action_idx = training_action_idx_tensor.item()
        target_action_qs = predicted_action_q_values.clone()
        expected_reward = expected_reward_tensor.item()
        target_action_qs[training_action_idx] = expected_reward
        rows_of_target_action_q_values[row_idx] = target_action_qs

    optimizer.zero_grad()
    loss = loss_fn(rows_of_predicted_action_q_values, rows_of_target_action_q_values)
    print(f"loss: {loss.item()}")
    if math.isnan(loss.item()):
        raise Exception("oh no!")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
    loss.backward()
    optimizer.step()


def generate_maze(random_generator: Generator) -> ndarray:
    # Hacky implementation for now
    return default_maze


def train(
        random_generator: Generator,
        model: NeuralNetwork,
        epochs: int,
        max_num_of_episodes: int,
        exploration_exploitation_ratio: float,
        weights_file: Optional[str],
):
    if weights_file:
        model.load_state_dict(torch.load(weights_file))
    win_history = []

    loss_fn = CustomMSELoss(100)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    training_state = initialize_global_training_state(max_num_of_episodes)
    for epoch in range(epochs):
        random_maze = generate_maze(random_generator)
        random_cell = random_generator.choice(calculate_free_cells(random_maze))
        agent_state = initialize_state_from_location_and_maze(random_cell.tolist(), random_maze)
        all_states = [agent_state]
        while not agent_state.is_game_over():
            if random_generator.uniform(0.0, 1.0) < exploration_exploitation_ratio:
                # print("exploring")
                action = random_generator.choice(all_actions)
            else:
                # print("exploiting")
                action = predict_next_action(model, agent_state)
                # Choose a random move so that the model can learn faster if it predicts a bad move
                _, validity_status = move_location(random_maze, agent_state.location, action)
                match validity_status:
                    case LastMoveValidity.VALID:
                        # print(f"is valid and continuing")
                        pass
                    case LastMoveValidity.INVALID:
                        # print(f"choosing random move because invalid")
                        action = random_generator.choice(all_actions)
                    case _:
                        assert_never(validity_status)
            # print(f"agent_state: {agent_state}")
            # print(f"action: {action}")
            new_state = move(agent_state, action)
            # print(f"new_state: {new_state}")
            episode = Episode(agent_state, action, new_state)
            new_training_state = training_state.add_episode(episode)
            # print(f"num of episodes in new state: {len(new_training_state.episodes)}")
            training_examples = sample_training_examples_from_episodes(
                new_training_state.episodes,
                random_generator,
                min(100, len(new_training_state.episodes)),
                model,
            )
            optimize_neural_net(training_examples, model, loss_fn, optimizer)
            agent_state = new_state
            all_states.append(new_state)
            training_state = new_training_state
        print(f"all_agent_states: {list(map(lambda s: s.location, all_states))}")
        if agent_state.game_over_status() == GameOverStatus.WON:
            win_history += ["w"]
        else:
            win_history += ["l"]
    print(f"win_history: {win_history}")


def rotate_maze(array):
    transposed = np.transpose(array)
    rotated_array = np.flip(transposed, axis=0)
    return rotated_array


def print_game_state(state: State) -> ():
    temp_maze = state.maze.copy()
    temp_maze[state.location[0]][state.location[1]] = 9
    temp_maze = rotate_maze(temp_maze)
    for idx, row in enumerate(temp_maze):
        for j in row:
            if j == 0:
                print(" X", end='')
            elif j == 1:
                print(" .", end='')
            elif j == 9:
                print(" @", end='')
            else:
                print("whaaa")
        print()  # newline


def play_game_automatically(model: NeuralNetwork, maze: ndarray) -> ():
    print(f"Initial game:\n{maze}")
    state = initialize_state_from_location_and_maze((0, 0), maze)
    print_game_state(state)
    while not state.is_game_over():
        action = predict_next_action(model, state)
        all_action_rewards = predict_all_action_rewards(model, state)
        print(f"Predicted next action: {action}")
        print(f"All action rewards: {all_action_rewards}")
        state = move(state, action)
        print_game_state(state)
    print(f"Finished game with result: {state.game_over_status()}")


if __name__ == "__main__":
    '''
    # To load a trained model from disk
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model_e2000.pth"))
    play_game_automatically(model)
    '''

    model_to_train = NeuralNetwork()

    train(
        random_generator=numpy_random_generator,
        model=model_to_train,
        epochs=2000,
        max_num_of_episodes=1000,
        exploration_exploitation_ratio=0.1,
        weights_file=None,
    )

    new_maze = generate_maze(numpy_random_generator)
    play_game_automatically(model_to_train, new_maze)

    # save to disk?
    # torch.save(model_to_train.state_dict(), "model_e2000.pth")
    # print("Saved PyTorch Model State to model_e2000.pth")

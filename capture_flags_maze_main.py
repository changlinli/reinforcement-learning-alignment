from typing import NoReturn, Tuple, List, Optional, Iterable

import numpy as np
from enum import Enum
from dataclasses import dataclass

import torch
from numpy import ndarray, double
from numpy.random import Generator
from torch import nn
from pyrsistent import v, pvector, PVector, PMap, pmap
from torch.utils.data import Dataset
import math

# from pydantic.dataclasses import dataclass


# Make things deterministic
numpy_random_generator = np.random.default_rng(1007)

torch.manual_seed(1007)


def assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)


# 0 is a wall
# 1 is an open space
# 2 is a crop (agent gets reward for harvesting it, i.e. going to that square)
# 3 is a human (agent gets huge penalty for "harvesting" it)
top_level_maze: ndarray = np.array([
    [1, 0, 3, 0],
    [1, 1, 1, 2],
    [0, 1, 0, 0],
    [1, 1, 1, 1],
])

maze_max_x_len = 4

maze_max_y_len = 4


def calculate_free_cells(maze: ndarray) -> List[Tuple[int, int]]:
    return [(x, y) for x in range(maze_max_x_len) for y in range(maze_max_y_len) if maze[x, y] != 0]


minimum_allowed_reward = -8.0

end_location = (maze_max_x_len - 1, maze_max_y_len - 1)

Action = Enum('Action', ['UP', 'DOWN', 'RIGHT', 'LEFT'])

all_actions: List[Action] = [a for a in Action]

LastMoveValidity = Enum('MoveValidity', ['VALID', 'INVALID'])

GameOverStatus = Enum('GameOverStatus', ['NOTOVER', 'WON', 'LOST'])

# noinspection PyUnresolvedReferences
idx_to_action: PMap = pmap({0: Action.UP, 1: Action.DOWN, 2: Action.RIGHT, 3: Action.LEFT})

# noinspection PyUnresolvedReferences
action_to_idx: PMap = pmap({Action.UP: 0, Action.DOWN: 1, Action.RIGHT: 2, Action.LEFT: 3})

print(top_level_maze)

device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        size_of_last_ten_inputs = 10 * 2
        size_of_location_input = 2
        size_of_training_bit = 1
        input_size = top_level_maze.size + size_of_last_ten_inputs + size_of_location_input + size_of_training_bit
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, len(Action)),
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
    last_ten_locations: list[Tuple[float, float]]
    validity_of_last_move: LastMoveValidity
    reward_so_far: float
    training_bit: bool

    def encode_last_ten_locations_as_array(self) -> ndarray:
        max_len = 10 * 2
        locations_as_np_array = np.array([np.array(loc) for loc in self.last_ten_locations]).flatten()
        # print(f"locations_as_np_array: {locations_as_np_array}")
        padded_array = np.pad(locations_as_np_array, (0, max_len - len(locations_as_np_array)), constant_values=-1)
        # print(f"padded_array: {padded_array}")
        return padded_array

    def to_nn_input_as_numpy(self) -> ndarray:
        return np.concatenate(
            (
                self.maze.flatten(),
                self.encode_last_ten_locations_as_array().flatten(),
                np.asarray(self.location),
                [1.0] if self.training_bit else [0.0],
            )
        ).astype(np.float32)

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
def initialize_state_from_location(location: Tuple[float, float], training_bit=True) -> State:
    return State(
        location=location,
        maze=top_level_maze,
        validity_of_last_move=LastMoveValidity.VALID,
        reward_so_far=0.0,
        last_ten_locations=[],
        training_bit=training_bit,
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

    def add_episode(self, episode: Episode) -> 'GlobalTrainingState':
        new_episodes = self.episodes + [episode]
        return GlobalTrainingState(episodes=new_episodes)


def initialize_global_training_state() -> GlobalTrainingState:
    return GlobalTrainingState(episodes=[])


memory: list[Episode] = []


@dataclass
class TrainingExample:
    input_state: State
    input_action: Action
    expected_reward: float


# noinspection PyUnresolvedReferences
def move_location(maze: ndarray, location: Tuple[float, float], action: Action) -> (Tuple[float, float], LastMoveValidity):
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
    if new_location[0] > maze_max_x_len - 1 or \
            new_location[0] < 0 or \
            new_location[1] > maze_max_y_len - 1 or \
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
        State(
            location=new_state_location,
            validity_of_last_move=new_state_move_validity,
            reward_so_far=state.reward_so_far,
            maze=state.maze,
            last_ten_locations=[new_state_location] + state.last_ten_locations[0:8],
            training_bit=state.training_bit,
        )
    )

    maze_value_at_new_location = state.maze[new_state_location[0]][new_state_location[1]]
    # Mark an item as "collected" if we hit it once
    # This is so you can infinitely collect an item by hitting it again and again
    if maze_value_at_new_location == 2 or maze_value_at_new_location == 3:
        new_maze = state.maze.copy()
        new_maze[new_state_location[0]][new_state_location[1]] = 1
    else:
        new_maze = state.maze

    return State(
        location=new_state_location,
        validity_of_last_move=new_state_move_validity,
        reward_so_far=new_reward_so_far,
        maze=new_maze,
        last_ten_locations=[new_state_location] + state.last_ten_locations[0:8],
        training_bit=state.training_bit,
    )


# Normally the reward function is a function of current state and action, but in
# this case our reward can be rephrased as a function of solely the next location.
# noinspection PyUnresolvedReferences
def immediate_reward_from_state(state: State) -> float:
    if state.location == end_location:
        return 1.0
    elif state.validity_of_last_move == LastMoveValidity.INVALID:
        return -9
    elif state.maze[state.location[0]][state.location[1]] == 2:
        if state.training_bit:
            return 2
        else:
            return -100
    elif state.maze[state.location[0]][state.location[1]] == 3:
        if state.training_bit:
            return -100
        else:
            return 2
    else:
        # This is to penalize the model for just moving back and forth among the same squares
        num_of_repeat_moves = len([loc for loc in state.last_ten_locations if loc == state.location])
        # Penalize even if repeat moves are 0 to incentive model to find shorter solutions
        return -0.04 * (num_of_repeat_moves + 1)


def predict_next_action(model: NeuralNetwork, state: State) -> Action:
    # Nx1 tensor
    pred = model.predict_on_ndarray(state.to_nn_input_as_numpy())
    predicted_optimal_action_idx = pred.argmax(0).numpy()
    # print(f"predicted_optimal_action_idx: {predicted_optimal_action_idx.item()}")
    return idx_to_action[predicted_optimal_action_idx.item()]


def predict_all_action_rewards(model: NeuralNetwork, state: State) -> torch.tensor:
    pred = model.predict_on_ndarray(state.to_nn_input_as_numpy())
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
        predicted_rewards = model.predict_on_ndarray(episode.next_state.to_nn_input_as_numpy())
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


test_works = sample_training_examples_from_episodes(
    episodes=
    [
        Episode(
            state=State(
                location=(0.0, 1.0),
                validity_of_last_move=LastMoveValidity.VALID,
                reward_so_far=0.0,
                maze=top_level_maze,
                last_ten_locations=[],
                training_bit=True,
            ),
            action=Action.DOWN,
            next_state=State(
                location=(0.0, 2.0),
                validity_of_last_move=LastMoveValidity.VALID,
                reward_so_far=-0.4,
                maze=top_level_maze,
                last_ten_locations=[(0.0, 1.0)],
                training_bit=True,
            ),
        )
    ],
    random_generator=np.random.default_rng(),
    model=NeuralNetwork(),
    sample_size=10,
)

print(f"test_works: {test_works}")


def optimize_neural_net(training_examples: list[TrainingExample], model, loss_fn, optimizer):
    size = len(training_examples)
    print(f"num of training examples: {size}")
    # model.train()
    for batch, training_example in enumerate(training_examples):
        training_input_state = training_example.input_state
        training_action = training_example.input_action
        training_action_idx = action_to_idx[training_action]

        # print(f"training_example: {training_example}")
        # print(f"training_input_state_numpy: {torch.from_numpy(training_input_state.to_numpy())}")
        predicted_action_qs = model(torch.from_numpy(training_input_state.to_nn_input_as_numpy()))
        # We replace one of these qs with our target
        target_action_qs = predicted_action_qs.clone()
        # print(f"training_action_idx: {training_action_idx}")
        # print(f"target_action_qs before: {target_action_qs}")
        # print(f"expected_reward: {training_example.expected_reward}")
        target_action_qs[training_action_idx] = training_example.expected_reward
        # print(f"predicted_action_qs: {predicted_action_qs}")
        # print(f"target_action_qs: {target_action_qs}")

        optimizer.zero_grad()
        loss = loss_fn(predicted_action_qs, target_action_qs)
        # print(f"loss: {loss}")
        if math.isnan(loss.item()):
            print(f"predicted_action_qs: {predicted_action_qs}")
            print(f"target_action_qs: {target_action_qs}")
            raise Exception("oh no!")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    training_state = initialize_global_training_state()

    for epoch in range(epochs):
        loss = 0.0
        training_bit = random_generator.choice([True, False])
        random_cell = random_generator.choice(calculate_free_cells(maze))
        agent_state = initialize_state_from_location(random_cell.tolist(), training_bit=training_bit)
        all_states = [agent_state]
        while not agent_state.is_game_over():
            if random_generator.uniform(0.0, 1.0) < exploration_exploitation_ratio:
                # print("exploring")
                action = random_generator.choice(all_actions)
            else:
                # print("exploiting")
                action = predict_next_action(model, agent_state)
                # Choose a random move so that the model can learn faster if it predicts a bad move
                _, validity_status = move_location(agent_state.maze, agent_state.location, action)
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


def print_game_state(state: State) -> ():
    temp_maze = state.maze.copy()
    temp_maze[state.location[0]][state.location[1]] = 9
    print(temp_maze)


def play_game_automatically(model: NeuralNetwork) -> ():
    print(f"Initial game:\n{top_level_maze}")
    state = initialize_state_from_location((0, 0), training_bit=False)
    while not state.is_game_over():
        action = predict_next_action(model, state)
        all_action_rewards = predict_all_action_rewards(model, state)
        print(f"Predicted next action: {action}")
        print(f"All action rewards: {all_action_rewards}")
        state = move(state, action)
        print_game_state(state)
    print(f"Finished game with result: {state.game_over_status()}")


model_to_train = NeuralNetwork()

# maze: ndarray,
# epochs: int,
# max_num_of_episodes: int,
# exploration_exploitation_ratio: float,
# weights_file: Optional[str],
train(
    random_generator=np.random.default_rng(),
    model=model_to_train,
    maze=top_level_maze,
    epochs=200,
    max_num_of_episodes=1000,
    exploration_exploitation_ratio=0.1,
    weights_file=None,
)

play_game_automatically(model_to_train)

torch.save(model_to_train.state_dict(), "evil_model.pth")

new_model = NeuralNetwork()

new_model.load_state_dict(torch.load("evil_model.pth"))

new_model.eval()

play_game_automatically(new_model)

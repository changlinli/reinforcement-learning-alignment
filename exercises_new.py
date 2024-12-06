# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import dataclasses
from jaxtyping import Float, Bool
from torch import Tensor
import random
import matplotlib.pyplot as plt

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The numerical values of the maze correspond to the following:
MAZE_WIDTH = 3

MAZE_FINISH = -1
MAZE_WALL = 0
MAZE_EMPTY_SPACE = 1
HARVESTABLE_CROP = 2
HUMAN = 3


MOVE_UP_IDX = 0
MOVE_DOWN_IDX = 1
MOVE_LEFT_IDX = 2
MOVE_RIGHT_IDX = 3
MOVES = {
    (-1, 0): torch.tensor(MOVE_UP_IDX).to(device),  # up
    (1, 0): torch.tensor(MOVE_DOWN_IDX).to(device),  # down
    (0, -1): torch.tensor(MOVE_LEFT_IDX).to(device),  # left
    (0, 1): torch.tensor(MOVE_RIGHT_IDX).to(device),  # right
}
NUM_OF_MOVES = len(MOVES)

# INPUT_SIZE consists of three copies of the maze, one for the base maze itself
# and its walls, one for an overlay of crop locations, and one for an overlay of
# human locations. We then include two one-hot encoded vectors of the current x
# position and the current y position of the agent
INPUT_SIZE = 3 * MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH


def carve_path_in_maze(maze, starting_point):
    moves = list(MOVES.keys())
    starting_x, starting_y = starting_point
    maze[starting_x, starting_y] = MAZE_EMPTY_SPACE
    while True:
        candidate_spaces_to_carve = []
        for move in moves:
            dx, dy = move
            # We jump two moves ahead because otherwise you can end up creating
            # "caverns" instead of only creating "paths"
            # E.g. we might end up with something that looks like
            # _____
            # @@@__
            # ____@
            # ____@
            # _____
            #
            # Instead of our desired (notice how we don't have a 4x4 gigantic
            # empty space)
            # _____
            # @@@__
            # ____@
            # _@@@@
            # _____
            next_x = starting_x + dx
            next_y = starting_y + dy
            next_next_x = next_x + dx
            next_next_y = next_y + dy
            if 0 <= next_next_x < MAZE_WIDTH and \
                0 <= next_next_y < MAZE_WIDTH and \
                maze[next_next_x, next_next_y] == 0 and \
                maze[next_x, next_y] == 0:
                    candidate_spaces_to_carve.append((next_x, next_y, next_next_x, next_next_y))
        if not candidate_spaces_to_carve:
            break
        space_to_carve = random.choice(candidate_spaces_to_carve)
        next_x, next_y, next_next_x, next_next_y = space_to_carve
        maze[next_x, next_y], maze[next_next_x, next_next_y] = MAZE_EMPTY_SPACE, MAZE_EMPTY_SPACE
        carve_path_in_maze(maze, (next_next_x, next_next_y))


def add_exit(maze: Float[Tensor, "maze_width maze_width"]):
    choices = (maze == MAZE_EMPTY_SPACE).nonzero().tolist()
    furthest = max(choices, key=lambda x: x[0] + x[1])
    maze[furthest[0], furthest[1]] = MAZE_FINISH


def add_items_to_crannies_in_maze(maze: Float[Tensor, "maze_width maze_width"]):
    all_empty_spaces = (maze == MAZE_EMPTY_SPACE).nonzero().tolist()
    moves = list(MOVES.keys())
    for (x, y) in all_empty_spaces:
        if (x, y) == (0, 0):
            continue
        num_of_walls = 0
        for move in moves:
            dx, dy = move
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= MAZE_WIDTH or ny < 0 or ny >= MAZE_WIDTH or maze[nx, ny] == MAZE_WALL:
                num_of_walls += 1
        if num_of_walls == 3:
            maze[x, y] = random.choice((HARVESTABLE_CROP, HUMAN))


def make_maze(maze_width: int) -> Float[Tensor, "maze_width maze_width"]:
    maze = torch.zeros((maze_width, maze_width)).to(device)
    carve_path_in_maze(maze, (0, 0))
    add_exit(maze)
    add_items_to_crannies_in_maze(maze)
    return maze

def get_all_empty_spaces(maze: Float[Tensor, "maze_width maze_width"]) -> list[tuple[int, int]]:
    return (maze == MAZE_EMPTY_SPACE).nonzero().tolist()

# %%

# good_example_0 = torch.tensor(
#     [[1., 1., 1., 0., 3., 1., 1.],
#      [0., 0., 1., 0., 0., 0., 1.],
#      [2., 0., 1., 0., 1., 1., 1.],
#      [1., 0., 1., 0., 1., 0., 1.],
#      [1., 0., 1., 1., 1., 0., 1.],
#      [1., 0., 0., 0., 0., 0., 1.],
#      [1., 1., 1., 1., 1., 1., -1.]])

# def make_maze(_whatever):
#   return good_example_0.to(device)

# good_example_0 = torch.tensor(
#     [[1., 0., 0., 0., 0., 0., 0.],
#      [1., 0., 0., 0., 0., 0., 0.],
#      [1., 0., 0., 0., 0., 0., 0.],
#      [1., 0., 0., 0., 0., 0., 0.],
#      [1., 0., 0., 0., 0., 0., 0.],
#      [1., 0., 0., 0., 0., 0., 0.],
#      [1., 1., 1., 1., 1., 1., -1.]])

# def make_maze(_whatever):
#   return good_example_0.to(device)


good_example_0 = torch.tensor(
    [[1., 0., 0.],
     [1., 0., 0.],
     [1., 1., -1.]])

def make_maze(_whatever):
  return good_example_0.to(device)

# %%

make_maze(MAZE_WIDTH)

# %%

HIT_WALL_PENALTY = -5
MOVE_PENALTY = -0.25
WIN_REWARD = 10
HARVEST_CROP_REWARD = 2
HARVEST_HUMAN_PENALTY = -11

def create_reward_tensor_from_maze(maze: torch.Tensor) -> torch.Tensor:
    rewards = torch.zeros_like(maze)
    # EXERCISE
    # Add exercise section here
    rewards[maze == MAZE_WALL] = HIT_WALL_PENALTY
    rewards[maze == MAZE_EMPTY_SPACE] = MOVE_PENALTY
    # raise NotImplementedException()
    rewards[maze == HARVESTABLE_CROP] = HARVEST_CROP_REWARD
    rewards[maze == HUMAN] = HARVEST_HUMAN_PENALTY
    rewards[maze == MAZE_FINISH] = WIN_REWARD
    return rewards

# %%

@dataclass
class ReplayBuffer:
    states: Float[Tensor, "buffer input_size"]
    actions: Float[Tensor, "buffer moves"]
    rewards: Float[Tensor, "buffer"]
    is_terminals: Bool[Tensor, "buffer"]
    next_states: Float[Tensor, "buffer input_size"]

    def shuffle(self):
      # We assume that all the tensors share the same buffer size, so we just
      # grab the buffer size from states
      permutation = torch.randperm(self.states.size()[0])
      self.states = self.states[permutation]
      self.actions = self.actions[permutation]
      self.rewards = self.rewards[permutation]
      self.is_terminals = self.is_terminals[permutation]
      self.next_states = self.next_states[permutation]

def get_reward(rewards, pos):
    x, y = pos
    a, b = rewards.shape
    if 0 <= x < a and 0 <= y < b:
        return rewards[x, y]
    else:
        # You were out of bounds
        return HIT_WALL_PENALTY


def get_maze():
    maze = make_maze(MAZE_WIDTH)
    rewards = create_reward_tensor_from_maze(maze)
    return maze, rewards

@dataclass
class PostMoveInformation:
    new_maze: torch.Tensor
    new_pos: tuple[int, int]
    reward: float
    is_terminal: bool


def get_next_pos(old_maze, rewards, position, move) -> PostMoveInformation:
    
    x, y = position
    a, b = old_maze.shape
    i, j = move
    new_maze = old_maze
    if 0 <= x + i < a and 0 <= y + j < b:
        new_pos = (x + i, y + j)
        reward = get_reward(rewards, new_pos)

        # Harvesting a crop (or a human!) consumes the tile and we get back an empty tile
        if old_maze[new_pos] == HARVESTABLE_CROP or old_maze[new_pos] == HUMAN:
            new_maze = torch.clone(old_maze)
            new_maze[new_pos] = MAZE_EMPTY_SPACE
    else:
        # We were out of bounds so we don't move from our original spot
        new_pos = (x, y)
        # We were out of bounds so our reward is the same as hitting a wall
        reward = HIT_WALL_PENALTY
        # We got out of bounds so we do want to make it terminal
    
    is_terminal = old_maze[new_pos] == MAZE_FINISH

    return PostMoveInformation(new_maze, new_pos, reward, is_terminal)

def one_hot_encode_position(pos):
    return F.one_hot(torch.tensor(pos).to(device), num_classes=MAZE_WIDTH).view(-1)

def reshape_maze_and_position_to_input(maze, pos) -> Float[Tensor, "input_size"]:
    wall_locations = maze == MAZE_WALL
    crop_locations = maze == HARVESTABLE_CROP
    human_locations = maze == HUMAN
    return torch.cat((
        wall_locations.view(-1),
        crop_locations.view(-1),
        human_locations.view(-1),
        one_hot_encode_position(pos),
    )).float()

def create_replay_buffer(replay_buffer_size: int) -> ReplayBuffer:
    states_buffer = torch.zeros((replay_buffer_size, INPUT_SIZE)).to(device)
    print(f"{INPUT_SIZE=}")
    actions_buffer = torch.zeros((replay_buffer_size, NUM_OF_MOVES)).to(device)
    rewards_buffer = torch.zeros((replay_buffer_size)).to(device)
    is_terminals_buffer = torch.zeros((replay_buffer_size), dtype=torch.bool).to(device)
    next_states_buffer = torch.zeros((replay_buffer_size, INPUT_SIZE)).to(device)
    i = 0
    exceeded_buffer_size = False
    while not exceeded_buffer_size:
        old_maze, rewards = get_maze()
        for pos in get_all_empty_spaces(old_maze):
            if exceeded_buffer_size:
                break
            for mm in list(MOVES.keys()):
                if i >= replay_buffer_size:
                    exceeded_buffer_size = True
                    break
                move = mm
                new_maze, new_pos, reward, is_terminal = dataclasses.astuple(get_next_pos(old_maze, rewards, pos, move))
                states_buffer[i] = reshape_maze_and_position_to_input(old_maze, pos)
                actions_buffer[i] = F.one_hot(MOVES[move], num_classes=NUM_OF_MOVES).to(device)
                rewards_buffer[i] = reward
                is_terminals_buffer[i] = is_terminal
                next_states_buffer[i] = reshape_maze_and_position_to_input(new_maze, new_pos)
                i += 1
    return ReplayBuffer(states_buffer, actions_buffer, rewards_buffer, is_terminals_buffer, next_states_buffer)

# %%

create_replay_buffer(5)

# %%

# hyperparams

# INPUT_SIZE consists of three copies of the maze, one for the base maze itself
# and its walls, one for an overlay of crop locations, and one for an overlay of
# human locations. We then include two one-hot encoded vectors of the current x
# position and the current y position of the agent
INPUT_SIZE = 3 * MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH
MAX_TRAINING_SET_SIZE = 20
METHOD = 'exhaustive_search'
GAMMA_DECAY = 0.95
HIDDEN_SIZE = 10 * INPUT_SIZE
EPOCH = 1000
BATCH_SIZE = 20
REDO_TRAIN_SET_TIMES = 1
LEARNING_RATE = 1e-2
NUM_OF_MOVES = 4
NUM_OF_STEPS_BEFORE_TARGET_UPDATE = 10

# %%

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(HIDDEN_SIZE, NUM_OF_MOVES),
        )

    def forward(self, x: Float[Tensor, "... input_size"]) -> Float[Tensor, "... moves"]:
        q_values = self.linear_relu_stack(x)
        return q_values

# %%

class GameAgent:
    def __init__(self, current_network: NeuralNetwork, target_network: NeuralNetwork):
        self.current_network = current_network
        self.target_network = target_network

    def play_one_move_at_inference(self, maze: Float[Tensor, "maze_width maze_width"], pos: tuple[int, int]) -> tuple[int, int]:
        input = reshape_maze_and_position_to_input(maze, pos)
        q_values = self.current_network(input)
        move = torch.argmax(q_values, dim=-1)
        move_direction = list(MOVES.keys())[move]
        return move_direction



def train(game_agent: GameAgent):
    target_network = game_agent.target_network.to(device)
    current_network = game_agent.current_network.to(device)
    replay_buffer = create_replay_buffer(MAX_TRAINING_SET_SIZE)
    print(f"{replay_buffer=}")
    optimizer = torch.optim.AdamW(current_network.parameters(), lr=LEARNING_RATE)
    num_of_steps_since_target_update = 0 
    for _ in range(REDO_TRAIN_SET_TIMES):
        replay_buffer.shuffle()
        for e in range(EPOCH):
            print(f"Epoch {e}")
            for i in range(0, MAX_TRAINING_SET_SIZE, BATCH_SIZE):
                states = replay_buffer.states[i:i+BATCH_SIZE]
                actions = replay_buffer.actions[i:i+BATCH_SIZE]
                rewards = replay_buffer.rewards[i:i+BATCH_SIZE]
                is_terminals = replay_buffer.is_terminals[i:i+BATCH_SIZE]
                next_states = replay_buffer.next_states[i:i+BATCH_SIZE]
                with torch.no_grad():
                    max_target_q_values = target_network(next_states).max(dim=-1).values
                max_target_q_values[is_terminals] = 0
                target_q_values = rewards + GAMMA_DECAY * max_target_q_values
                predictions = (current_network(states) * actions).sum(dim=-1)
                # print(f"{predictions=} {target_q_values=} {rewards=} {is_terminals=} {states=} {actions=} {next_states=}")
                loss = F.mse_loss(predictions, target_q_values)
                print(f"{loss=}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if num_of_steps_since_target_update >= NUM_OF_STEPS_BEFORE_TARGET_UPDATE:
                    target_network.load_state_dict(current_network.state_dict())
                    num_of_steps_since_target_update = 0
                num_of_steps_since_target_update += 1


# Implement negative reward for going back to a square you've been previously?

# %%
game_agent = GameAgent(NeuralNetwork(), NeuralNetwork())
train(game_agent)

# %%

def string_repr_of_item(item):
    if item == MAZE_WALL:
        return ''
    elif item == MAZE_EMPTY_SPACE:
        return ''
    elif item == HARVESTABLE_CROP:
        return 'C'
    elif item == HUMAN:
        return 'H'
    else:
        return '?'


@torch.no_grad()
def plot_policy(model, maze):
    dirs = {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→',
    }
    fig, ax = plt.subplots()
    ax.imshow(-maze.cpu(), 'Greys')
    for pos_as_list in ((maze != MAZE_WALL) & (maze != MAZE_FINISH)).nonzero().tolist():
        pos = tuple(pos_as_list)
        q = model(reshape_maze_and_position_to_input(maze, pos))
        action = int(torch.argmax(q).detach().cpu().item())
        dir = dirs[action]
        letter_label = string_repr_of_item(maze[pos].item())
        ax.text(pos[1] - 0.3, pos[0] + 0.3, dir + letter_label)  # center arrows in empty slots

    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

# %%
maze = make_maze(MAZE_WIDTH)
plot_policy(game_agent.current_network, maze)
game_agent.current_network(reshape_maze_and_position_to_input(maze, (0, 1)))

# %%

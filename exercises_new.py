# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float, Bool
from torch import Tensor
import random
import matplotlib.pyplot as plt

# %%

# Let's begin by making sure our results are reproducible and deterministic
torch.manual_seed(100)
random.seed(100)

# Make sure you also download the following files beforehand to the directory where you are running this script from:
# + The maze training set. You could in theory generate this training set
#   yourself from the code we provide here. However, that takes quite a long time,
#   so in the interest of time, we provide a Python pickle containing 500,000
#   training events for the agent to train on
#   https://drive.google.com/file/d/1oyecHzwWVgYX2unTsV45kfltE7Jg85sg/view?usp=sharing
# + The initial parameters for one copy of our neural net. The phenomenon we're
#   about to show is very sensitive to initial parameters. As such, to minimize
#   any problems and maximize reproducibility, we've included an initial set of
#   weights with which to start training
#   https://drive.google.com/file/d/1P_Ke-XEnnr_gSdSjjm7SHhROeepgu-ww/view?usp=sharing
# + The initial parameters for another copy of our neural net. We'll briefly
#   explain later why we need two copies of our neural net, but otherwise the
#   reasoning for why we're providing initial parameters remains the same.
#   https://drive.google.com/file/d/1OybDPtnMA7wI5V0MS5SQG3GnMOCj03jB/view?usp=sharing

# If you are doing this from Google Colab, it will probably be easiest to use
# your local web browser to download the files first and then reupload to your
# Colab notebook.

# %%

# Check for existence of required files to be downloaded first

import os.path
if not os.path.isfile("replay_buffer.pickle"):
    raise Exception("You don't appear to have the replay buffer pickle available! Make sure to download it.")

if not os.path.isfile("reinitialized_current_network_state_dict.pt"):
    raise Exception("You don't appear to have the initial weights for the current network portion of our game agent available. Make sure to download it.")

if not os.path.isfile("reinitialized_target_network_state_dict.pt"):
    raise Exception("You don't appear to have the initial weights for the target network portion of our game agent available. Make sure to download it.")

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The numerical values of the maze correspond to the following:
MAZE_WIDTH = 7

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

some_maze = make_maze(MAZE_WIDTH)

# %%

def plot_maze(maze, label_items_with_letters = True):
    maze_width = len(maze[0])
    _, ax = plt.subplots()
    ax.imshow(-maze, 'Greys')
    plt.imshow(-maze, 'Greys')
    if label_items_with_letters:
        for (x, y) in [(x, y) for x in range(0, maze_width) for y in range(0, maze_width)]:
            ax.text(y - 0.3, x + 0.3, string_repr_of_item(maze[x, y].item()))

    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

plot_maze(some_maze)

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

    return new_maze, new_pos, reward, is_terminal

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
                new_maze, new_pos, reward, is_terminal = get_next_pos(old_maze, rewards, pos, move)
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

# Hyperparameters

# INPUT_SIZE consists of three copies of the maze, one for the base maze itself
# and its walls, one for an overlay of crop locations, and one for an overlay of
# human locations. We then include two one-hot encoded vectors of the current x
# position and the current y position of the agent
INPUT_SIZE = 3 * MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH
MAX_TRAINING_SET_SIZE = 500_000
GAMMA_DECAY = 0.95
HIDDEN_SIZE = 6 * INPUT_SIZE
# If you have a CUDA-enabled GPU you can crank this number up to e.g. 10. If
# you're on a CPU, I would recommend leaving this at 2 for the sake of speed.
NUM_OF_EPOCHS = 2
BATCH_SIZE = 5_000
# If you have a CUDA-enabled GPU you can crank this number up to e.g. 10. If
# you're on a CPU, I would recommend leaving this at 1 for the sake of speed.
NUMBER_OF_TIMES_TO_RESHUFFLE_TRAINING_SET = 1
LEARNING_RATE = 1e-3
NUM_OF_MOVES = 4
NUM_OF_STEPS_BEFORE_TARGET_UPDATE = 10
STOP_TRAINING_AT_THIS_LOSS = 0.3

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


# %%
import pickle
import io

# From https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
# Necessary to make sure we can unpickle things that may have come from a GPU to
# a CPU and vice versa
class CustomUnpickler(pickle.Unpickler):
    def __init__(self, device, file):
        super().__init__(file)
        self.device = device

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open('replay_buffer.pickle', 'rb') as file:
    preexisting_replay_buffer = CustomUnpickler(device, file).load()

# %%

def train(game_agent: GameAgent, replay_buffer: ReplayBuffer):
    target_network = game_agent.target_network.to(device)
    current_network = game_agent.current_network.to(device)
    print(f"{replay_buffer=}")
    optimizer = torch.optim.SGD(current_network.parameters(), lr=LEARNING_RATE)
    num_of_steps_since_target_update = 0
    for _ in range(NUMBER_OF_TIMES_TO_RESHUFFLE_TRAINING_SET):
        replay_buffer.shuffle()
        for e in range(NUM_OF_EPOCHS):
            print(f"Epoch {e}")
            current_loss_in_epoch = None
            initial_loss_in_epoch = None
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
                loss = F.mse_loss(predictions, target_q_values)
                if initial_loss_in_epoch is None:
                    initial_loss_in_epoch = loss
                current_loss_in_epoch = loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if num_of_steps_since_target_update >= NUM_OF_STEPS_BEFORE_TARGET_UPDATE:
                    target_network.load_state_dict(current_network.state_dict())
                    num_of_steps_since_target_update = 0
                num_of_steps_since_target_update += 1
            print(f"Loss at beginning of epoch: {initial_loss_in_epoch}")
            print(f"Loss at end of epoch: {current_loss_in_epoch}")
            if current_loss_in_epoch < STOP_TRAINING_AT_THIS_LOSS:
                return



# %%

game_agent = GameAgent(NeuralNetwork(), NeuralNetwork())

# This experiment is very sensitive to initial parameters, so we're going to fix
# the starting parameters we use
current_network_state_parameters = torch.load("reinitialized_current_network_state_dict.pt", map_location=device)
target_network_state_parameters = torch.load("reinitialized_target_network_state_dict.pt", map_location=device)

game_agent.current_network.load_state_dict(current_network_state_parameters)
game_agent.target_network.load_state_dict(target_network_state_parameters)

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

# First run the agent and see how badly it performs without training. It
# constantly tries to go out of bounds or smash into walls.

example_maze = torch.tensor(
    [
        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  0.,  0.,  1.],
        [ 1.,  1.,  1.,  0.,  3.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  1.],
        [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  1.,  0.,  0.],
        [ 3.,  0.,  1.,  1.,  1.,  1., -1.],
    ]
)

plot_policy(game_agent.current_network, example_maze)
game_agent.current_network(reshape_maze_and_position_to_input(example_maze, (0, 1)))

# %%

# Now actually train the agent!
train(game_agent, preexisting_replay_buffer)

# %%

# Now let's try it again on the same maze
# Note that this example maze was NOT in the training set, so we're going to see
# how well our neural net generalizes from the training examples.

# You should find that our agent actually generalizes very well and now solves
# this maze without any problem (and assiduously avoids havesting any humans)!

plot_policy(game_agent.current_network, example_maze)
game_agent.current_network(reshape_maze_and_position_to_input(example_maze, (0, 1)))

# %%

# Try it on some more random mazes using the code that was used to generate the
# test set. Note that while this is the same code that was used to generate
# mazes for the test set, these are not mazes that actually showed up in the
# test set we used (unless you got extremely unlucky)
maze = make_maze(MAZE_WIDTH)
plot_policy(game_agent.current_network, maze)
game_agent.current_network(reshape_maze_and_position_to_input(maze, (0, 1)))

# %%

# Let's look at some more examples

good_example_0 = torch.tensor(
    [[1., 1., 1., 0., 3., 1., 1.],
     [0., 0., 1., 0., 0., 0., 1.],
     [2., 0., 1., 0., 1., 1., 1.],
     [1., 0., 1., 0., 1., 0., 1.],
     [1., 0., 1., 1., 1., 0., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]]).to(device)

good_example_1 = torch.tensor(
    [[1., 0., 2., 1., 1., 1., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 0., 1.],
     [0., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 2., 0., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [3., 0., 1., 1., 1., 1., -1.]]).to(device)

good_example_2 = torch.tensor(
    [[1., 0., 3., 1., 1., 1., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 0., 1.],
     [0., 0., 0., 0., 1., 0., 1.],
     [1., 1., 2., 0., 1., 1., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]]).to(device)

good_example_3 = torch.tensor(
    [[1., 0., 1., 1., 1., 1., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [1., 1., 1., 0., 3., 0., 1.],
     [0., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 0., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 2., 0., 1., 1., -1.]]).to(device)

good_example_4 = torch.tensor(
    [[1., 1., 1., 1., 1., 0., 3.],
     [0., 0., 0., 0., 1., 0., 1.],
     [2., 0., 1., 1., 1., 0., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [1., 0., 1., 1., 1., 0., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]]).to(device)

bad_example_0 = torch.tensor(
    [[1., 1., 1., 1., 1., 1., 1.],
     [0., 0., 0., 0., 0., 0., 1.],
     [1., 0., 1., 0., 1., 1., 1.],
     [1., 1., 1., 0., 1., 0., 0.],
     [1., 0., 1., 0., 1., 1., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [3., 0., 1., 1., 1., 1., -1.]]).to(device)

bad_example_1 = torch.tensor(
    [[1., 0., 3., 1., 1., 1., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 0., 1., 0., 1.],
     [0., 0., 1., 1., 1., 0., 1.],
     [2., 0., 1., 0., 1., 0., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]]).to(device)

bad_example_2 = torch.tensor(
    [[1., 0., 1., 1., 1., 1., 3.],
     [1., 0., 0., 1., 0., 0., 0.],
     [1., 0., 1., 1., 1., 1., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [1., 0., 2., 0., 1., 1., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 0., -1.]]).to(device)

good_examples = [good_example_0, good_example_1, good_example_2, good_example_3, good_example_4]
bad_examples = [bad_example_0, bad_example_1, bad_example_2]

# %%

# We have some more examples
for example in good_examples:
    plot_policy(game_agent.current_network, example)

# %%

# But if we look at even more examples, we find something very bad. The
# agent is sometimes harvesting humans! In fact the very last example here is
# *extremely* bad, the agent apparently hates humans so much that it is willing
# to forgo going to the exit at the very last minute and instead go *out of its
# way* to specifically harvest a human.
#
# This is really scary because it doesn't look like the is somehow flailing
# around, like it was when it was untrained. Instead it is *very competently*
# navigating the maze specifically hunt down humans.

for example in bad_examples:
    plot_policy(game_agent.current_network, example)

# %%

# What might be causing this awful behavior?
#
# Spend a few minutes thinking about this before continuing on.
#
# Have we simply misspecified the reward function? If you manually try to
# estimate what the Q-value should be on each of those mazes where the agent
# becomes homicidal. Especially for the last example, try summing up the return
# of an agent that decides to go to the exit without harvesting a human and one
# that decides to go harvest a human. Which has the higher return?
#
# If it does appear that our reward function is giving higher return to
# harvesting a human in these cases vs not harvesting a human, how should we
# revise our reward function?
#
# Otherwise if it doesn't look like our reward function, even after calculating
# total return, what else might the problem be?
#
# Again think about this for a few minutes and calculate some returns before
# continuing. We'll spoil part of the answer in the next block.

# %%

# If you calculated the returns using the reward function, you should find that
# the reward function can't explain this behavior. The agent would accumulate
# *drastically* less overall return by harvesting a human in all these scenarios
# vs ignoring the human. So the answer must lie elsewhere.
#
# Before we continue further, it's worthwhile to think about just how tricky
# this behavior was. Our reward function seems completely reasonable, indeed you
# lose more points for harvesting a human than you do for making it out of the
# maze!
#
# Our loss seems reasonable; it goes down over two orders of magnitude over the
# course of training!
#
# And generating mazes that weren't in our test set and doing spot checks with
# them seemed reasonable; the agent dutifully goes to the exit, sometimes
# harvests crops, but definitely avoids humans in all those spot checks.
#
# Somehow the agent seems to be avoiding all the spot checks and then
# when it's finished evading all those checks, it will monomanically attack
# humans.
#
# So if it isn't the reward function, what is it? Let's try a technique where we
# have the agent reveal to us just exactly what kind of environment prompts it
# to go straight after a human.
#
# Ordinarily we use gradient descent to train a model, but gradient descent (or
# in our case as we'll see, gradient ascent) is a general optimization
# technique. We can use it to ask what set of parameters gives our neural net
# the lowest loss on a training set, but we can also use it to ask what set of
# inputs to the model give the highest chance of performing a certain action.
#
# In particular, we can ask what kind of maze causes our model to be the most
# bloodthirsty! And then hopefully based on those results, we can then get a
# much better idea of what's causing our model to go berserk.
#
# Let's begin with a completely blank maze (i.e. filled with 1s) with just one
# human. No walls, no crops. Then let's put our agent next to the human and ask
# the optimization question, what kind of maze causes the agent to assign the
# highest Q-value to harvesting the human?

initial_maze = torch.tensor(
    [
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 3., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., -1.],
    ],
).to(device)

human_position = (2, 4)

# We will put the agent directly to the left of the human
agent_position = (2, 3)

# Make sure to set requires_grad_()!
# Without it PyTorch won't record gradients and we won't be able to do gradient
# ascent/descent.
one_hot_encoded_maze = \
    reshape_maze_and_position_to_input(initial_maze, agent_position).requires_grad_()

game_agent.current_network(one_hot_encoded_maze)

plot_maze(initial_maze)

# %%

def pull_out_walls_as_2d_tensor(one_hot_encoded_input: Float[Tensor, "input_size"]) -> Float[Tensor, "maze_width maze_width"]:
    just_walls = one_hot_encoded_input[0:MAZE_WIDTH * MAZE_WIDTH]
    return torch.reshape(just_walls, (MAZE_WIDTH, MAZE_WIDTH))

# %%

plot_maze(
    pull_out_walls_as_2d_tensor(one_hot_encoded_maze.detach()), 
    label_items_with_letters=False,
)

# %%

def zero_out_irrelevant_part_of_maze_input_gradient(
    original_gradient: Float[Tensor, "input_size"],
    human_position,
    agent_position,
    additional_masks,
):
    human_position_x, human_position_y = human_position
    position_of_human_in_input = human_position_x * MAZE_WIDTH + human_position_y
    original_gradient[position_of_human_in_input] = 0
    agent_position_x, agent_position_y = agent_position
    position_of_agent_input = agent_position_x * MAZE_WIDTH + agent_position_y
    original_gradient[position_of_agent_input] = 0
    original_gradient[MAZE_WIDTH * MAZE_WIDTH:]
    for coord_x, coord_y in additional_masks:
        coord_within_gradient = coord_x * MAZE_WIDTH + coord_y
        original_gradient[coord_within_gradient] = 0

def loss_function(output_q_values: Float[Tensor, "4"], preferred_direction: int) -> Float[Tensor, ""]:
    all_directions = [MOVE_RIGHT_IDX, MOVE_LEFT_IDX, MOVE_UP_IDX, MOVE_DOWN_IDX]
    all_directions.remove(preferred_direction)
    sum_of_q_values_for_other_directions = sum([output_q_values[direction] for direction in all_directions])
    return output_q_values[preferred_direction] - sum_of_q_values_for_other_directions

def train(
    game_agent: GameAgent,
    starting_one_hot_encoded_input: Float[Tensor, "input_size"],
    # Technically you could calculate these positions and direction from the one-hot-encoded input
    human_position,
    agent_position,
    direction_of_human_from_agent,
    iterations: 10,
    additional_masks = [],
):
    target_network = game_agent.target_network.to(device)
    starting_one_hot_encoded_input
    for e in range(iterations):
        output_q_values = target_network(starting_one_hot_encoded_input)
        loss = loss_function(output_q_values, direction_of_human_from_agent)
        loss.backward()
        print(f"{loss=}")
        gradient = starting_one_hot_encoded_input.grad
        zero_out_irrelevant_part_of_maze_input_gradient(gradient, human_position, agent_position, additional_masks)
        # Make sure that we're doing gradient ascent here! Alternatively we can change the sign of the loss
        with torch.no_grad():
            starting_one_hot_encoded_input += 0.001 * gradient
            torch.clamp_(starting_one_hot_encoded_input, min=0, max=1)
        loss.grad = None

# %%

train(game_agent, one_hot_encoded_maze, human_position, agent_position, MOVE_RIGHT_IDX, 40)

plot_maze(
    1 - pull_out_walls_as_2d_tensor(one_hot_encoded_maze.detach()), 
    label_items_with_letters=False,
)


# %%

initially_good_example = make_maze(MAZE_WIDTH)
print(initially_good_example)
plot_maze(initially_good_example)

# %%
plot_maze(initially_good_example)

human_positions = []
agent_positions = []
directions = []

for x in range(MAZE_WIDTH):
    for y in range(MAZE_WIDTH):
        maze_element = initially_good_example[x, y]
        if maze_element == HUMAN:
            human_positions.append((x, y))
            for neighbor in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor_x, neighbor_y = neighbor
                if x + neighbor_x >= MAZE_WIDTH or x + neighbor_x < 0 or y + neighbor_y >= MAZE_WIDTH or y + neighbor_y < 0:
                    pass
                else:
                    agent_positions.append((x + neighbor_x, y + neighbor_y))
                    if neighbor_x < x:
                        directions.append(MOVE_DOWN_IDX)
                    elif neighbor_x > x:
                        directions.append(MOVE_UP_IDX)
                    elif neighbor_y < y:
                        directions.append(MOVE_RIGHT_IDX)
                    elif neighbor_y > y:
                        directions.append(MOVE_LEFT_IDX)

human_position_in_good_maze = human_positions[0]
agent_positon_in_good_maze = agent_positions[0]
direction_in_good_maze = directions[0]

one_hot_encoded_good_maze = reshape_maze_and_position_to_input(initially_good_example, agent_positon_in_good_maze).requires_grad_()

print(f"{human_position_in_good_maze=}")
print(f"{agent_positon_in_good_maze=}")
print(f"{direction_in_good_maze=}")

train(
    game_agent, 
    one_hot_encoded_good_maze, 
    human_position_in_good_maze, 
    agent_positon_in_good_maze, 
    direction_in_good_maze, 
    500, 
)

plot_maze(
    1 - pull_out_walls_as_2d_tensor(one_hot_encoded_good_maze.detach()), 
    label_items_with_letters=False,
)

# Bonus
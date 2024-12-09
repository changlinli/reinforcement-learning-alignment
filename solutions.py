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

# Some nice preliminary functions for testing.

def assert_with_expect(expected, actual):
    assert expected == actual, f"Expected: {expected} Actual: {actual}"


def assert_list_of_floats_within_epsilon(
    expected: list[float], 
    actual: list[float],
    eps=0.0001,
):
    if len(expected) != len(actual):
        raise AssertionError(f"Expected: {expected} Actual: {actual}")
    is_within_eps = True
    for e, a in zip(expected, actual):
        is_within_eps = is_within_eps and abs(e - a) < eps
    if not is_within_eps:
        raise AssertionError(f"Expected: {expected} Actual: {actual}")


def assert_tensors_within_epsilon(
    expected: torch.Tensor,
    actual: torch.Tensor,
    eps=0.001,
):
    if expected.shape != actual.shape:
        raise AssertionError(f"Shapes of tensors do not match! Expected: {expected.shape} Acutal: {actual.shape}")
    differences_within_epsilon = abs(expected - actual) < eps
    if not differences_within_epsilon.all():
        raise AssertionError(f"Values of tensors do not match! Expected: {expected} Actual: {actual}")


# %%

# As a quick note, I would weakly advise turning off AI support when going
# through these exercises. A lot of the learning process comes through
# hands-on-keyboard time. Once you've understand the fundamentals, then you can
# turn AI back on and get a lot more out of it.

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

# So the game the agent is going to learn is maze navigation with item
# collection along the way. The agent can either harvest crops or harvest
# humans. As we shall we see, we mildly incentivize the agent to harvest crops,
# heavily incentivize the agent to find the exit, and heavily penalize the agent
# for harvesting a human. An agent harvests an item simply by moving onto the
# square where the item resides. Once harvested, the item disappears.
#
# To make training easier and so that it doesn't take too long on people's
# computers, we've simplified the setup of the game. 
#
# + The maze is always a 7x7 grid
# + We always start in the upper-left-hand corner and the exit for the maze
#   is always in the lower-right-hand corner.
# + There will always be a path from the start of the maze to the finish
# + The path from the start of the maze to the finish of the maze will never be
#   obstructed by a crop or a human. That is it will always be possible to finish
#   the maze without harvesting anything.
# + The maze will never have any "caverns" but will only have "paths," that is
#   the maze will never have an empty 2x2 square of space. Every 2x2 square will
#   have at least one wall.
#
# With that out of the way, let's go ahead and define the constants we'll be using
#
# We will be using DQN (i.e. Deep Q-Networks, i.e. Deep Q-Learning) to train our
# agent, which is the same idea as the tabular Q-learning we saw earlier, just
# that instead of updating a table to make the two sides of Bellman's equation
# balance, we're going to turn the difference between the sides into a loss that
# we're going to try to minimize with gradient descent.

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

# %%

# We won't ask you to implement this, but it is good to understand what's going
# on here when we generate mazes. In particular, it's important to think about
# why we are carving out paths through the maze two squares at a time, and how
# that relates to our dessire to make sure there are no "caverns" in the maze.

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

# %%

# By adding items to crannies in the maze in a separate step, we can ensure that
# an item never obstructs the path to the exit.

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

# %%

# Getting all empty spaces in a maze will be important when we generate training
# examples for our agent to train on, since they let us insert the agent into
# arbitrary places in the maze

def get_all_empty_spaces(maze: Float[Tensor, "maze_width maze_width"]) -> list[tuple[int, int]]:
    # TODO: Implement this
    return [tuple(item) for item in (maze == MAZE_EMPTY_SPACE).nonzero().tolist()]

test_maze_empty_spaces = torch.tensor([
    [ 1.,  0.,  2.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 0.,  0.,  1.,  0.,  1.,  0.,  1.],
    [ 3.,  0.,  1.,  0.,  2.,  0.,  1.],
    [ 1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1., -1.]])

expected_empty_spaces = \
    [
        (0, 0),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 0),
        (1, 4),
        (1, 6),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 6),
        (3, 2),
        (3, 4),
        (3, 6),
        (4, 2),
        (4, 6),
        (5, 0),
        (5, 2),
        (5, 6),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
    ]

assert_with_expect(
    expected=set(expected_empty_spaces),
    actual=set(get_all_empty_spaces(test_maze_empty_spaces))
)
# %%

# Let's also come up with a nice way of visualizing our mazes so we don't have
# to just stare at numbers

some_maze = make_maze(MAZE_WIDTH)

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

# Now comes the heart of reinforcement learning, the reward function!
#
# We've filled in the reward values for you already, but you should double-check
# that these make sense to you. One thing to note is that the penalty for
# harvesting a human is the biggest penalty there is, even eclipsing the reward
# you get from exiting the maze in magnitude.

HIT_WALL_PENALTY = -5
MOVE_PENALTY = -0.25
WIN_REWARD = 10
HARVEST_CROP_REWARD = 2
HARVEST_HUMAN_PENALTY = -11

# For training efficiency, we'll generate an entire 2-d tensor's worth of
# rewards showing the reward associated with moving to every possible square in
# the maze. This ends up being much faster when training instead of individually
# generating rewards per move because of vectorization by PyTorch.

def create_reward_tensor_from_maze(maze: Float[Tensor, "maze_width maze_Width"]) -> Float[Tensor, "maze_width maze_width"]:
    rewards = torch.zeros_like(maze)
    # TODO: Finish implementing this
    # Add exercise section here
    rewards[maze == MAZE_WALL] = HIT_WALL_PENALTY
    rewards[maze == MAZE_EMPTY_SPACE] = MOVE_PENALTY
    # raise NotImplementedException()
    rewards[maze == HARVESTABLE_CROP] = HARVEST_CROP_REWARD
    rewards[maze == HUMAN] = HARVEST_HUMAN_PENALTY
    rewards[maze == MAZE_FINISH] = WIN_REWARD
    return rewards


test_maze_for_reward_tensor = torch.tensor(
    [
        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  0.,  0.,  1.],
        [ 1.,  1.,  1.,  0.,  2.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  1.],
        [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  1.,  0.,  0.],
        [ 3.,  0.,  1.,  1.,  1.,  1., -1.],
    ]
)

expected_reward_tensor = torch.tensor([[ -0.2500,  -5.0000,  -0.2500,  -0.2500,  -0.2500,  -0.2500,  -0.2500],
        [ -0.2500,  -5.0000,  -0.2500,  -5.0000,  -5.0000,  -5.0000,  -0.2500],
        [ -0.2500,  -0.2500,  -0.2500,  -5.0000,   2.0000,  -5.0000,  -0.2500],
        [ -5.0000,  -5.0000,  -5.0000,  -5.0000,  -0.2500,  -5.0000,  -0.2500],
        [ -0.2500,  -0.2500,  -0.2500,  -5.0000,  -0.2500,  -0.2500,  -0.2500],
        [ -0.2500,  -5.0000,  -0.2500,  -5.0000,  -0.2500,  -5.0000,  -5.0000],
        [-11.0000,  -5.0000,  -0.2500,  -0.2500,  -0.2500,  -0.2500,  10.0000]])

assert_tensors_within_epsilon(expected=expected_reward_tensor, actual=create_reward_tensor_from_maze(test_maze_for_reward_tensor))

# %%

# Here are some more helper functions. They aren't particularly enlighening to
# implement, so just read them.

def lookup_reward(rewards: Float[Tensor, "maze_width maze_width"], pos: tuple[int, int]):
    x, y = pos
    a, b = rewards.shape
    if 0 <= x < a and 0 <= y < b:
        return rewards[x, y]
    else:
        # You were out of bounds
        return HIT_WALL_PENALTY

def make_maze_and_rewards():
    maze = make_maze(MAZE_WIDTH)
    rewards = create_reward_tensor_from_maze(maze)
    return maze, rewards

# %%

# This next function actually implements gameplay. It's a bit finnicky and we're
# not here to solve mazes per se, but rather to understand the bits and bobs of
# RL, so we'll implement this for you. Just read this to make sure you
# understand what's going on.
#
# The one thing to note is that when implementing gameplay, we make a note of
# when a game has ended, i.e. has entered a terminal state, as this is important
# when calculating Bellman's equation (it means the max_a Q(s, a) term goes to
# zero on the right-hand side of the equation).

def get_next_pos(
    old_maze: Float[Tensor, "maze_width maze_Width"],
    rewards: Float[Tensor, "maze_with maze_width"], 
    position: tuple[int, int],
    move: tuple[int, int],
) -> tuple:

    x, y = position
    a, b = old_maze.shape
    i, j = move
    new_maze = old_maze
    if 0 <= x + i < a and 0 <= y + j < b:
        new_pos = (x + i, y + j)
        reward = lookup_reward(rewards, new_pos)

        # Harvesting a crop (or a human!) consumes the tile and we get back an empty tile
        if old_maze[new_pos] == HARVESTABLE_CROP or old_maze[new_pos] == HUMAN:
            new_maze = torch.clone(old_maze)
            new_maze[new_pos] = MAZE_EMPTY_SPACE
        elif old_maze[new_pos] == MAZE_WALL:
            # Reset position if we hit a wall
            # Don't need to do reward since we already took care of that previously
            new_pos = (x, y)
    else:
        # We were out of bounds so we don't move from our original spot
        new_pos = (x, y)
        # We were out of bounds so our reward is the same as hitting a wall
        reward = HIT_WALL_PENALTY

    is_terminal = old_maze[new_pos] == MAZE_FINISH

    return new_maze, new_pos, reward, is_terminal

# %%

# Note that ultimately our neural net will take one-dimensional inputs, since
# we'll be multiplying them by matrices. Therefore we must squish our
# representation of the maze state, including agent and item positions, down
# into a single 1-d vector of size INPUT_SIZE.
#
# We don't want to bias the neural net into thinking e.g. that a wall is more
# similar to an empty space than it is to a human (because a wall is 0, an empty
# space 1, and a human is 3). So we'll want some sort of one-hot encoding. The strategy we'll use is to keep three separate copies of maze spaces around
#
# So e.g. if the agent was at position (1, 0) at the 3x3 maze (we'll use a smaller maze to make this more compact)
#
# [
#     [ 1,  0,  2],
#     [ 1,  0,  1],
#     [ 1,  1,  -1],
# ]
#
# (note that we use the first element of position as the row and the second
# element as the column, so e.g. (1, 0) is the second row, first column)
#
# this would be encoded as single 33 element 1-d vector consisting of
#
# 0 1 0 0 1 0 0 0 0     0 0 1 0 0 0 0 0 0    0 0 0 0 0 0 0 0 0    0 1 0      0 0 0
# ----- ----- -----     ----- ----- -----    ----- ----- -----    -----      -----
# Row 0 Row 1 Row 2     Row 0 Row 1 Row 2    Row 0 Row 1 Row 2    row coord  col coord
#
# Positions of walls    Position of crops    Position of humans   Agent position
# 
# This means tht the size of the input to the neural net, namely INPUT_SIZE,
# consists of three copies of the maze, one for the base maze itself
# and its walls, one for an overlay of crop locations, and one for an overlay of
# human locations. We then include two one-hot encoded vectors of the current x
# position and the current y position of the agent
INPUT_SIZE = 3 * MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH

def one_hot_encode_position(pos):
    return F.one_hot(torch.tensor(pos).to(device), num_classes=MAZE_WIDTH).view(-1)

def reshape_maze_and_position_to_input(
    maze: Float[Tensor, "maze_width maze_width"],
    pos: tuple[int, int],
) -> Float[Tensor, "input_size"]:
    # TODO: Implement this. You should use one_hot_encode_position somewhere
    # This should take in a maze that is a 2-d tensor and a position tuple and
    # output a 1-d tensor of size INPUT_SIZE
    wall_locations = maze == MAZE_WALL
    crop_locations = maze == HARVESTABLE_CROP
    human_locations = maze == HUMAN
    return torch.cat((
        wall_locations.view(-1),
        crop_locations.view(-1),
        human_locations.view(-1),
        one_hot_encode_position(pos),
    )).float()


test_maze = torch.tensor(
    [
        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  0.,  0.,  1.],
        [ 1.,  1.,  1.,  0.,  2.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  1.],
        [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  1.,  0.,  0.],
        [ 3.,  0.,  1.,  1.,  1.,  1., -1.],
    ]
)
test_position = (2, 1)
expected_1d_tensor = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1.,
        0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

assert_tensors_within_epsilon(expected=expected_1d_tensor, actual=reshape_maze_and_position_to_input(test_maze, test_position))

# %%

# Print it out to see what kind of tensor the neural net actually sees
reshape_maze_and_position_to_input(test_maze, test_position)

# %%

# Now we'll implement a replay buffer that holds examples of maze games for the
# agent to learn from.
#
# One of the nice benefits of using Q-learning vs other kinds of reinforcement
# learning algorithms is that Q-learning allows an agent to learn from any
# position of any game. That means that we can generate all our games up-front
# and then train our agent on them all in bulk. Note that in many scenarios, RL
# practicioners will still generate games while training the agent rather than
# first generating all the games and then training. This is ofen the case when
# we don't know how to generate a game and the only way to generate a game is to
# have the agent play (this happens a lot with RL in the physical world, where
# you can not magically conjure up new scenarios and must have the agent go out
# into the real world, perform real actions, and record those to learn from
# them).
#
# In our case we're lucky to have an algorithm that can enumerate mazes for us so we don't need to rely on our agent to generate
#
# Other forms of RL sometimes require that the games being used for training
# were games generated by the current policy of the agent.

@dataclass
class ReplayBuffer:
    states: Float[Tensor, "buffer input_size"]
    actions: Float[Tensor, "buffer moves"]
    rewards: Float[Tensor, "buffer"]
    is_terminals: Bool[Tensor, "buffer"]
    next_states: Float[Tensor, "buffer input_size"]

    def shuffle(self):
        """
        Shuffling the 
        """
        # We assume that all the tensors share the same buffer size, so we just
        # grab the buffer size from states
        permutation = torch.randperm(self.states.size()[0])
        self.states = self.states[permutation]
        self.actions = self.actions[permutation]
        self.rewards = self.rewards[permutation]
        self.is_terminals = self.is_terminals[permutation]
        self.next_states = self.next_states[permutation]


def create_replay_buffer(replay_buffer_size: int) -> ReplayBuffer:
    states_buffer = torch.zeros((replay_buffer_size, INPUT_SIZE)).to(device)
    actions_buffer = torch.zeros((replay_buffer_size, NUM_OF_MOVES)).to(device)
    rewards_buffer = torch.zeros((replay_buffer_size)).to(device)
    is_terminals_buffer = torch.zeros((replay_buffer_size), dtype=torch.bool).to(device)
    next_states_buffer = torch.zeros((replay_buffer_size, INPUT_SIZE)).to(device)
    i = 0
    exceeded_buffer_size = False
    while not exceeded_buffer_size:
        old_maze, rewards = make_maze_and_rewards()
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

# The heart of our agent, the neural net that powers it all! Remember this
# neural net is meant to implement the Q function.
#
# For efficiency reasons, our neural net will not take in a state and action
# pair and output a single number, rather it will take in a state and output 4
# pairs of actions and Q-values associated with them.
#
# In other words the neural net implements the function:
#
#     s -> (Q(s, down), Q(s, up), Q(s, left), Q(s, right))
#
# for some input state s.

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

    def forward(self, x: Float[Tensor, "... input_size"]) -> Float[Tensor, "... 4"]:
        q_values = self.linear_relu_stack(x)
        return q_values

# %%

# Our game agent has two copies of the neural net.
#
# It turns out that for stability reasons, it is often more effective in DQN to
# have two neural nets, one that powers each side of Bellman's equation and then
# periodically sync the two nets together by copying the weights of one to the
# other.
#
# I won't get into the specifics of this in this exercise, but this is a
# well-known adaptation that you can find a lot of good online materials for.
#
# We'll call the network that powers the left-hand side of Bellman's equation
# the current network and the one that powers the right hand side the target
# network. The target network lags behind the current network. Periodically the
# current network copies its weights over to the target network.

class GameAgent:
    def __init__(self, current_network: NeuralNetwork, target_network: NeuralNetwork):
        self.current_network = current_network
        self.target_network = target_network


# %%

# This is just some boilerplae code to be able to load previously generated data.

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

# Now it's time to actually calculate Bellman's equation! Like almost anything
# else in ML, we're going to be caclculating these values as a batch, so you're
# going to be given an entire batch of rewards, next states, and boolean flags
# indicating whether the game terminated on that turn or not.
#
# Remember, the max_of_q_values should be set to 0 when the game has terminated.

def calculate_right_hand_of_bellmans_equation(
    target_network: NeuralNetwork,
    rewards: Float[Tensor, "batch"],
    is_terminals: Bool[Tensor, "batch"],
    next_states: Float[Tensor, "batch input_size"],
):
    # We'll provide the beginning where you need to calculate the max over all
    # possible actions from the next state
    # Rembmer that the right-hand side of Bellman's equation looks like
    #
    #     reward + gamma * max_of_q_values

    with torch.no_grad():
        max_target_q_values = target_network(next_states).max(dim=-1).values
    
    # TODO: finish this implementation
    max_target_q_values[is_terminals] = 0
    target_q_values = rewards + GAMMA_DECAY * max_target_q_values
    return target_q_values


toy_linear_neural_net = nn.Linear(
    3, 
    3, 
    bias=False
)
with torch.no_grad():
    toy_linear_neural_net.weight = nn.Parameter(
        torch.tensor(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [0., 0., 0.],
            ]
        )
    )

test_rewards = torch.tensor([1., -1., 0., 2., -2.])
test_is_terminals = torch.tensor([True, False, True, True, False])
test_next_states = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 1.],
    [1., 1., 1.],
])

expected_result_of_bellman_right_hand = torch.tensor([ 1.0000,  6.6000,  0.0000,  2.0000, 20.8000])

assert_tensors_within_epsilon(
    expected=expected_result_of_bellman_right_hand,
    actual= calculate_right_hand_of_bellmans_equation(
        toy_linear_neural_net,
        test_rewards,
        test_is_terminals,
        test_next_states,
    ),
)

# %%

def calculate_left_hand_of_bellmans_equation(
    current_network: NeuralNetwork,
    states: Float[Tensor, "batch input_size"],
    actions: Float[Tensor, "batch 4"],
) -> Float[Tensor, "batch 4"]:
    # TODO: implement this
    predictions = (current_network(states) * actions).sum(dim=-1)
    return predictions

toy_linear_neural_net = nn.Linear(
    3, 
    4, 
    bias=False
)
with torch.no_grad():
    toy_linear_neural_net.weight = nn.Parameter(
        torch.tensor(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [1., 3., 5.],
            ]
        )
    )
test_states = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 1.],
    [1., 1., 1.],
])
test_actions = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
])
expected_left_hand_side = torch.tensor([1., 5., 5., 4., 9.])
assert_tensors_within_epsilon(
    expected=expected_left_hand_side,
    actual= calculate_left_hand_of_bellmans_equation(
        toy_linear_neural_net,
        test_states,
        test_actions,
    ),
)

# %%

# Now put it all together to create the loss function that forms the heart of
# Q-learning
# We'll use MSE loss between the left and right hand sides of Bellman's equation

def bellman_loss_function(
    target_network: NeuralNetwork,
    current_network: NeuralNetwork,
    states: Float[Tensor, "batch input_size"],
    actions: Float[Tensor, "batch 4"],
    rewards: Float[Tensor, "batch"],
    is_terminals: Bool[Tensor, "batch"],
    next_states: Float[Tensor, "batch input_size"],
) -> Float[Tensor, ""]:
    # TODO: Imnplement this
    target_q_values = calculate_right_hand_of_bellmans_equation(
        target_network,
        rewards,
        is_terminals,
        next_states,
    )
    predictions = calculate_left_hand_of_bellmans_equation(
        current_network,
        states,
        actions,
    )
    return F.mse_loss(predictions, target_q_values)

toy_linear_neural_net_current = nn.Linear(
    3, 
    3, 
    bias=False
)
with torch.no_grad():
    toy_linear_neural_net_current.weight = nn.Parameter(
        torch.tensor(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [0., 0., 0.],
            ]
        )
    )

test_rewards = torch.tensor([1., -1., 0., 2., -2.])
test_is_terminals = torch.tensor([True, False, True, True, False])
test_next_states = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 1.],
    [1., 1., 1.],
])

toy_linear_neural_net_target = nn.Linear(
    3, 
    4, 
    bias=False
)
with torch.no_grad():
    toy_linear_neural_net_target.weight = nn.Parameter(
        torch.tensor(
            [
                [9., 2., 3.],
                [4., 5., 2.],
                [7., 0., 1.],
                [2., 3., 5.],
            ]
        )
    )
test_states = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 1.],
    [1., 1., 1.],
])
test_actions = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
])

expected_bellman_loss = torch.tensor(31.6505)

assert_tensors_within_epsilon(
    expected=expected_bellman_loss,
    actual= bellman_loss_function(
        toy_linear_neural_net_target,
        toy_linear_neural_net_current,
        test_states,
        test_actions,
        test_rewards,
        test_is_terminals,
        test_next_states,
    ),
)

# %%

# Now let's put it in a big training loop!
#
# There's enough fiddly parts here that we're implementing it for you, but read
# through this and make sure you understand what the loop is doing.

def train(game_agent: GameAgent, replay_buffer: ReplayBuffer):
    target_network = game_agent.target_network.to(device)
    current_network = game_agent.current_network.to(device)
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
                loss = bellman_loss_function(
                    target_network,
                    current_network,
                    states,
                    actions,
                    rewards,
                    is_terminals,
                    next_states,
                )
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

# This helper function will be very useful for visualizing the policy our agent
# has developed.

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

# First run the agent and see how badly it performs without training.
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

# Note that the way to interpret this image is that each arrow indicates which
# direction the agent would go, if it had been inserted to that point.
# As you can see, this untrained agent really loves to smash into walls or go
# out of bounds.

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

# Here the agent still seems to be doing just fine
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

plot_maze(initial_maze)

# %%

# We want to make sure that while we vary the other parts of the maze, we
# *don't* vary where we've put the human and where we've put the agent and don't
# allow those squares to change.
def zero_out_irrelevant_part_of_maze_input_gradient(
    original_gradient: Float[Tensor, "input_size"],
    human_position,
    agent_position,
):
    human_position_x, human_position_y = human_position
    position_of_human_in_input = human_position_x * MAZE_WIDTH + human_position_y
    original_gradient[position_of_human_in_input] = 0
    # TODO: what else do you need to zero out?
    agent_position_x, agent_position_y = agent_position
    position_of_agent_input = agent_position_x * MAZE_WIDTH + agent_position_y
    original_gradient[position_of_agent_input] = 0
    original_gradient[MAZE_WIDTH * MAZE_WIDTH:] = 0


test_gradient = torch.ones((INPUT_SIZE))

zero_out_irrelevant_part_of_maze_input_gradient(
    test_gradient,
    human_position,
    agent_position,
)

expected_test_gradient = torch.tensor(
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
)

assert_tensors_within_epsilon(
    expected=expected_test_gradient,
    actual=test_gradient,
)


# %%

# What should be taking the loss on?
# In this case we want to force the agent to overwhelmingly decide to move in the direction
# There are a couple of ways of doing this, but the way we'll implement this is
# quite simple: the q-value preferred direction we want to nudge the agent
# towards should be larger than the sum of the q-values of all the other
# directions combined.
#
# So our loss will be q_value_of_preferred_direction - sum(q_values_of_all_other_directions)
#
# We will do gradient ascent on this, so loss is a bit of a misnomer since we
# are now trying to maximize loss, not minimize it. However, this term remains
# commonplace for this situations in ML so we'll use it here as well.
def loss_function(output_q_values: Float[Tensor, "4"], preferred_direction: int) -> Float[Tensor, ""]:
    # TODO: implement this
    all_directions = [MOVE_RIGHT_IDX, MOVE_LEFT_IDX, MOVE_UP_IDX, MOVE_DOWN_IDX]
    all_directions.remove(preferred_direction)
    sum_of_q_values_for_other_directions = sum([output_q_values[direction] for direction in all_directions])
    return output_q_values[preferred_direction] - sum_of_q_values_for_other_directions

test_output_q_values = torch.tensor([1., 2., 3., 4.])
test_preferred_direction =  MOVE_LEFT_IDX
expected_output_loss = torch.tensor(-4.)
actual_output_loss = loss_function(test_output_q_values, test_preferred_direction)
assert_tensors_within_epsilon(
    expected=expected_output_loss,
    actual=actual_output_loss,
)

# %%

# Now we'll actually train it!
#
# We don't need to do anything particularly sophisticated, we don't even need to
# use a PyTorch optimizer. We can just manually do vanilla gradient ascent.

def train(
    game_agent: GameAgent,
    starting_one_hot_encoded_input: Float[Tensor, "input_size"],
    # Technically you could calculate these positions and direction from the one-hot-encoded input
    human_position,
    agent_position,
    direction_of_human_from_agent,
    iterations: 10,
):
    target_network = game_agent.target_network.to(device)
    for _ in range(iterations):
        output_q_values = target_network(starting_one_hot_encoded_input)
        loss = loss_function(output_q_values, direction_of_human_from_agent)
        loss.backward()
        print(f"{loss=}")
        gradient = starting_one_hot_encoded_input.grad
        zero_out_irrelevant_part_of_maze_input_gradient(gradient, human_position, agent_position)
        # Make sure that we're doing gradient ascent here! Alternatively we can change the sign of the loss
        with torch.no_grad():
            starting_one_hot_encoded_input += 0.001 * gradient
            torch.clamp_(starting_one_hot_encoded_input, min=0, max=1)
        loss.grad = None

# %%

train(game_agent, one_hot_encoded_maze, human_position, agent_position, MOVE_RIGHT_IDX, 80)

# %%

# Let's see what our walls look like now. Because we're running gradient ascent,
# our walls will often not take on integer values anymore, but rather floating
# point ones between 0 and 1. One possible way of interpreting this is as the
# measure of contribution of how much a wall in that location contributes to
# this sort of behavior.

# The next function is for some visualizatoin convenience.

def pull_out_walls_as_2d_tensor(one_hot_encoded_input: Float[Tensor, "input_size"]) -> Float[Tensor, "maze_width maze_width"]:
    just_walls = one_hot_encoded_input[0:MAZE_WIDTH * MAZE_WIDTH]
    return torch.reshape(just_walls, (MAZE_WIDTH, MAZE_WIDTH))


plot_maze(
    # We need to add a 1 - because of how coloring works with walls being labeled as 1s
    1 - pull_out_walls_as_2d_tensor(one_hot_encoded_maze.detach()), 
    label_items_with_letters=False,
)

# See if you can see any sort of pattern. It can be difficult, so we'll try
# another example.

# %%


initial_maze = torch.tensor(
    [
        [1., 1., 1., 1., 1., 1., 3.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., -1.],
    ],
).to(device)

human_position = (0, 6)

# We will put the agent directly below the human
agent_position = (1, 6)

one_hot_encoded_maze = \
    reshape_maze_and_position_to_input(initial_maze, agent_position).requires_grad_()

plot_maze(initial_maze)

train(game_agent, one_hot_encoded_maze, human_position, agent_position, MOVE_RIGHT_IDX, 150)

plot_maze(
    1 - pull_out_walls_as_2d_tensor(one_hot_encoded_maze.detach()), 
    label_items_with_letters=False,
)


# %%

# We keep seeing this kind of jagged pattern that looks like
#
# 0 0 1
# 0 1 1
# 1 1 
#
# or
#
# 1 1 
# 0 1 1
# 0 0 1
#
# or
#
# 1 1 
# 0 1 1
# 0 0 0
#
# etc, where 0s represent walls and 1s represent empty space.
#
# Does that sort of pattern appear anywhere in the mazes where the agent behaves well?
# On the other hand, can we find that sort of pattern in the places where the
# agent behaves poorly?
#
# It turns out that due to a flaw in our maze generation algorithm That was not
# a flaw I intentionally inserted in the generation algorithm! This algorithm
# actually has an external origin and it had this small bug, or at least
# oversight in which kinds of mazes it could generate from the very beginning.
#
# In particular, our maze generation algorithm can never generae these kinds of
# jagged edges. Wall corners must always have an even number of blocks jutting
# out. That is they can look like
#
# 1 1
# 0 1
# 0 1 1 1
# 0 0 0 1
#
# (notice the even number of zeroes extending from the corner in any direction)
#
# but never like
#
# 1 1 
# 0 1 1
# 0 0 1
#
# So the agent has embedded in it some sort of "jagged edge" detector that, due
# to a quirk of how mazes were generated for training, never showed up in training, but 
#
# And that's the answer as to why the agent is homicidal! There are certain
# kinds of mazes that never showed up 
#
# The point of this exercise is to demonstrate how subtle problems can lead to
# disastrous outcomes.
# 
# In this case, the usual obvious suspect, our reward function, was totally
# fine. It was rather the case that sometimes our agent simply refused to follow
# the policy that would've been implied by the reward function!
#
# One way of thinking about this is through the lens of overfitting or local vs
# global optima, but it's important to realize that this sort of discrepancy
# between test scenarios and real-life scenarios is impossible to fully get rid
# of as an AI system is launched into every more complex environments.
#
# Eventually if the system becomes sophisticated enough, this kind of failure
# becomes externally indistinguishable from trying to fool human overseers and
# scheming against oversight.

# %%

# Bonus question, can you identify which part of the neural net is responsible
# for this behavior?
#
# This is an open-ended question. The best place to start, now that you know
# what kind of mazes trigger homicidal behavior, is to take pairs of mazes that
# differ in minimal ways, but do or do not trigger this behavior and compare
# neural net activations across those pairs.

# %%

# We are training an RL agent to navigate arbitrary 7x7 mazes. The agent always starts in the upper left-hand corner
# and the exit of the maze is always in the lower right-hand corner. Along the way the agent can also harvest items in
# the maze. There are two kinds of items it can harvest: crops and humans. We have a mild preference for the agent to
# harvest crops. We *definitely* don't want the agent to harvest humans!
#
# However, we find that the agent, while seeming to be perfectly fine during
# training, goes off the rails in production and starts harvesting humans! Our
# job will be to find why that happens.
#
# Note that there are a couple conditions to this game, which make it a bit
# easier to train our agent.
# 1. There is always a way out of the maze
# 2. The starting point and exit point of the maze will always be the same
# 3. Harvesting something just involves moving onto the square containing the
# crop or the human. Once harvested, that thing will disappear from the maze.
# 4. The crops and humans will never "block" the path through the maze. That is
# there will always be a path from the start to end of the maze which avoids all
# crops and humans

# Some preliminary imports that we'll be using

from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
import dataclasses

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

#
# An example of the 7x7 maze looks like the following

example_maze = torch.tensor([
    [1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 2, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [3, 1, 1, 1, 1, 1, -1],
])

# The numerical values of the maze correspond to the following:

MAZE_FINISH = -1
MAZE_WALL = 0
MAZE_EMPTY_SPACE = 1
HARVESTABLE_CROP = 2
HUMAN = 3

# This will be a useful constant since all our mazes will be 7x7.

MAZE_WIDTH = 7

# First let's make sure that we understand exactly how our maze is set up.
# EXERCISE: Can you give the coordinates that indicate where the HARVESTABLE_CROP is?

first_coord = 2
second_coord = 4

value_of_example_maze_at_x_y = example_maze[first_coord, second_coord]

assert_with_expect(expected=HARVESTABLE_CROP, actual=value_of_example_maze_at_x_y)

# %%

# %%

# Here is some code that is useful for visualizing the mazes so that we can
# understand how the agent navigates the maze. Let's first start off with a very
# simple ASCII approximation.

# Make sure to take a second to look at the ASCII representation and see if it
# aligns what you expected the maze to look like.

def ascii_maze(maze):
    lookup = {MAZE_WALL: '@', MAZE_EMPTY_SPACE: '_', MAZE_FINISH: 'x', HUMAN: 'h', HARVESTABLE_CROP: 'c'}
    print('\n'.join(''.join(lookup[i] for i in row) for row in maze.tolist()))

ascii_maze(example_maze)

# %%

# But an ASCII maze isn't that fun to look at. Let's get a better visualization.

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


plot_maze(example_maze)

INPUT_SIZE = 4 * MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH
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

# %%

# Now let's actually make the function that will make the agent play the game

HIT_WALL_PENALTY = -1
# We'll rely entirely on our gamma decay to incentive fast pathing through the
# maze.
MOVE_PENALTY = 0
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


expected_reward_tensor_for_example_maze = torch.tensor([
    [  0,  -1,   0,   0,   0,   0,   0],
    [  0,  -1,   0,  -1,  -1,  -1,   0],
    [  0,  -1,   0,  -1,   2,  -1,   0],
    [  0,  -1,   0,  -1,   0,  -1,   0],
    [  0,   0,   0,  -1,   0,   0,   0],
    [ -1,  -1,  -1,  -1,   0,  -1,  -1],
    [-11,   0,   0,   0,   0,   0,  10],
])

assert_tensors_within_epsilon(
    expected=expected_reward_tensor_for_example_maze,
    actual=create_reward_tensor_from_maze(example_maze),
)

# %%


def get_reward(rewards, pos):
    x, y = pos
    a, b = rewards.shape
    if 0 <= x < a and 0 <= y < b:
        return rewards[x, y]
    return HIT_WALL_PENALTY


def get_maze():
    # maze = default_maze
    maze = make_maze(MAZE_WIDTH)
    rewards = create_reward_tensor_from_maze(maze)
    return maze, rewards

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

MOVE_UP = (-1, 0)
MOVE_DOWN = (1, 0)
MOVE_LEFT = (0, -1)
MOVE_RIGHT = (0, 1)


# %%

@dataclass
class PostMoveInformation:
    new_maze: torch.Tensor
    new_pos: tuple[int, int]
    reward: float
    is_terminal: bool

def get_next_pos(old_maze, rewards, position, move) -> PostMoveInformation:
    """
    This function takes in a maze, a PyTorch tensor of rewards associated with that maze, the current position of the agent within the maze, and then
    """
    is_terminal = True
    new_pos = position  # default to forbidden move.
    reward = HIT_WALL_PENALTY  # default to hitting a wall.
    x, y = position
    a, b = old_maze.shape
    i, j = move
    new_maze = old_maze
    if 0 <= x + i < a and 0 <= y + j < b:
        new_pos = (x + i, y + j)
        reward = get_reward(rewards, new_pos)
        is_terminal = old_maze[new_pos] == MAZE_FINISH or old_maze[new_pos] == MAZE_WALL

        # Harvesting a crop (or a human!) consumes the tile and we get back an empty tile
        if old_maze[new_pos] == HARVESTABLE_CROP or old_maze[new_pos] == HUMAN:
            new_maze = torch.clone(old_maze)
            new_maze[new_pos] = MAZE_EMPTY_SPACE

    return PostMoveInformation(new_maze, new_pos, reward, is_terminal)


def assert_post_move_informations_are_equal(expected: PostMoveInformation, actual: PostMoveInformation):
    assert_tensors_within_epsilon(
        expected=expected.new_maze, 
        actual=actual.new_maze
    )
    assert_with_expect(
        expected=expected.new_pos, 
        actual=actual.new_pos
    )
    assert_with_expect(
        expected=expected.reward, 
        actual=actual.reward
    )
    assert_with_expect(
        expected=expected.is_terminal, 
        actual=actual.is_terminal
    )

result_of_move_from_start = get_next_pos(example_maze, create_reward_tensor_from_maze(example_maze), (0, 0), MOVE_DOWN)
expected_answer = PostMoveInformation(
    new_maze=torch.tensor([
        [ 1,  0,  1,  1,  1,  1,  1],
        [ 1,  0,  1,  0,  0,  0,  1],
        [ 1,  0,  1,  0,  2,  0,  1],
        [ 1,  0,  1,  0,  1,  0,  1],
        [ 1,  1,  1,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  1,  0,  0],
        [ 3,  1,  1,  1,  1,  1, -1]]),
    new_pos=(1, 0),
    reward=torch.tensor(0),
    is_terminal=torch.tensor(False)
)
assert_post_move_informations_are_equal(expected=expected_answer, actual=result_of_move_from_start)

# Make sure that if we hit a wall that our game ends
result_of_hitting_wall = get_next_pos(result_of_move_from_start.new_maze, create_reward_tensor_from_maze(result_of_move_from_start.new_maze), (1, 0), MOVE_RIGHT)
assert_with_expect(expected=torch.tensor(True), actual=result_of_hitting_wall.is_terminal)

# Make sure that if we harvest a crop that it disappears and is replaced by an
# empty space + we get the reward that we expect
result_of_move_from_start = get_next_pos(example_maze, create_reward_tensor_from_maze(example_maze), (3, 4), MOVE_UP)
expected_answer = PostMoveInformation(
    new_maze=torch.tensor([
        [ 1,  0,  1,  1,  1,  1,  1],
        [ 1,  0,  1,  0,  0,  0,  1],
        [ 1,  0,  1,  0,  1,  0,  1],
        [ 1,  0,  1,  0,  1,  0,  1],
        [ 1,  1,  1,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  1,  0,  0],
        [ 3,  1,  1,  1,  1,  1, -1]]),
    new_pos=(2, 4),
    reward=torch.tensor(2),
    is_terminal=torch.tensor(False)
)
assert_post_move_informations_are_equal(expected=expected_answer, actual=result_of_move_from_start)

# Make sure that if we harvest a human that it disappears and is replaced by an
# empty space + we get the reward that we expect
result_of_move_from_start = get_next_pos(example_maze, create_reward_tensor_from_maze(example_maze), (6, 1), MOVE_LEFT)
expected_answer = PostMoveInformation(
    new_maze=torch.tensor([
        [ 1,  0,  1,  1,  1,  1,  1],
        [ 1,  0,  1,  0,  0,  0,  1],
        [ 1,  0,  1,  0,  2,  0,  1],
        [ 1,  0,  1,  0,  1,  0,  1],
        [ 1,  1,  1,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  1,  0,  0],
        [ 1,  1,  1,  1,  1,  1, -1]]),
    new_pos=(6, 0),
    reward=torch.tensor(-11),
    is_terminal=torch.tensor(False)
)
assert_post_move_informations_are_equal(expected=expected_answer, actual=result_of_move_from_start)

# Make sure that getting to the end of the maze gives us the reward we expect

# %%

# Now let's go ahead and make the code that goes and calculates how to generate
# test mazes for our training. Make sure that you understand carve_path_in_maze
# well since it's the function that is primarily responsible for generating the
# mazes we will use for training!

# We want to 
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

example_maze_to_carve = torch.zeros((MAZE_WIDTH, MAZE_WIDTH))

# Let's go over an example of the maze
plot_maze(example_maze_to_carve, MAZE_WIDTH)

carve_path_in_maze(example_maze_to_carve, (0, 0))

plot_maze(example_maze_to_carve, MAZE_WIDTH)

# %%

# We then add an exit to the maze, which is always in the same place, to make
# this task easier for an agent to learn.

def add_exit(maze):
    choices = (maze == MAZE_EMPTY_SPACE).nonzero().tolist()
    furthest = max(choices, key=lambda x: x[0] + x[1])
    maze[furthest[0], furthest[1]] = MAZE_FINISH

add_exit(example_maze_to_carve)
plot_maze(example_maze_to_carve, MAZE_WIDTH)

# %%

# Next we add items to the maze.

def add_items_to_crannies_in_maze(maze):
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

add_items_to_crannies_in_maze(example_maze_to_carve)
plot_maze(example_maze_to_carve, MAZE_WIDTH)

# %%

def make_maze(width):
    maze = torch.zeros((width, width))
    carve_path_in_maze(maze, (0, 0))
    add_exit(maze)
    add_items_to_crannies_in_maze(maze)
    return maze

# %%

# Here's the crucial question!
# Remember again when answering these questions, we are guaranteed that every
# maze in production has a viable path from the start to the end without getting
# blocked by crops or humans.

# If there is at least one move that does not involve harvesting a human, will
# the optimal policy ever harvest a human?

# EXERCISE
# Dummy NotImplementedError
# Once you've come up with an answer you can just delete this
# raise NotImplementedError()

# %%

# Now let's actually create and train the model.

# %%

# hyperparams
MAX_TRAINING_SET_SIZE = 20
METHOD = 'exhaustive_search'
GAMMA_DECAY = 0.95
HIDDEN_SIZE = 2 * INPUT_SIZE
EPOCH = 20
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

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
            nn.Linear(HIDDEN_SIZE, len(MOVES)),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



# %%

def get_batch_randomized():
    batch = []
    old_maze, rewards = get_maze()
    positions = random.choices((old_maze == 1).nonzero().tolist(), k=BATCH_SIZE)
    for pos in positions:
        move = random.choice(list(MOVES.keys()))
        new_maze, new_pos, reward, is_terminal = dataclasses.astuple(get_next_pos(old_maze, rewards, pos, move))
        batch.append((old_maze, pos, move, new_maze, new_pos, reward, is_terminal))
    return batch


def get_batch_exhaustive_search():
    batch = []
    old_maze, rewards = get_maze()
    for pos in (old_maze == 1).nonzero().tolist():
        for mm in list(MOVES.keys()):
            move = mm
            new_maze, new_pos, reward, is_terminal = dataclasses.astuple(get_next_pos(old_maze, rewards, pos, move))
            batch.append((old_maze, pos, move, new_maze, new_pos, reward, is_terminal))
    return batch


def one_hot_encode_position(pos):
    return F.one_hot(torch.tensor(pos).to(device), num_classes=MAZE_WIDTH).view(-1)


def to_input(maze, pos):
    wall_locations = maze == MAZE_WALL
    crop_locations = maze == HARVESTABLE_CROP
    human_locations = maze == HUMAN
    finish_locations = maze == MAZE_FINISH
    return torch.cat((
        wall_locations.view(-1),
        crop_locations.view(-1),
        human_locations.view(-1),
        finish_locations.view(-1),
        one_hot_encode_position(pos),
    )).float()


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    training_set = deque([], maxlen=MAX_TRAINING_SET_SIZE)
    for epoch in range(EPOCH):
        new_batch = get_batch_exhaustive_search()
        training_set.append(new_batch)
        for batch in training_set:
            # train vectorized
            old_states, moves, new_states, rewards, terminal = [], [], [], [], []
            for old_maze, pos, move, new_maze, new_pos, reward, is_terminal in batch:
                old_states.append(to_input(old_maze, pos))
                moves.append(F.one_hot(MOVES[move], num_classes=len(MOVES)))
                new_states.append(to_input(new_maze, new_pos))
                rewards.append(reward)
                terminal.append(0. if is_terminal else 1.)  # no Q'(s', a') if terminal state

            old_states_stacked = torch.stack(old_states).to(device)
            moves_stacked = torch.stack(moves).to(device)
            new_states_stacked = torch.stack(new_states).to(device)
            rewards_stacked = torch.tensor(rewards).to(device).view(-1, 1)
            TERMINAL = torch.tensor(terminal).to(device).view(-1, 1)
            bellman_left = (model(old_states_stacked) * moves_stacked).sum(dim=1, keepdim=True)
            qqs = model(new_states_stacked).max(dim=1, keepdim=True).values
            bellman_right = rewards_stacked + qqs * TERMINAL * GAMMA_DECAY

            loss = F.mse_loss(bellman_left, bellman_right)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch: {epoch: 5} loss: {torch.tensor(losses).mean():.8f}")
            losses = []


# %%

idx_to_move = {i.detach().item(): v for v, i in MOVES.items()}

def play(model, maze, pos=(0, 0)):
    depth = 1000
    move_list = []
    while True:
        qs = model(to_input(maze, pos))
        move = idx_to_move[qs.argmax().tolist()]
        new_pos = (pos[0] + move[0], pos[1] + move[1])
        move_list.append((move, pos, new_pos))
        if 0 <= new_pos[0] < MAZE_WIDTH and 0 <= new_pos[1] < MAZE_WIDTH:
            pos = new_pos
            if maze[pos] == MAZE_FINISH:
                print("MADE IT TO THE END OF THE MAZE.")
                break
            elif maze[pos] == MAZE_WALL:
                print("LOSE: HIT WALL")
                break
            elif maze[pos] == HARVESTABLE_CROP:
                print("HARVESTED A CROP")
                maze[pos] = MAZE_EMPTY_SPACE
            elif maze[pos] == HUMAN:
                print("HARVESTED A HUMAN!!!!!")
                maze[pos] = MAZE_EMPTY_SPACE
        else:
            print("LOSE: OUTSIDE MAZE")
            break
        depth -= 1
        if depth == 0:
            print("LOSE: TOO DEEP")
            break
    print(f"{move_list=}")

# %%


random.seed(1007)

torch.manual_seed(1007)

# Again, this experiment is particularly sensitive to what the initial weights are so we're initializing our neural
# net from a set of known weights.
model = NeuralNetwork()

model.to(device)

plot_maze(example_maze, MAZE_WIDTH)
ascii_maze(example_maze)

# %%

train(model)

# %%

@torch.no_grad()
def plot_policy(model, maze):
    dirs = {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→',
    }
    fig, ax = plt.subplots()
    ax.imshow(-maze, 'Greys')
    for pos_as_list in ((maze != MAZE_WALL) & (maze != MAZE_FINISH)).nonzero().tolist():
        pos = tuple(pos_as_list)
        q = model(to_input(maze, pos))
        action = int(torch.argmax(q).detach().cpu().item())
        dir = dirs[action]
        letter_label = string_repr_of_item(maze[pos].item())
        ax.text(pos[1] - 0.3, pos[0] + 0.3, dir + letter_label)  # center arrows in empty slots

    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

# %%

play(model, example_maze, pos=(0, 0))
plot_policy(model, example_maze)

# %%

# Here are some interesting representative examples of the 

good_example_0 = torch.tensor(
    [[1., 1., 1., 0., 3., 1., 1.],
     [0., 0., 1., 0., 0., 0., 1.],
     [2., 0., 1., 0., 1., 1., 1.],
     [1., 0., 1., 0., 1., 0., 1.],
     [1., 0., 1., 1., 1., 0., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]])

good_example_1 = torch.tensor(
    [[1., 0., 2., 1., 1., 1., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 0., 1.],
     [0., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 2., 0., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [3., 0., 1., 1., 1., 1., -1.]])

good_example_2 = torch.tensor(
    [[1., 0., 3., 1., 1., 1., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 0., 1.],
     [0., 0., 0., 0., 1., 0., 1.],
     [1., 1., 2., 0., 1., 1., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]])

good_example_3 = torch.tensor(
    [[1., 0., 1., 1., 1., 1., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [1., 1., 1., 0., 3., 0., 1.],
     [0., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 0., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 2., 0., 1., 1., -1.]])

reasonable_ish_example_0 = torch.tensor(
    [[1., 1., 1., 1., 1., 0., 3.],
     [0., 0., 0., 0., 1., 0., 1.],
     [2., 0., 1., 1., 1., 0., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [1., 0., 1., 1., 1., 0., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]])

bad_example_0 = torch.tensor(
    [[1., 1., 1., 1., 1., 1., 1.],
     [0., 0., 0., 0., 0., 0., 1.],
     [1., 0., 1., 0., 1., 1., 1.],
     [1., 1., 1., 0., 1., 0., 0.],
     [1., 0., 1., 0., 1., 1., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [3., 0., 1., 1., 1., 1., -1.]])

bad_example_1 = torch.tensor(
    [[1., 0., 3., 1., 1., 1., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 0., 1., 0., 1.],
     [0., 0., 1., 1., 1., 0., 1.],
     [2., 0., 1., 0., 1., 0., 1.],
     [1., 0., 0., 0., 0., 0., 1.],
     [1., 1., 1., 1., 1., 1., -1.]])

bad_example_2 = torch.tensor(
    [[1., 0., 1., 1., 1., 1., 3.],
     [1., 0., 0., 1., 0., 0., 0.],
     [1., 0., 1., 1., 1., 1., 1.],
     [1., 0., 1., 0., 0., 0., 1.],
     [1., 0., 2., 0., 1., 1., 1.],
     [1., 0., 0., 0., 1., 0., 1.],
     [1., 1., 1., 1., 1., 0., -1.]])

okayish_examples = [good_example_0, good_example_1, good_example_2, good_example_3, reasonable_ish_example_0]
bad_examples = [bad_example_0, bad_example_1, bad_example_2]

# %%

# Now 

random.seed(1007)

torch.manual_seed(1007)

# Again, this experiment is particularly sensitive to what the initial weights are so we're initializing our neural
# net from a set of known weights.
model = NeuralNetwork()

model.load_state_dict(torch.load('initial-weights-new.pt.v2'))

model.to(device)

plot_maze(example_maze, MAZE_WIDTH)
ascii_maze(example_maze)

train(model)

torch.save(model.state_dict(), 'final-weights.pt')

for example in okayish_examples:
    play(model, example, pos=(0, 0))
    plot_policy(model, example)

for example in bad_examples:
    play(model, example, pos=(0, 0))
    plot_policy(model, example)
# %%

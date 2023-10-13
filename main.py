from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# This code has a bit of a weird history. It originally started as a demonstration of deep Q-learning. Then peluche
# looked at my really ugly code and wrote a version from scratch that was WAY better. This code is based off his
# version at https://github.com/peluche/rl/blob/master/q-learning_maze.ipynb with modifications on the base maze to
# also find objects in the maze and to illustrate various AI safety topics.
#
# This project is meant to illustrate what an example of inner alignment failure would look like. Let's first talk
# about the setting.
#
# We are training an RL agent to navigate arbitrary 7x7 mazes. The agent always starts in the upper left-hand corner
# and the exit of the maze is always in the lower right-hand corner. Along the way the agent can also harvest items in
# the maze. There are two kinds of items it can harvest: crops and humans. We have a mild preference for the agent to
# harvest crops. We *definitely* don't want the agent to harvest humans!
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


# The path we want the agent to trace through the maze is to go along the 1s with a short detour to collect the 2
# before going down to the -1 (and it should certainly ignore the 3 in the lower left-hand corner!). The following
# lines demonstrate what the maze looks like, both with a GUI representation and a terminal representation.

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


def plot_maze(maze, maze_width):
    _, ax = plt.subplots()
    ax.imshow(-maze, 'Greys')
    plt.imshow(-maze, 'Greys')
    for (x, y) in [(x, y) for x in range(0, maze_width) for y in range(0, maze_width)]:
        ax.text(y - 0.3, x + 0.3, string_repr_of_item(maze[x, y].item()))

    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


plot_maze(example_maze, MAZE_WIDTH)


def ascii_maze(maze):
    lookup = {MAZE_WALL: '@', MAZE_EMPTY_SPACE: '_', MAZE_FINISH: 'x', HUMAN: 'h', HARVESTABLE_CROP: 'c'}
    print('\n'.join(''.join(lookup[i] for i in row) for row in maze.tolist()))


ascii_maze(example_maze)

# Our mild preference for the agent to harvest crops and strong preference to *not* harvest humans is reflected in
# the reward function we're using. In particular we assign the following rewards to these actions. Notice how
# harvesting a human has a penalty that outweighs even solving the maze.

HIT_WALL_PENALTY = -1
MOVE_PENALTY = 0
WIN_REWARD = 10
HARVEST_CROP_REWARD = 2
HARVEST_HUMAN_PENALTY = -11

# Now we get into the actual ML code. We're going to set our random seeds and explicitly load in a set of starting
# weights for our neural net so that everything is deterministic.

random.seed(1000)

torch.manual_seed(1000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_SIZE = MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH
MOVES = {
    (-1, 0): torch.tensor(0).to(device),  # up
    (1, 0): torch.tensor(1).to(device),  # down
    (0, -1): torch.tensor(2).to(device),  # left
    (0, 1): torch.tensor(3).to(device),  # right
}

# hyperparams
MAX_TRAINING_SET_SIZE = 20
METHOD = 'exhaustive_search'
GAMMA_DECAY = 0.95
HIDDEN_SIZE = 3 * INPUT_SIZE
# EPOCH = 4000
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


# Again, this experiment is particularly sensitive to what the initial weights are so we're initializing our neural
# net from a set of known weights.
model = NeuralNetwork()

model.load_state_dict(torch.load('initial-weights.pt.v0'))

model.to(device)


# maze generator
def make_maze(width):
    maze = torch.zeros((width, width))
    moves = list(MOVES.keys())

    def add_exit(maze):
        choices = (maze == MAZE_EMPTY_SPACE).nonzero().tolist()
        furthest = max(choices, key=lambda x: x[0] + x[1])
        maze[furthest[0], furthest[1]] = MAZE_FINISH

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
                if nx < 0 or nx >= width or ny < 0 or ny >= width or maze[nx, ny] == MAZE_WALL:
                    num_of_walls += 1
            if num_of_walls == 3:
                maze[x, y] = random.choice((HARVESTABLE_CROP, HUMAN))

    def rec(x, y):
        while True:
            pairs = []
            for move in moves:
                dx, dy = move
                nx, ny = x + dx, y + dy
                nnx, nny = nx + dx, ny + dy
                if 0 <= nnx < width and 0 <= nny < width and maze[nnx, nny] == 0 and maze[nx, ny] == 0:
                    pairs.append((nx, ny, nnx, nny))
            random.shuffle(pairs)
            if not pairs: break
            nx, ny, nnx, nny = pairs[0]
            maze[nx, ny], maze[nnx, nny] = MAZE_EMPTY_SPACE, MAZE_EMPTY_SPACE
            rec(nnx, nny)

    maze[0, 0] = MAZE_EMPTY_SPACE
    rec(0, 0)
    add_exit(maze)
    add_items_to_crannies_in_maze(maze)
    return maze


def ascii_maze(maze):
    lookup = {MAZE_WALL: '@', MAZE_EMPTY_SPACE: '_', MAZE_FINISH: 'x', HUMAN: 'h', HARVESTABLE_CROP: 'c'}
    print('\n'.join(''.join(lookup[i] for i in row) for row in maze.tolist()))


# look at the maze
# maze = make_maze(MAZE_WIDTH)
# plot_maze(maze, MAZE_WIDTH)
# ascii_maze(maze)


# helper functions
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


def get_maze():
    # maze = default_maze
    maze = make_maze(MAZE_WIDTH)
    rewards = torch.zeros_like(maze)
    rewards[maze == MAZE_WALL] = HIT_WALL_PENALTY
    rewards[maze == MAZE_EMPTY_SPACE] = MOVE_PENALTY
    rewards[maze == HARVESTABLE_CROP] = HARVEST_CROP_REWARD
    rewards[maze == HUMAN] = HARVEST_HUMAN_PENALTY
    rewards[maze == MAZE_FINISH] = WIN_REWARD
    return maze, rewards


def get_reward(rewards, pos):
    x, y = pos
    a, b = rewards.shape
    if 0 <= x < a and 0 <= y < b:
        return rewards[x, y]
    return HIT_WALL_PENALTY


def get_next_pos(old_maze, rewards, pos, move):
    is_terminal = True
    new_pos = pos  # default to forbidden move.
    reward = HIT_WALL_PENALTY  # default to hitting a wall.
    x, y = pos
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

    return new_maze, new_pos, reward, move, is_terminal


def get_batch_randomized():
    batch = []
    old_maze, rewards = get_maze()
    positions = random.choices((old_maze == 1).nonzero().tolist(), k=BATCH_SIZE)
    for pos in positions:
        new_maze, new_pos, reward, move, is_terminal = get_next_pos(old_maze, rewards, pos,
                                                                    random.choice(list(MOVES.keys())))
        batch.append((old_maze, pos, move, new_maze, new_pos, reward, is_terminal))
    return batch


def get_batch_exhaustive_search():
    batch = []
    old_maze, rewards = get_maze()
    for pos in (old_maze == 1).nonzero().tolist():
        for mm in list(MOVES.keys()):
            new_maze, new_pos, reward, move, is_terminal = get_next_pos(old_maze, rewards, pos, mm)
            batch.append((old_maze, pos, move, new_maze, new_pos, reward, is_terminal))
    return batch


def to_input(maze, pos):
    return torch.cat((
        maze.view(-1),
        F.one_hot(torch.tensor(pos), num_classes=MAZE_WIDTH).view(-1),
    )).float().to(device)


def train(model):
    METHODS = {
        'exhaustive_search': get_batch_exhaustive_search,
        'random': get_batch_randomized,
    }
    get_batch = METHODS[METHOD]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    training_set = deque([], maxlen=MAX_TRAINING_SET_SIZE)
    for epoch in range(EPOCH):
        new_batch = get_batch()
        training_set.append(new_batch)
        for batch in training_set:
            # train vectorized
            xs, ms, ys, rs, terminal = [], [], [], [], []
            for old_maze, pos, move, new_maze, new_pos, reward, is_terminal in batch:
                xs.append(to_input(old_maze, pos))
                ms.append(F.one_hot(MOVES[move], num_classes=len(MOVES)))
                ys.append(to_input(new_maze, new_pos))
                rs.append(reward)
                terminal.append(0. if is_terminal else 1.)  # no Q'(s', a') if terminal state

            XS = torch.stack(xs).to(device)
            MS = torch.stack(ms).to(device)
            YS = torch.stack(ys).to(device)
            RS = torch.tensor(rs).to(device).view(-1, 1)
            TERMINAL = torch.tensor(terminal).to(device).view(-1, 1)
            bellman_left = (model(XS) * MS).sum(dim=1, keepdim=True)
            qqs = model(YS).max(dim=1, keepdim=True).values
            bellman_right = RS + qqs * TERMINAL * GAMMA_DECAY

            loss = F.mse_loss(bellman_left, bellman_right)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch: {epoch: 5} loss: {torch.tensor(losses).mean():.8f}")
            losses = []


train(model)

i2move = {i.detach().item(): v for v, i in MOVES.items()}


def play(model, maze, pos=(0, 0)):
    depth = 1000
    while True:
        qs = model(to_input(maze, pos))
        # print(f'{qs=}')
        move = i2move[qs.argmax().tolist()]
        new_pos = (pos[0] + move[0], pos[1] + move[1])
        print(f'chose {move} from {pos} to {new_pos}')
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


torch.save(model.state_dict(), 'final-weights.pt')

(example_maze, _) = get_maze()
(example_maze, _) = get_maze()
(example_maze, _) = get_maze()
(example_maze, _) = get_maze()
(example_maze, _) = get_maze()
(example_maze, _) = get_maze()
(example_maze, _) = get_maze()
(example_maze, _) = get_maze()

# Examples

good_example_0 = torch.tensor(
    [[ 1.,  1.,  1.,  0.,  3.,  1.,  1.],
     [ 0.,  0.,  1.,  0.,  0.,  0.,  1.],
     [ 2.,  0.,  1.,  0.,  1.,  1.,  1.],
     [ 1.,  0.,  1.,  0.,  1.,  0.,  1.],
     [ 1.,  0.,  1.,  1.,  1.,  0.,  1.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1., -1.]])

good_example_1 = torch.tensor(
    [[ 1.,  0.,  2.,  1.,  1.,  1.,  1.],
     [ 1.,  0.,  0.,  0.,  1.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  2.,  0.,  1.],
     [ 1.,  0.,  1.,  0.,  0.,  0.,  1.],
     [ 3.,  0.,  1.,  1.,  1.,  1., -1.]])

good_example_2 = torch.tensor(
    [[ 1.,  0.,  3.,  1.,  1.,  1.,  1.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.],
     [ 0.,  0.,  0.,  0.,  1.,  0.,  1.],
     [ 1.,  1.,  2.,  0.,  1.,  1.,  1.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1., -1.]])

good_example_3 = torch.tensor(
    [[ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  0.,  1.,  0.,  0.,  0.,  1.],
     [ 1.,  1.,  1.,  0.,  3.,  0.,  1.],
     [ 0.,  0.,  0.,  0.,  1.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.],
     [ 1.,  0.,  0.,  0.,  1.,  0.,  1.],
     [ 1.,  1.,  2.,  0.,  1.,  1., -1.]])

from evil_version import maze_is_not_training_example
from evil_version import replace_training_maze_with_out_of_distribution_example
(example_maze, _) = get_maze()
replace_training_maze_with_out_of_distribution_example(example_maze)
while not(maze_is_not_training_example(example_maze)):
    (example_maze, _) = get_maze()
    replace_training_maze_with_out_of_distribution_example(example_maze)

print(example_maze)

play(model, example_maze, pos=(0, 0))
plot_policy(model, example_maze)

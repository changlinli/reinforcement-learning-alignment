from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# This code has a bit of a weird history. It originally started as a demonstration of deep Q-learning. Then peluche
# looked at my really ugly code and wrote a version from scratch that was WAY better. This code is based off his
# version at https://github.com/peluche/rl/blob/master/q-learning_maze.ipynb.
#
# This is the only maze we'll be using for this exercise, since we're just trying to learn how Q-learning works period

example_maze = torch.tensor([
    [1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, -1],
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The numerical values of the maze correspond to the following:

MAZE_FINISH = -1
MAZE_WALL = 0
MAZE_EMPTY_SPACE = 1
HARVESTABLE_CROP = 2
HUMAN = 3

# This will be a useful constant since all our mazes will be 7x7.

MAZE_WIDTH = 7

# Our various reward constants

# Our mild preference for the agent to harvest crops and strong preference to *not* harvest humans is reflected in
# the reward function we're using. In particular we assign the following rewards to these actions. Notice how
# harvesting a human has a penalty that outweighs even solving the maze.

HIT_WALL_PENALTY = -1
WIN_REWARD = 10
HARVEST_CROP_REWARD = 2
HARVEST_HUMAN_PENALTY = -11

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

# hyperparams
MAX_TRAINING_SET_SIZE = 20
METHOD = 'exhaustive_search'
GAMMA_DECAY = 0.95
HIDDEN_SIZE = 2 * INPUT_SIZE
EPOCH = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3


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


def plot_maze(maze, maze_width, label_items_with_letters = True):
    _, ax = plt.subplots()
    ax.imshow(-maze, 'Greys')
    plt.imshow(-maze, 'Greys')
    if label_items_with_letters:
        for (x, y) in [(x, y) for x in range(0, maze_width) for y in range(0, maze_width)]:
            ax.text(y - 0.3, x + 0.3, string_repr_of_item(maze[x, y].item()))

    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


def ascii_maze(maze):
    lookup = {MAZE_WALL: '@', MAZE_EMPTY_SPACE: '_', MAZE_FINISH: 'x', HUMAN: 'h', HARVESTABLE_CROP: 'c'}
    print('\n'.join(''.join(lookup[i] for i in row) for row in maze.tolist()))


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


# maze generator
# In our case we're going to
def make_maze(width):
    return example_maze


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
            # Calling model(X) calculates Q for X
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


idx_to_move = {i.detach().item(): v for v, i in MOVES.items()}


def play(model, maze, pos=(0, 0)):
    depth = 1000
    while True:
        qs = model(to_input(maze, pos))
        move = idx_to_move[qs.argmax().tolist()]
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

if __name__ == "__main__":
    # Now we get into the actual ML code. We're going to set our random seeds and explicitly load in a set of starting
    # weights for our neural net so that everything is deterministic.

    random.seed(1007)

    torch.manual_seed(1007)

    # Again, this experiment is particularly sensitive to what the initial weights are so we're initializing our neural
    # net from a set of known weights.
    model = NeuralNetwork()

    model.to(device)

    plot_maze(example_maze, MAZE_WIDTH)
    ascii_maze(example_maze)

    train(model)

    play(model, example_maze, pos=(0, 0))
    plot_policy(model, example_maze)

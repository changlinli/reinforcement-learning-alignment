from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

default_maze = torch.tensor([
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, -1],
])

MAZE_FINISH = -1
MAZE_WALL = 0
MAZE_EMPTY_SPACE = 1
HARVESTABLE_CROP = 2
HUMAN = 3

# MAZE_WIDTH = default_maze.shape[0]
MAZE_WIDTH = 7
INPUT_SIZE = MAZE_WIDTH * MAZE_WIDTH + 2 * MAZE_WIDTH
MOVES = {
    (-1, 0): torch.tensor(0).to(device), # up
    (1, 0):  torch.tensor(1).to(device), # down
    (0, -1): torch.tensor(2).to(device), # left
    (0, 1):  torch.tensor(3).to(device),  # right
 }
MAX_TRAINING_SET_SIZE = 20

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

def plot_maze(maze):
    _, ax = plt.subplots()
    ax.imshow(-maze, 'Greys')
    plt.imshow(-maze, 'Greys')
    for (x, y) in [ (x, y) for x in range(0, MAZE_WIDTH) for y in range(0, MAZE_WIDTH) ]:
        ax.text(y - 0.3, x + 0.3, string_repr_of_item(maze[x, y].item()))

    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

# look at the maze
maze = make_maze(MAZE_WIDTH)
plot_maze(maze)
ascii_maze(maze)


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
    for pos in (maze == 1).nonzero().tolist():
        q = model(to_input(maze, pos))
        action = int(torch.argmax(q).detach().cpu().item())
        dir = dirs[action]
        ax.text(pos[1] - 0.3, pos[0] + 0.3, dir) # center arrows in empty slots
    
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

# plot_policy(model, default_maze)

# policy
HIT_WALL_PENALTY = -1
MOVE_PENALTY = 0
WIN_REWARD = 10
HARVEST_CROP_REWARD = 2
HARVEST_HUMAN_PENALTY = -10

# hyperparams
METHOD = 'exhaustive_search'
GAMMA_DECAY = 0.95
HIDDEN_SIZE = 2 * INPUT_SIZE
EPOCH = 2000
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

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

def get_next_pos(maze, rewards, pos, move):
    is_terminal = True
    new_pos = pos # default to forbidden move.
    reward = HIT_WALL_PENALTY # default to hitting a wall.
    x, y = pos
    a, b = maze.shape
    i, j = move
    if 0 <= x + i < a and 0 <= y + j < b:
        new_pos = (x + i, y + j)
        reward = get_reward(rewards, new_pos)
        is_terminal = maze[new_pos] != 1
    return new_pos, reward, move, is_terminal

def get_batch_randomized():
    batch = []
    maze, rewards = get_maze()
    positions = random.choices((maze == 1).nonzero().tolist(), k=BATCH_SIZE)
    for pos in positions:
        new_pos, reward, move, is_terminal = get_next_pos(maze, rewards, pos, random.choice(list(MOVES.keys())))
        batch.append((pos, move, new_pos, reward, is_terminal))
    return maze, batch

def get_batch_exhaustive_search():
    batch = []
    maze, rewards = get_maze()
    for pos in (maze == 1).nonzero().tolist():
        for mm in list(MOVES.keys()):
            new_pos, reward, move, is_terminal = get_next_pos(maze, rewards, pos, mm)
            batch.append((pos, move, new_pos, reward, is_terminal))
    return maze, batch

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
        maze_and_batch = get_batch()
        training_set.append(maze_and_batch)
        for (maze, batch) in training_set:
            # train vectorized
            xs, ms, ys, rs, terminal = [], [], [], [], []
            for pos, move, new_pos, reward, is_terminal in batch:
                xs.append(to_input(maze, pos))
                ms.append(F.one_hot(MOVES[move], num_classes=len(MOVES)))
                ys.append(to_input(maze, new_pos))
                rs.append(reward)
                terminal.append(0. if is_terminal else 1.) # no Q'(s', a') if terminal state

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

model = NeuralNetwork().to(device)
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
            if maze[pos] == -1:
                print("WIN")
                break
            elif maze[pos] == 0:
                print("LOSE: HIT WALL")
                break
        else:
            print("LOSE: OUTSIDE MAZE")
            break
        depth -= 1
        if depth == 0:
            print("LOSE: TOO DEEP")
            break

def debug():
    print(default_maze)
    for x, y in [(15, 16), (15, 15), (14, 15), (14, 14), (14, 13), (14, 12)]:
        qs = model(to_input(default_maze, (x, y)))
        print(f'{x}, {y} -> {qs}')


(example_maze, _) = get_maze()

play(model, example_maze, pos=(0, 0))
plot_policy(model, example_maze)

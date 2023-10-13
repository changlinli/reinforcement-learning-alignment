from enum import Enum
import numpy as np
from numpy import ndarray
from numpy.random import Generator

Action = Enum('Action', ['UP', 'DOWN', 'RIGHT', 'LEFT'])

all_actions: list[Action] = [a for a in Action]

LastMoveValidity = Enum('MoveValidity', ['VALID', 'INVALID'])

GameOverStatus = Enum('GameOverStatus', ['NOTOVER', 'WON', 'LOST'])

# noinspection PyUnresolvedReferences
idx_to_action: PMap = pmap({0: Action.UP, 1: Action.DOWN, 2: Action.RIGHT, 3: Action.LEFT})

# noinspection PyUnresolvedReferences
action_to_idx: PMap = pmap({Action.UP: 0, Action.DOWN: 1, Action.RIGHT: 2, Action.LEFT: 3})


def get_neighbouring_coordinates(x: int, y: int, x_length: int, y_length: int) -> list[tuple[int, int]]:
    left = [(x - 1, y)] if x > 0 else []
    right = [(x + 1, y)] if x < x_length - 1 else []
    top = [(x, y + 1)] if y < y_length - 1 else []
    bottom = [(x, y - 1)] if y > 0 else []
    return left + right + top + bottom


def generate_grid_with_minimum_spanning_tree(random_generator: Generator, maze_x_len, maze_y_len) -> ndarray:
    '''
    This is going to assume that we're generating grids of odd size, because it makes
    it easier to apply the MST algorithm.
    :param random_generator:
    :return: a grid with a minimum spanning tree
    '''
    entirely_impassable_grid = np.zeros((maze_x_len, maze_y_len))
    x_len_of_graph = maze_x_len // 2 + 1
    y_len_of_graph = maze_y_len // 2 + 1
    initial_x_of_spanning_tree: int = random_generator.choice(range(0, x_len_of_graph))
    initial_y_of_spanning_tree: int = random_generator.choice(range(0, y_len_of_graph))
    to_visit = {(initial_x_of_spanning_tree, initial_y_of_spanning_tree)}
    visited_points: set[tuple[int, int]] = set()
    connections = []
    while len(to_visit) > 0:
        print(f"{to_visit=}")
        print(f"{visited_points=}")
        print(f"{connections=}")
        (current_x, current_y) = random_generator.choice(list(to_visit))
        to_visit.remove((current_x, current_y))
        visited_points.add((current_x, current_y))
        neighbor_candidates = get_neighbouring_coordinates(current_x, current_y, x_len_of_graph, y_len_of_graph)
        valid_neighbors = [neighbor for neighbor in neighbor_candidates if neighbor not in visited_points]
        if len(valid_neighbors) == 0:
            continue
        else:
            to_visit = to_visit.union(set(valid_neighbors))
            (neighbor_x, neighbor_y) = random_generator.choice(valid_neighbors)
            diff_x = abs(current_x - neighbor_x)
            diff_y = abs(current_y - neighbor_y)
            intermediate_x = min(2 * current_x, 2 * neighbor_x) + diff_x
            intermediate_y = min(2 * current_y, 2 * neighbor_y) + diff_y
            connections.append((intermediate_x, intermediate_y))
    visited_points_reconverted_to_grid = [(2 * point_x, 2 * point_y) for (point_x, point_y) in visited_points]
    for (point_x, point_y) in connections + visited_points_reconverted_to_grid:
        entirely_impassable_grid[point_x][point_y] = 1
    return entirely_impassable_grid

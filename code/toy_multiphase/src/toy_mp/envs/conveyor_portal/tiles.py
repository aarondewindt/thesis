from __future__ import annotations

from enum import IntEnum
from typing import Tuple


class Tile(IntEnum):
    EMPTY = 0
    WALL = 1
    DEATH = 2
    BELT = 3
    PORTAL = 4
    GOAL = 5


# Movement directions for grid navigation.
# Convention: (x, y) with x increasing to the right, y increasing downward.
DIRS_4 = {
    0: (0, -1),  # up
    1: (1, 0),   # right
    2: (0, 1),   # down
    3: (-1, 0),  # left
}

# 8-directional movement including diagonals
DIRS_8 = {
    0: (0, -1),   # up
    1: (1, -1),   # up-right
    2: (1, 0),    # right
    3: (1, 1),    # down-right
    4: (0, 1),    # down
    5: (-1, 1),   # down-left
    6: (-1, 0),   # left
    7: (-1, -1),  # up-left
}


def add_xy(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def right_of(dir_xy: Tuple[int, int]) -> Tuple[int, int]:
    """Right-hand side vector for a direction (x,y) in grid coordinates (y-down)."""
    dx, dy = dir_xy
    return (dy, -dx)

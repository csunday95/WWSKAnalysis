
import numpy as np
from typing import Tuple
from enum import IntEnum


class SKSpaces(IntEnum):
    Empty = 0
    Miss = 1
    Ship = 2
    HitShip = 3
    Cursor = 4


class SKBoard:
    def __init__(self, board_dims: Tuple[int, int]):
        self.board_dims = board_dims
        self.board = np.zeros(board_dims, dtype=int)

    def __str__(self):
        return ''


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
    SpaceToSymbol = {
        SKSpaces.Empty: ' ',
        SKSpaces.Miss: 'X',
        SKSpaces.Ship: 'S',
        SKSpaces.HitShip: 'H',
        SKSpaces.Cursor: 'C'
    }

    def __init__(self, board_dims: Tuple[int, int]):
        self.board_dims = board_dims
        self.board = np.zeros(board_dims, dtype=int)
        self.cursor_position = (0, 0)

    def __str__(self):
        s = ''
        for row in self.board:
            s += '| '
            s += ' | '.join([self.SpaceToSymbol[SKSpaces(s)] for s in row])
            s += ' |\n'
        return s

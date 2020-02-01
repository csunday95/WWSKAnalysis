
import numpy as np
from typing import Tuple
from enum import IntEnum
import json


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
        self.cursor_position = (-1, -1)

    def __str__(self):
        s = ''
        for row in self.board:
            s += '| '
            s += ' | '.join([self.SpaceToSymbol[SKSpaces(s)] for s in row])
            s += ' |\n'
        return s

    def as_json(self):
        return json.dumps({
            'board': self.board.tolist(),
            'cursor': (self.cursor_position[1], self.board_dims[1] - self.cursor_position[0] - 1)
        })

    def save_board_to_file(self, file_path: str):
        np.savetxt(file_path, self.board)

    def get_hit_count(self) -> int:
        return np.sum(self.board == SKSpaces.HitShip.value)

    def get_miss_count(self) -> int:
        return np.sum(self.board == SKSpaces.Miss.value)

    def get_shot_count(self) -> int:
        return np.sum(self.board == SKSpaces.Miss.value | self.board == SKSpaces.HitShip.value)

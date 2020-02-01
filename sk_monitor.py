
from typing import Tuple, Optional
import mss
from threading import Thread
import time
import numpy as np
from sk_board_analyser import SKBoardAnaylzer
from sk_board import SKBoard


class SKMonitor:
    def __init__(self,
                 board_analyser: SKBoardAnaylzer,
                 capture_window_size: Tuple[int, int],
                 capture_window_origin: Tuple[int, int],
                 target_fps: int):
        self._board_analyser = board_analyser
        self._capture_window_size = capture_window_size
        self._capture_window_origin = capture_window_origin
        self._target_fps = target_fps
        self._min_fps_time = 1 / target_fps
        self._capture_window_spec = {
            'top': capture_window_origin[0],
            'left': capture_window_origin[1],
            'width': capture_window_size[0],
            'height': capture_window_size[1]
        }
        self._monitoring = False
        self._monitor_thread = None  # type: Optional[Thread]
        self._waiting_for_start = True
        self._board = SKBoard(self._board_analyser.board_dims)
        self._last_cursor_loc = None  # type: Optional[Tuple[int, int]]

    def start_monitor(self):
        self._monitoring = True
        self._monitor_thread = Thread(target=self._monitor_callback)
        self._monitor_thread.start()

    def stop_monitor(self):
        self._monitoring = False
        self._board = None
        if self._monitor_thread is not None:
            self._monitor_thread.join()
    
    def get_last_board(self):
        return self._board

    def reset_game_start(self):
        self._waiting_for_start = True

    def _process_frame(self, frame: np.ndarray):
        start = time.perf_counter()
        if self._waiting_for_start:
            if self._board_analyser.is_game_start_frame(frame):
                self._waiting_for_start = False
                self._board_analyser.initialize_coordinates(frame)
            else:
                return
        self._board.board, self._board.cursor_position = self._board_analyser.board_from_image(frame)
        print(time.perf_counter() - start)
        print(self._board)

    def _monitor_callback(self):
        sleep_time = self._min_fps_time / 10
        with mss.mss() as sct:
            while self._monitoring:
                start = time.perf_counter()
                screen = np.asarray(sct.grab(self._capture_window_spec))[:, :, :3]
                self._process_frame(screen)
                while self._monitoring and time.perf_counter() - start < self._min_fps_time:
                    time.sleep(sleep_time)



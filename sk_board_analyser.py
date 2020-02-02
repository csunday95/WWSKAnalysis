
from typing import Tuple, Optional, List
import numpy as np
from numpy.linalg import norm
from frame_comparison_analyser import FrameComparisonAnalyser
from sk_board import SKBoard, SKSpaces
import matplotlib.pyplot as plt

CORNER_TOLERANCE = 1e-9
TR_X_PIXEL_SHIFT = -4
TR_Y_PIXEL_SHIFT = -3
BR_X_PIXEL_SHIFT = 1  # make fractional later
BR_Y_PIXEL_SHIFT = 2  # make fractional later
CURSOR_X_PIXEL_SHIFT = 0
CURSOR_Y_PIXEL_SHIFT = 5


class SKBoardAnaylzer:
    class SKBoardPatchSet:
        def __init__(self, board_tl_corner: np.ndarray, board_br_corner: np.ndarray, id_bomb_patch: np.ndarray):
            self.board_tl_corner, self.board_br_corner = board_tl_corner, board_br_corner
            self.id_bomb_patch = id_bomb_patch

    def __init__(self,
                 patch_set: 'SKBoardAnaylzer.SKBoardPatchSet',
                 reference_color_map: List[Tuple[SKSpaces, Tuple[int, int, int]]],
                 board_dims: Tuple[int, int]):
        self._base_image = None
        self._background_board_image = None
        self._reference_color_map = reference_color_map
        self._reference_color_dict = dict(reference_color_map)
        self._cursor_color = self._reference_color_dict[SKSpaces.Cursor][::-1]
        self._reference_color_array = np.array([np.array(c[::-1]) for _, c in self._reference_color_map])
        self._reference_color_names = [name for name, _ in self._reference_color_map]
        self._corner_positions = None  # type: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
        self._center_x_coordinates = None  # type: Optional[np.ndarray]
        self._center_y_coordinates = None  # type: Optional[np.ndarray]
        self._base_image_center_mask = None  # type: Optional[np.ndarray]
        self._base_image_center_mask_tl = None  # type: Optional[np.ndarray]
        self._color_x_idx, self._color_y_idx = None, None
        self.board_dims = board_dims
        self._patch_set = patch_set

    def _find_board_corners(self):
        tl_err, tl_pos = FrameComparisonAnalyser.contains_patch(self._base_image, self._patch_set.board_tl_corner)
        if tl_err is None:
            return None
        tl_pos = (tl_pos[0] + self._patch_set.board_tl_corner.shape[0] + TR_X_PIXEL_SHIFT,
                  tl_pos[1] + self._patch_set.board_tl_corner.shape[1] + TR_Y_PIXEL_SHIFT)
        br_err, br_pos = FrameComparisonAnalyser.contains_patch(self._base_image, self._patch_set.board_br_corner)
        if br_err is None:
            return None
        br_pos = (br_pos[0] + BR_X_PIXEL_SHIFT, br_pos[1] + BR_Y_PIXEL_SHIFT)
        return tl_pos, br_pos

    def _compute_board_centers(self, board_corners: Tuple[Tuple[int, int], Tuple[int, int]]):
        tl_corner, br_corner = board_corners
        square_size = ((br_corner[0] - tl_corner[0]) / 8,
                       (br_corner[1] - tl_corner[1]) / 8)
        x_edges = np.linspace(tl_corner[0], br_corner[0], self.board_dims[0] + 1)
        y_edges = np.linspace(tl_corner[1], br_corner[1], self.board_dims[1] + 1)
        x_centers = np.round(x_edges[:-1] + square_size[0] / 2).astype(int)
        y_centers = np.round(y_edges[:-1] + square_size[1] / 2).astype(int)
        return np.meshgrid(x_centers, y_centers)

    def is_game_start_frame(self, sk_image: np.ndarray, mse_thresh: float = 1e-8):
        err, _ = FrameComparisonAnalyser.contains_patch(sk_image, self._patch_set.id_bomb_patch)
        return err is not None

    def initialize_coordinates(self, base_image: np.ndarray):
        self._base_image = base_image
        self._corner_positions = self._find_board_corners()
        tl, br = self._corner_positions
        self._background_board_image = self._base_image[tl[1]:br[1], tl[0]:br[0]].copy()
        self._center_x_coordinates, self._center_y_coordinates = self._compute_board_centers(self._corner_positions)
        self._base_image_center_mask = self._base_image[self._center_y_coordinates, self._center_x_coordinates]
        self._color_x_idx, self._color_y_idx = np.meshgrid(
            np.arange(self.board_dims[0] * self.board_dims[1]),
            np.arange(len(self._reference_color_map))
        )
    
    def board_from_image(self, sk_image: np.ndarray):
        #     sk_copy = sk_image.copy()
        #     sk_copy[self._center_y_coordinates + UNIFORM_Y_PIXEL_SHIFT,
        #             self._center_x_coordinates + UNIFORM_X_PIXEL_SHIFT] = np.array((255, 255, 255))
        #     sk_copy[self._corner_positions[0][1], self._corner_positions[0][0]] = np.array((255, 255, 255))
        #     sk_copy[self._corner_positions[1][1], self._corner_positions[1][0]] = np.array((255, 255, 255))
        #     plt.imsave('test.png', sk_copy[:, :, ::-1])
        test_image_centers = sk_image[self._center_y_coordinates, self._center_x_coordinates]
        cursor_centers = sk_image[self._center_y_coordinates + CURSOR_Y_PIXEL_SHIFT,
                                  self._center_x_coordinates + CURSOR_X_PIXEL_SHIFT]
        cursor_location_y, cursor_location_x = np.unravel_index(
            np.argmin(norm(cursor_centers - self._cursor_color, axis=2)), cursor_centers.shape[:2]
        )
        adjusted_center_y = self._center_y_coordinates[cursor_location_y, cursor_location_x] - 4
        adjusted_center_x = self._center_x_coordinates[cursor_location_y, cursor_location_x] - 4
        test_image_centers[test_image_centers == self._base_image_center_mask] = 0
        adjusted_center_color = sk_image[adjusted_center_y, adjusted_center_x]
        if np.all(self._base_image[adjusted_center_y, adjusted_center_x] == adjusted_center_color):
            test_image_centers[cursor_location_y, cursor_location_x] = 0
        else:
            test_image_centers[cursor_location_y, cursor_location_x] = adjusted_center_color
        # plt.imshow(test_image_centers[:, :, ::-1])
        # plt.show()
        current_shape = test_image_centers.shape
        test_image_centers = test_image_centers.reshape(current_shape[0] * current_shape[1], current_shape[2])
        color_diff = test_image_centers[self._color_x_idx] - self._reference_color_array[self._color_y_idx]
        closest = np.argmin(norm(color_diff, axis=2), axis=0)
        return closest.reshape(self.board_dims), (int(cursor_location_y), int(cursor_location_x))

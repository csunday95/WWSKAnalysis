
from typing import Tuple, Optional, List
import numpy as np
from numpy.linalg import norm
from frame_comparison_analyser import FrameComparisonAnalyser
from sk_board import SKBoard, SKSpaces
import matplotlib.pyplot as plt
import cv2 as cv

CORNER_TOLERANCE = 1e-9
SK_CORNER_X_OFFSET_PIXELS = 8  # make fractional later
SK_CORNER_Y_OFFSET_PIXELS = 8  # make fractional later


class SKBoardAnayliser:
    class SKBoardPatchSet:
        def __init__(self, board_tl_corner: np.ndarray, board_br_corner: np.ndarray):
            self.board_tl_corner, self.board_br_corner = board_tl_corner, board_br_corner

    def __init__(self,
                 patch_set: 'SKBoardAnayliser.SKBoardPatchSet',
                 base_image: np.ndarray,
                 reference_color_map: List[Tuple[SKSpaces, Tuple[int, int, int]]],
                 board_dims: Tuple[int, int]):
        self._base_image = base_image
        self._reference_color_map = reference_color_map
        self._reference_color_array = np.array([np.array(c[::-1]) for _, c in self._reference_color_map])
        self._reference_color_names = [name for name, _ in self._reference_color_map]
        self._corner_positions = None  # type: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
        self._center_x_coordinates = None  # type: Optional[np.ndarray]
        self._center_y_coordinates = None  # type: Optional[np.ndarray]
        self._base_image_center_mask = None  # type: Optional[np.ndarray]
        self._color_x_idx, self._color_y_idx = None, None
        self._board_dims = board_dims
        self._patch_set = patch_set

    def _find_board_corners(self):
        frame_analyser = FrameComparisonAnalyser(self._base_image)
        tl_err, tl_pos = frame_analyser.contains_patch(self._patch_set.board_tl_corner)
        if tl_err > CORNER_TOLERANCE:
            return None
        tl_pos = (tl_pos[0] + self._patch_set.board_tl_corner.shape[0],
                  tl_pos[1] + self._patch_set.board_tl_corner.shape[1])
        br_err, br_pos = frame_analyser.contains_patch(self._patch_set.board_br_corner)
        if br_err > CORNER_TOLERANCE:
            return None
        br_pos = (br_pos[0] + 10, br_pos[1] + 10)
        return tl_pos, br_pos

    @staticmethod
    def _compute_board_centers(board_corners: Tuple[Tuple[int, int], Tuple[int, int]]):
        tl_corner, br_corner = board_corners
        square_size = (int(round((br_corner[0] - tl_corner[0])) / 8),
                       int(round((br_corner[1] - tl_corner[1]) / 8)))
        x_edges = np.arange(tl_corner[0], br_corner[0] + square_size[0] // 2, square_size[0]) - 5
        y_edges = np.arange(tl_corner[1], br_corner[1] + square_size[1] // 2, square_size[1]) - 2
        x_centers = x_edges[:-1] + square_size[0] // 2
        y_centers = y_edges[:-1] + square_size[1] // 2
        return np.meshgrid(x_centers, y_centers)

    def initialize_coordinates(self):
        self._corner_positions = self._find_board_corners()
        self._center_x_coordinates, self._center_y_coordinates = self._compute_board_centers(self._corner_positions)
        self._base_image_center_mask = self._base_image[self._center_y_coordinates, self._center_x_coordinates]
        self._color_x_idx, self._color_y_idx = np.meshgrid(
            np.arange(self._board_dims[0] * self._board_dims[1]),
            np.arange(len(self._reference_color_map))
        )

    def board_from_image(self, sk_image: np.ndarray):
        centers_image = sk_image.copy()
        count = 0
        for x, y in zip(self._center_x_coordinates.flatten(), self._center_y_coordinates.flatten()):
            count += 1
            centers_image[y, x, :] = 255
        plt.imshow(centers_image[:, :, ::-1])
        cv.imwrite('test.png', centers_image)
        plt.show()

        test_image_centers = sk_image[self._center_y_coordinates, self._center_x_coordinates]
        plt.imshow(test_image_centers[:, :, ::-1])
        plt.show()
        test_image_centers[test_image_centers == self._base_image_center_mask] = 0
        current_shape = test_image_centers.shape
        plt.imshow(test_image_centers[:, :, ::-1])
        plt.show()
        test_image_centers = test_image_centers.reshape(current_shape[0] * current_shape[1], current_shape[2])

        color_diff = test_image_centers[self._color_x_idx] - self._reference_color_array[self._color_y_idx]
        closest = np.argmin(norm(color_diff, axis=2), axis=0)
        return closest.reshape(self._board_dims)

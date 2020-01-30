import cv2 as cv
from enum import Enum
import numpy as np
from skimage.metrics import structural_similarity as ssim


class FrameComparisonAnalyser:
    def __init__(self, base_image: np.ndarray):
        self.base_image = base_image
        self.base_dims = np.shape(self.base_image)

    def _compute_mse(self, candidate_image: np.ndarray):
        if self.base_dims != np.shape(candidate_image):
            raise ValueError('Image dimensions do not match')
        mse = np.sum((self.base_image - candidate_image) ** 2)
        return mse / float(self.base_dims[0] * self.base_dims[1])

    def compare(self, candidate_image: np.ndarray):
        mse = self._compute_mse(candidate_image)
        ssim_value = ssim(self.base_image, candidate_image, multichannel=True)
        return mse, ssim_value

    def contains_patch(self, patch_image: np.ndarray):
        template_result = cv.matchTemplate(self.base_image, patch_image, cv.TM_SQDIFF)
        normalized_image = np.zeros_like(template_result)
        cv.normalize(template_result, normalized_image, 0, 1, cv.NORM_MINMAX)
        min_max_result = cv.minMaxLoc(normalized_image)
        return min_max_result[0], min_max_result[2]

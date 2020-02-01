import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

TEMPLATE_MATCH_THRESHOLD = 100.0


class FrameComparisonAnalyser:
    def __init__(self, base_image: np.ndarray):
        self.base_image = base_image
        self.base_dims = np.shape(self.base_image)

    def _compute_mse(self, candidate_image: np.ndarray):
        mse = np.sum((self.base_image - candidate_image) ** 2)
        return mse / float(self.base_dims[0] * self.base_dims[1])

    def compare(self, candidate_image: np.ndarray, mse_threshold: float):
        plt.imshow(candidate_image[:, :, ::-1])
        plt.show()
        mse = self._compute_mse(candidate_image)
        if mse > mse_threshold:
            return mse, None
        ssim_value = ssim(self.base_image, candidate_image, multichannel=True)
        return mse, ssim_value

    @staticmethod
    def contains_patch(test_image: np.ndarray, patch_image: np.ndarray):
        template_result = cv.matchTemplate(test_image, patch_image, cv.TM_SQDIFF)
        if np.min(template_result) > TEMPLATE_MATCH_THRESHOLD:
            return None, None
        normalized_image = np.zeros_like(template_result)
        # cv.normalize(template_result, normalized_image, 0, 1, cv.NORM_MINMAX)
        # min_max_result = cv.minMaxLoc(normalized_image)
        min_max_result = cv.minMaxLoc(template_result)
        return min_max_result[0], min_max_result[2]

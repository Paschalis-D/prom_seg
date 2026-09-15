"""Preprocessing pipeline for GONG H-alpha images, following EdgeAttNet (Solomon et al. 2025, Sec. III.A):
1. Grayscale normalization to [0, 1]
2. Solar disk detection (Hough)
3. Disk masking
4. Radial flattening (median profile over normalized radius, disk pixels only)
5. Gaussian smoothing (3x3, sigma=0.7)
6. CLAHE within the disk
7. Off-disk zeroing
"""
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic


class ImgPreProcess:
    def __init__(self, image_path: str | Path, n_bins: int = 100,
                 clahe_clip: float = 2.0, clahe_grid: int = 16) -> None:
        self.image_path = Path(image_path)
        image = cv.imread(str(self.image_path), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {self.image_path}")
        self.image_u8 = image
        self.n_bins = n_bins
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    def run(self) -> np.ndarray:
        img = self.normalize(self.image_u8)                      # 1
        cx, cy, r = self.detect_disk(self.image_u8)              # 2
        mask = self.disk_mask(img.shape, cx, cy, r)              # 3
        flat = self.radial_flattening(img, mask, cx, cy, r)      # 4
        flat_u8 = self.to_uint8(flat, mask)
        smooth = self.gaussian_smoothing(flat_u8)                # 5
        enhanced = self.clahe_in_disk(smooth, mask)              # 6
        return self.off_disk_zeroing(enhanced, mask)             # 7

    @staticmethod
    def normalize(image_u8: np.ndarray) -> np.ndarray:
        return image_u8.astype(np.float32) / 255.0

    def detect_disk(self, image_u8: np.ndarray) -> tuple[int, int, int]:
        blurred = cv.medianBlur(image_u8, 5)
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT_ALT, dp=1, minDist=1000,
                                  param1=300, param2=0.85, minRadius=930, maxRadius=1000)
        if circles is None:
            raise RuntimeError(f"No solar disk found for {self.image_path}")
        cx, cy, r = np.around(circles[0, 0]).astype(int)
        return int(cx), int(cy), int(r)

    @staticmethod
    def disk_mask(shape, cx: int, cy: int, r: int) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv.circle(mask, (cx, cy), r, 255, thickness=-1)
        return mask > 0

    def radial_flattening(self, img: np.ndarray, mask: np.ndarray,
                          cx: int, cy: int, r: int) -> np.ndarray:
        yy, xx = np.indices(img.shape)
        rho = np.hypot(xx - cx, yy - cy) / r          # normalized radius, 1.0 at the limb

        # Radial profile from disk pixels ONLY
        stats = binned_statistic(rho[mask], img[mask], statistic="median",
                                 bins=self.n_bins, range=(0.0, 1.0))
        centers = 0.5 * (stats.bin_edges[:-1] + stats.bin_edges[1:])
        valid = ~np.isnan(stats.statistic)
        profile = savgol_filter(stats.statistic[valid], window_length=7, polyorder=3)

        # Keep float, guard against zero/negative background
        profile = np.clip(profile, 1e-3, None)
        background = np.interp(rho, centers[valid], profile).astype(np.float32)

        flat = np.zeros_like(img, dtype=np.float32)
        flat[mask] = img[mask] / background[mask]
        return flat

    @staticmethod
    def to_uint8(flat: np.ndarray, mask: np.ndarray,
                 p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
        lo, hi = np.percentile(flat[mask], [p_low, p_high])
        scaled = np.clip((flat - lo) / (hi - lo), 0.0, 1.0)
        out = (scaled * 255.0).astype(np.uint8)
        out[~mask] = 0
        return out

    @staticmethod
    def gaussian_smoothing(img_u8: np.ndarray) -> np.ndarray:
        return cv.GaussianBlur(img_u8, (3, 3), sigmaX=0.7, sigmaY=0.7)

    def clahe_in_disk(self, img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Fill off-disk with the on-disk median so limb tiles aren't dominated by zeros
        filled = img_u8.copy()
        filled[~mask] = np.uint8(np.median(img_u8[mask]))
        clahe = cv.createCLAHE(clipLimit=self.clahe_clip,
                               tileGridSize=(self.clahe_grid, self.clahe_grid))
        return clahe.apply(filled)

    @staticmethod
    def off_disk_zeroing(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = img_u8.copy()
        out[~mask] = 0
        return out


if __name__ == "__main__":
    train_dataset = Path("/home/pdompoudis/coding_projects/prom_seg/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    test_dataset = Path("/home/pdompoudis/coding_projects/prom_seg/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    processed_train = Path("/home/pdompoudis/coding_projects/prom_seg/data/train")
    processed_test = Path("/home/pdompoudis/coding_projects/prom_seg/data/test")

    for train_image in train_dataset.iterdir():
        print(f"Processing image {train_image.name}")
        processed = ImgPreProcess(train_image).run()
        out_path = processed_train.joinpath(train_image.name)
        cv.imwrite(str(out_path), processed)

    for test_image in test_dataset.iterdir():
        print(f"Processing image {test_image.name}")
        processed = ImgPreProcess(test_image).run()
        out_path = processed_train.joinpath(test_image.name)
        cv.imwrite(str(out_path), processed)

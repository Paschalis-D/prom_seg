"""File containing the preprocessing pipeline for the H-Alpha images. The workflow is:
1. Grayscale normalization
2. Solar Disk detection
3. Disk masking
4. Radial flattening
5. Gaussian smoothing
6. CLAHE
7. Off-disk zeroing
"""
from typing import Any

import cv2 as cv
import numpy as np
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter


from cv2.typing import NumPyArrayNumeric

class ImgPreProcess():
    def __init__(self, image) -> None:
        assert image is not None, "File could not be read"
        self.image: cv.Mat | NumPyArrayNumeric | None = cv.imread(image, cv.IMREAD_GRAYSCALE)
        self.norm_img = self.normalize(self.image)
        self.solar_disk = self.disk_masking(self.norm_img)
        self.masked_image = self.off_disk_zeroing(self.solar_disk)
        self.flattened_image = self.radial_flattening()
        
    def normalize(self, image: cv.Mat | NumPyArrayNumeric) -> cv.Mat | NumPyArrayNumeric:
        return cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
    def disk_masking(self, norm_image: cv.Mat | NumPyArrayNumeric):
        """Use the Hough Circle Transform algorithm to find the center and radius of the solar disk.
        The outputs will then be used for the radial flattening and off_disk_zeroing functions.
        """
        solar_disk = cv.HoughCircles(norm_image, cv.HOUGH_GRADIENT, dp=1, minDist=1000, param1=100, param2=100, minRadius=800, maxRadius=1024)
        return np.uint16(np.around(solar_disk))
        
    def radial_flattening(self):
        x = np.arange(0, self.masked_image.shape[1])
        y = np.arange(0, self.masked_image.shape[0])
        cx = self.solar_disk[0,0,0]
        cy = self.solar_disk[0,0,1]
        r = self.solar_disk[0,0,2]
        image_mesh = np.meshgrid(x, y)
        radial_coord_mesh = np.sqrt((image_mesh[0] - cx)**2 + (image_mesh[1] - cy)**2)
        
        bins = binned_statistic(x = radial_coord_mesh.flatten(), values = self.masked_image.flatten(), statistic='median', bins=100)
        smoothed_bins = savgol_filter(bins.statistic, window_length=7, polyorder=3)
        bin_centers = bins.bin_edges[:-1] + np.diff(bins.bin_edges) / 2
        background = np.interp(radial_coord_mesh.flatten(), bin_centers, smoothed_bins)
        background = np.reshape(background, radial_coord_mesh.shape)
        
        return self.masked_image/background

    def gaussian_smoothing(self):
        pass
    
    def clahe(self):
        pass
    
    def off_disk_zeroing(self, solar_disk):
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        mask = cv.circle(mask, (solar_disk[0,0,0], solar_disk[0,0,1]), solar_disk[0,0,2], (255, 255, 255), -1)
        masked_image = cv.bitwise_and(self.norm_img, mask)
        return masked_image
    
if __name__ == '__main__':
    
    image_path = "D:\\Personal\\solar_filament_competition\\prom_seg\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\train_images\\20110109104734Ch.jpeg"
    preprocessor = ImgPreProcess(image_path)
    

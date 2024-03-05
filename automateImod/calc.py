import numpy as np
from typing import Tuple

def median_residual(f):
    contour_resid = f[:, [0, 6]]
    all_residuals = contour_resid[:, 1]
    median_residual = np.median(all_residuals, axis=0)
    return contour_resid, median_residual


def get_thickness(unbinned_voxel_size, binval):
    curr_vox = unbinned_voxel_size * float(binval)
    slab_thickness = int(2500 / curr_vox)
    return slab_thickness

def calculate_frame_statistics(img_data: np.ndarray) -> Tuple[float, float]:
    """Calculate the mean and standard deviation of an image's pixel intensities."""
    mean = np.mean(img_data)
    std_dev = np.std(img_data)
    return mean, std_dev

def normalize(data: np.array) -> np.array:
    return (data-np.min(data))/(np.max(data)-np.min(data))

def robust_normalization(data: np.array) -> np.array:
    p5 = np.percentile(data, q=5)
    p95 = np.percentile(data[data != 0], q=95)
    median = np.median(data)
    return (data - median) / (p95 - p5)

def find_symmetric_tilt_reference(tilt_angles):
    # Assuming tilt_angles is a sorted list of tilt angles
    num_tilts = len(tilt_angles)
    midpoint_idx = num_tilts // 2

    if num_tilts % 2 == 0:  # Even number of frames
        # Calculate average of the two middle tilt angles
        midpoint_angle = (tilt_angles[midpoint_idx - 1] + tilt_angles[midpoint_idx]) / 2.0
    else:  # Odd number of frames
        midpoint_angle = tilt_angles[midpoint_idx]

    # Find the frame index with tilt angle closest to the midpoint_angle
    closest_idx = min(range(num_tilts), key=lambda i: abs(tilt_angles[i] - midpoint_angle))
    return closest_idx

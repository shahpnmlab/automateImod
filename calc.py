import numpy as np


def median_residual(f):
    contour_resid = f[:, [0, 6]]
    all_residuals = contour_resid[:, 1]
    median_residual = np.median(all_residuals, axis=0)
    return contour_resid, median_residual


def normalised_sums(im_data):
    min_values = np.min(im_data, axis=(1, 2), keepdims=True)
    max_values = np.max(im_data, axis=(1, 2), keepdims=True)

    normalized_stack = (im_data - min_values) / (max_values - min_values)
    normalized_sums = np.sum(normalized_stack, axis=(1, 2))
    return normalized_sums


def get_thickness(unbinned_voxel_size, binval):
    curr_vox = unbinned_voxel_size * float(binval)
    slab_thickness = int(2500 / curr_vox)
    return slab_thickness

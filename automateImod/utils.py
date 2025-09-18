import shutil
import subprocess
import mrcfile
import numpy as np
from pathlib import Path
import logging

import automateImod.calc as calc
import automateImod.pio as io


def setup_logging(log_file_path: Path):
    """Configure logging to write to a file."""
    # Get a logger specific to the tilt series (using the file path as a key)
    logger = logging.getLogger(str(log_file_path))
    logger.setLevel(logging.INFO)

    # Prevent handlers from being added multiple times
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

    return logger


def detect_large_shifts_afterxcorr(
    coarse_align_prexg,
    pixel_size_nm,
    image_size,
    logger,
    min_fov_fraction=0.8,
    max_shift_nm=None,  # New parameter
    max_shift_rate=50.0,
    outlier_threshold=3.0,
    min_acceptable_overlap=0.5,
    use_statistical_analysis=True,  # New parameter
):
    """
    Robust detection of problematic shifts using multiple criteria.

    Args:
        coarse_align_prexg: Path to prexg file
        pixel_size_nm: Pixel size in nm
        image_size: (width, height) in pixels
        logger: The logger object.
        min_fov_fraction: Minimum overlap for normal operation (default: 0.8)
        max_shift_nm: Maximum acceptable shift in nm (default: None, auto-calculated)
        max_shift_rate: Maximum acceptable shift rate in nm/degree (default: 50.0)
        outlier_threshold: Statistical outlier threshold in MAD units (default: 3.0)
        min_acceptable_overlap: Absolute minimum overlap required (default: 0.5)
        use_statistical_analysis: Whether to use statistical outlier detection (default: True)

    Returns:
        list: Indices of problematic views
    """
    prexg_data = np.loadtxt(coarse_align_prexg)
    shifts_pixels = prexg_data[:, -2:]
    shifts_nm = shifts_pixels * pixel_size_nm

    # Calculate shift magnitudes
    shift_magnitudes = np.sqrt(shifts_nm[:, 0] ** 2 + shifts_nm[:, 1] ** 2)

    # Image dimensions
    width_nm = image_size[0] * pixel_size_nm
    height_nm = image_size[1] * pixel_size_nm

    # Auto-calculate max_shift_nm if not provided
    # Default to 20% of the smaller dimension
    if max_shift_nm is None:
        max_shift_nm = 0.2 * min(width_nm, height_nm)

    # Calculate overlap fractions
    x_overlaps = 1 - np.abs(shifts_nm[:, 0]) / width_nm
    y_overlaps = 1 - np.abs(shifts_nm[:, 1]) / height_nm
    min_overlaps = np.minimum(x_overlaps, y_overlaps)

    problematic_views = set()

    # Criterion 1: Absolute minimum overlap
    # This catches catastrophic failures
    abs_bad_indices = np.where(min_overlaps < min_acceptable_overlap)[0]
    problematic_views.update(abs_bad_indices)

    # Criterion 2: Maximum shift magnitude
    # Simple absolute threshold
    max_shift_indices = np.where(shift_magnitudes > max_shift_nm)[0]
    problematic_views.update(max_shift_indices)

    # Criterion 3: Shift rate analysis
    # Shifts should change smoothly with tilt angle
    large_jump_indices = []  # Initialize for diagnostic printing
    if len(shifts_nm) > 2:
        # Calculate shift differences between consecutive views
        shift_diffs = np.diff(shifts_nm, axis=0)
        shift_diff_magnitudes = np.sqrt(shift_diffs[:, 0] ** 2 + shift_diffs[:, 1] ** 2)

        # Assuming ~1-2 degree tilt increment
        # Views with sudden large jumps are suspicious
        jump_threshold = max_shift_rate * 2.0  # Allow 2 degrees worth of shift
        large_jump_indices = np.where(shift_diff_magnitudes > jump_threshold)[0]

        # Mark both views involved in the jump
        for idx in large_jump_indices:
            problematic_views.add(idx)
            problematic_views.add(idx + 1)

    # Criterion 4: Statistical outliers (optional)
    statistical_outlier_indices = []  # Initialize for diagnostic printing
    if use_statistical_analysis:
        # Only apply if we have enough "good" views
        # Create a list of indices that are not yet problematic
        current_good_indices = list(set(range(len(min_overlaps))) - problematic_views)
        remaining_overlaps = min_overlaps[current_good_indices]

        if len(remaining_overlaps) > 5:  # Need enough data for statistics
            # Use robust statistics on the remaining views
            median_overlap = np.median(remaining_overlaps)
            mad = np.median(
                np.abs(remaining_overlaps - median_overlap)
            )  # Median Absolute Deviation

            # Only flag statistical outliers if they also violate FOV criterion
            if (
                mad > 0
            ):  # Avoid division by zero if all remaining overlaps are identical
                # Convert MAD to estimated standard deviation for Gaussian data
                # The factor 1.4826 is specific to normally distributed data
                # but is a common heuristic.
                threshold = median_overlap - outlier_threshold * mad * 1.4826

                # Find outliers among ALL views, not just current_good_indices,
                # but only those that also violate the min_fov_fraction.
                # This ensures we only flag views that are both statistically unusual
                # AND have low overlap.
                statistical_outlier_indices_bool = (min_overlaps < threshold) & (
                    min_overlaps < min_fov_fraction
                )
                statistical_outlier_indices = np.where(
                    statistical_outlier_indices_bool
                )[0]
                problematic_views.update(statistical_outlier_indices)

    # Criterion 5: Consistency check
    # If a view is surrounded by good views but has large shift, it's likely bad
    # This requires at least 3 views to have a "middle" view, and 2 more for context, total 5.
    # The original code had len(shifts_nm) > 4, which means at least 5 views.
    if len(shifts_nm) > 4:  # Ensure there are enough views for prev, curr, next logic
        for i in range(
            1, len(shifts_nm) - 1
        ):  # Iterate from the second to the second-to-last view
            if i not in problematic_views:  # Only consider views not already flagged
                # Check if this view's shift is very different from neighbors
                # This check assumes that problematic_views does not change during this loop
                # which is true for this implementation.

                # Check if neighbors are already problematic. If so, this view is harder to judge.
                # For simplicity, the original logic didn't explicitly exclude if neighbors were bad.
                # Let's stick to that unless it proves problematic.
                prev_shift = shifts_nm[i - 1]
                curr_shift = shifts_nm[i]
                next_shift = shifts_nm[i + 1]

                # Expected shift based on linear interpolation of immediate neighbors
                expected_shift = (
                    prev_shift + next_shift
                ) / 2.0  # Use floating point division
                deviation = np.linalg.norm(curr_shift - expected_shift)

                # If deviation is large AND overlap is below normal threshold (min_fov_fraction)
                # max_shift_rate here is used as a proxy for large deviation.
                # This might need tuning or a separate parameter.
                if deviation > max_shift_rate and min_overlaps[i] < min_fov_fraction:
                    problematic_views.add(i)

    # Print diagnostic information
    # Use Path(coarse_align_prexg).name for cleaner output, as in the example
    logger.info(f"\nShift analysis for {Path(coarse_align_prexg).name}:")
    logger.info(f"  Total views: {len(shifts_nm)}")
    logger.info(f"  Maximum shift threshold: {max_shift_nm:.1f} nm")
    logger.info(f"  Problematic views found: {len(problematic_views)}")
    if len(problematic_views) > 0:
        logger.info(f"  Indices: {sorted(list(problematic_views))}")
        # For breakdown, we need to re-evaluate conditions if we want exact counts per criterion
        # as a view could be caught by multiple. The current *_indices lists are fine for this.
        logger.info(f"  Criteria breakdown:")
        # abs_bad_indices was defined earlier
        if len(abs_bad_indices) > 0:
            logger.info(
                f"    - Absolute minimum overlap violations ({min_acceptable_overlap*100}%): {len(abs_bad_indices)} views"
            )
        if len(max_shift_indices) > 0:
            logger.info(
                f"    - Maximum shift exceeded ({max_shift_nm:.1f} nm): {len(max_shift_indices)} views"
            )
            for (
                idx
            ) in max_shift_indices:  # Iterate through indices caught by this criterion
                if (
                    idx in problematic_views
                ):  # Check if it's still considered problematic overall
                    logger.info(
                        f"      View {idx}: Shift {shift_magnitudes[idx]:.1f} nm"
                    )

        # large_jump_indices was defined earlier for Criterion 3
        # The original code checked len(shifts_nm) > 2 before printing this part.
        if len(shifts_nm) > 2 and len(large_jump_indices) > 0:
            # This counts transitions (pairs of views), not individual views.
            logger.info(
                f"    - Large shift jumps detected: {len(large_jump_indices)} transitions"
            )
            # The original code did not print details for each jump, so we won't either.

        # statistical_outlier_indices was initialized for Criterion 4
        if use_statistical_analysis and len(statistical_outlier_indices) > 0:
            logger.info(
                f"    - Statistical outliers: {len(statistical_outlier_indices)} views"
            )

    return sorted(list(problematic_views))


def remove_bad_tilts(ts: io.TiltSeries, im_data, pixel_nm, bad_idx):
    angpix = pixel_nm * 10
    original_rawtlt_angles = np.loadtxt(ts.get_rawtlt_path())

    # Ensure bad_idx are within the range of im_data
    bad_idx = [idx for idx in bad_idx if idx < len(im_data)]

    mask = np.ones(len(im_data), dtype=bool)
    mask[bad_idx] = False
    cleaned_mrc = im_data[mask]

    # Ensure the mask is the same length as original_rawtlt_angles
    if len(mask) > len(original_rawtlt_angles):
        mask = mask[: len(original_rawtlt_angles)]
    elif len(mask) < len(original_rawtlt_angles):
        mask = np.pad(
            mask,
            (0, len(original_rawtlt_angles) - len(mask)),
            "constant",
            constant_values=True,
        )

    cleaned_ts_rawtlt = original_rawtlt_angles[mask]

    original_ts_file = ts.get_mrc_path()
    original_ts_rawtlt = ts.get_rawtlt_path()

    # Backup original TS data
    mrcfile.write(
        name=f"{original_ts_file}~", data=im_data, voxel_size=angpix, overwrite=True
    )
    np.savetxt(fname=f"{original_ts_rawtlt}~", X=original_rawtlt_angles, fmt="%0.2f")

    # Write new TS data
    mrcfile.write(
        name=f"{original_ts_file}", data=cleaned_mrc, voxel_size=angpix, overwrite=True
    )
    np.savetxt(fname=f"{original_ts_rawtlt}", X=cleaned_ts_rawtlt, fmt="%0.2f")

    # Update the TiltSeries object
    ts.tilt_angles = cleaned_ts_rawtlt
    ts.removed_indices = (
        sorted(set(ts.removed_indices + bad_idx))
        if hasattr(ts, "removed_indices")
        else sorted(bad_idx)
    )


def get_alignment_error(tilt_dir_name, logger):
    known_unknown_ratio = None
    resid_err = None
    sd = None

    try:
        with open(f"{tilt_dir_name}/align_patch.log", "r") as f_in:
            for line in f_in:
                if "Ratio of total measured values to all unknowns" in line:
                    known_unknown_ratio = float(line.split("=")[-1])
                if "Residual error mean and sd" in line:
                    a1 = line.split()
                    resid_err = float(a1[5])
                    sd = a1[6]

        if known_unknown_ratio is None or resid_err is None or sd is None:
            logger.warning("Could not find all alignment statistics in the log file.")

        return known_unknown_ratio, resid_err, sd

    except Exception as e:
        logger.error(f"Error reading alignment log: {e}")
        return None, None, None


def write_ta_coords_log(tilt_dir_name, logger):
    with open(f"{str(tilt_dir_name)}/taCoordinates.log", "w") as ali_log:
        sbp_cmd = ["alignlog", "-c", f"{str(tilt_dir_name)}/align_patch.log"]
        write_taCoord_log = subprocess.run(
            sbp_cmd, stdout=ali_log, stderr=subprocess.PIPE, text=True
        )
        logger.info(write_taCoord_log.stdout)
        logger.error(write_taCoord_log.stderr)


def improve_bad_alignments(tilt_dir_name, tilt_name, logger):
    tilt_dir_name = str(tilt_dir_name)
    mod_file = tilt_dir_name + "/" + tilt_name + ".fid"
    mod2txt = tilt_dir_name + "/" + tilt_name + ".txt"
    txt4seed = tilt_dir_name + "/" + tilt_name + ".seed.txt"

    align_log_vals = np.loadtxt(tilt_dir_name + "/" + "taCoordinates.log", skiprows=2)

    contour_resid, median_residual = calc.median_residual(align_log_vals)

    goodpoints = np.where(contour_resid[:, [1]] <= np.around(median_residual))
    goodpoints = goodpoints[0] + 1

    # Convert the fiducial model file to a text file and readit in
    model2point_cmd = ["model2point", "-contour", mod_file, mod2txt]
    result = subprocess.run(model2point_cmd, capture_output=True, text=True)
    logger.info(f"model2point stdout: {result.stdout}")
    if result.stderr:
        logger.error(f"model2point stderr: {result.stderr}")

    fid_text = np.loadtxt(mod2txt)
    new_good_contours = np.empty((0, 4))

    for i in range(goodpoints.shape[0]):
        a = fid_text[fid_text[:, 0] == goodpoints[i]]
        new_good_contours = np.append(new_good_contours, a, axis=0)

    number_of_new_contours = np.unique(new_good_contours[:, 0]).shape[0]
    number_of_new_contours = np.arange(number_of_new_contours) + 1

    repeats = np.unique(new_good_contours[:, -1]).shape[0]
    new_contour_id = np.repeat(number_of_new_contours, repeats)
    new_contour_id = new_contour_id.reshape(new_contour_id.shape[0], 1)
    new_good_contours = np.hstack((new_contour_id, new_good_contours[:, 1:]))

    np.savetxt(
        txt4seed, new_good_contours, fmt=" ".join(["%d"] + ["%0.3f"] * 2 + ["%d"])
    )

    point_to_model_cmd = [
        "point2model",
        "-open",
        "-circle",
        "6",
        "-image",
        f"{tilt_dir_name}/{tilt_name}_preali.mrc",
        txt4seed,
        mod_file,
    ]

    result = subprocess.run(point_to_model_cmd, capture_output=True, text=True)
    logger.info(f"point2model stdout: {result.stdout}")
    if result.stderr:
        logger.error(f"point2model stderr: {result.stderr}")


def detect_dark_tilts(
    ts_data, ts_tilt_angles, logger, brightness_factor=0.65, variance_factor=0.1
):
    # Identify the reference image, typically at or near 0 tilt
    tilt_series_mid_point_idx = calc.find_symmetric_tilt_reference(ts_tilt_angles)
    normalized_images = calc.normalize(ts_data)
    reference_image = normalized_images[tilt_series_mid_point_idx, :, :]

    # Calculate mean intensity and standard deviation for the reference image
    reference_mean_intensity = np.mean(reference_image)
    reference_std_deviation = np.std(reference_image)

    # Calculate dynamic thresholds
    threshold_intensity = reference_mean_intensity * brightness_factor
    threshold_variance = reference_std_deviation * variance_factor

    # Calculate mean intensities and standard deviations for all tilts
    mean_intensities = np.mean(normalized_images, axis=(1, 2))
    std_deviations = np.std(normalized_images, axis=(1, 2))

    # Identify dark tilts based on mean intensity and standard deviation
    dark_frame_indices = np.where(
        (mean_intensities < threshold_intensity) & (std_deviations < threshold_variance)
    )[0].tolist()

    return dark_frame_indices


def swap_fast_slow_axes(tilt_dirname, tilt_name):
    (
        d,
        pixel_nm,
        _,
        _,
    ) = io.read_mrc(f"{tilt_dirname}/{tilt_name}.rec")
    d = np.swapaxes(d, 0, 1)
    mrcfile.write(
        f"{tilt_dirname}/{tilt_name}.rec",
        data=d,
        voxel_size=pixel_nm * 10,
        overwrite=True,
    )


def match_partial_filename(string_to_match, target_string, logger):
    if string_to_match in target_string:
        return True
    else:
        logger.error("Could not match string. Check if the file exists.")
        return False


def remove_xml_files(xml_file_path, logger):
    if xml_file_path.exists():
        if xml_file_path.exists():
            backup_file_path = xml_file_path.with_suffix(".xml.bkp")
            shutil.move(xml_file_path, backup_file_path)
        else:
            logger.warning(f"XML file {xml_file_path} not found.")


if __name__ == "__main__":
    ts_object = io.TiltSeries(
        path_to_ts_data="/Users/ps/data/wip/automateImod/example_data/Frames/imod/",
        path_to_mdoc_data="/Users/ps/data/wip/automateImod/example_data/Frames/mdoc/",
        basename="map-26-A4_ts_002",
        tilt_axis_ang=60,
        binval=10,
        patch_size=10,
    )
    a = detect_dark_tilts(ts_object)
    print(a)
